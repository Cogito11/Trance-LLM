"""Focused regression checks for data leakage, context bounds and rendering."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_trance12 as p


class PreparationTests(unittest.TestCase):
    def test_preserves_task_instruction(self):
        turns = p.normalize([
            {"role": "system", "content": "Rewrite concisely."},
            {"role": "user", "content": "Some text"},
            {"role": "assistant", "content": "Text"},
        ])
        self.assertEqual(turns[0][1], "Rewrite concisely.\n\nInput:\nSome text")

    def test_atomic_marker_bound_counts_utf8(self):
        turns = [("user", "é"), ("assistant", "こんにちは")]
        # C raw_file_list collapses all three markers and skips the EOT newline.
        expected = len("\né\n\n\nこんにちは\n\n".encode("utf-8")) + 3
        self.assertEqual(p.raw_token_bound(turns), expected)
        self.assertEqual(p.parse(p.render(turns).split("<eot>")[0]), turns)

    def test_rejects_orphan_turn_and_literal_control_marker(self):
        self.assertIsNone(p.normalize([{"role": "assistant", "content": "Orphan"}]))
        self.assertIsNone(p.normalize([
            {"role": "user", "content": "Print <eot>"},
            {"role": "assistant", "content": "Okay"},
        ]))

    def test_common_greetings_do_not_erase_casual_data(self):
        self.assertIsNone(p.heldout_prompt_key("Hi there!"))
        self.assertIsNotNone(p.heldout_prompt_key("Explain why the sky appears blue during the daytime."))

    def test_global_dedup_heldout_prompts_and_complete_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            for name in ("smoltalk", "ultrachat"):
                (data / name).mkdir(parents=True)
            duplicate = [("user", "Shared train prompt"), ("assistant", "Same answer")]
            heldout = [("user", "Explain why the sky appears blue during the daytime."), ("assistant", "Heldout answer")]
            prefix = [("user", "First question"), ("assistant", "Complete answer")]
            long = prefix + [("user", "Follow-up"), ("assistant", "x"*200)]
            (data / "smoltalk/train.txt").write_text(p.render(duplicate) + p.render(long) + p.render([
                ("user", "EXPLAIN why the sky appears   blue during the daytime."), ("assistant", "Different train answer")]))
            (data / "ultrachat/train.txt").write_text(p.render(duplicate) + p.render([
                ("user", "Overlong initial question"), ("assistant", "x"*200)]))
            (data / "smoltalk/validation.txt").write_text(p.render(heldout))
            (data / "ultrachat/validation.txt").write_text(p.render([
                ("user", "Another heldout"), ("assistant", "Test answer")]))
            with patch.multiple(p, ROOT=root, OUT=data / "trance12", NEW=data / "new",
                                SOURCES={"smoltalk": (0, 2), "ultrachat": (0, 1)}, CONTEXT=100), contextlib.redirect_stdout(io.StringIO()):
                p.build()
            train = (data / "trance12/train.txt").read_text()
            self.assertNotIn("Different train answer", train)
            self.assertNotIn("Follow-up", train)
            self.assertEqual(train.count("Complete answer"), 2)
            self.assertEqual(train.count("Same answer"), 2)
            for text in p.iter_text(data / "trance12/train.txt"):
                self.assertLessEqual(p.raw_token_bound(p.parse(text)), 100)
            manifest = json.loads((data / "trance12/manifest.json").read_text())
            self.assertEqual(manifest["sources"]["ultrachat"]["train_duplicate"], 1)
            self.assertEqual(manifest["sources"]["smoltalk"]["train_heldout_prompt_removed"], 1)
            self.assertEqual(manifest["sources"]["ultrachat"]["train_too_long"], 1)


if __name__ == "__main__":
    unittest.main()
