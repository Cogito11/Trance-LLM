import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_trance12_curriculum import assemble, iter_stories, key, reading_turns, story_turns
from prepare_trance12 import parse, raw_token_bound, render


def story(name):
    return (f"One morning, {name} found a little red box beside the garden gate. "
            "Inside the box was a small toy bird with a broken wing. "
            "The child took it home and asked a friend to help repair it. "
            "They worked together until the bird looked as good as new. "
            "Then they put it on the shelf and smiled.")


def qa(context, question, answer):
    return {"context": context, "question": question,
            "answers": {"text": [answer], "answer_start": [context.index(answer)]}}


class CurriculumTests(unittest.TestCase):
    def test_complete_story_and_context_bound(self):
        text = story("Maya")
        turns = story_turns(text)
        self.assertIsNotNone(turns)
        self.assertTrue(text.endswith(turns[-1][1]))
        self.assertTrue(turns[-1][1].endswith("Then they put it on the shelf and smiled."))
        self.assertLessEqual(raw_token_bound(turns), 1024)
        self.assertIsNone(story_turns(text * 10))
        self.assertIsNone(story_turns(text[:-1]))
        self.assertIsNone(story_turns(text + " <assistant>"))
        self.assertIsNone(story_turns("u don't have to worry. " + text))

    def test_upstream_file_without_final_delimiter(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"stories.txt"
            path.write_text(story("Maya") + "\n<|endoftext|>\n" + story("Owen"))
            self.assertEqual([s.strip() for s in iter_stories(path)], [story("Maya"), story("Owen")])

    def test_reading_requires_exact_answer_span(self):
        row = qa("Mara put the book in the kitchen drawer.",
                 "Where did Mara put the book?", "kitchen drawer")
        turns = reading_turns(row)
        self.assertEqual(turns[-1], ("assistant", "kitchen drawer"))
        self.assertIn(row["context"], turns[0][1])
        row["answers"]["answer_start"][0] = 0
        self.assertIsNone(reading_turns(row))
        row = qa("A " * 600 + "red book.", "What was red?", "book")
        self.assertIsNone(reading_turns(row))

    def test_reproducible_weights_and_split_isolation(self):
        train_qa = qa("Mara put the book in the kitchen drawer.",
                      "Where did Mara put the book?", "kitchen drawer")
        valid_qa = qa("Owen left the cup on the silver tray.",
                      "Where did Owen leave the cup?", "silver tray")
        leaked_context = qa(valid_qa["context"], "What did Owen leave?", "cup")
        train_legacy = [("user", "Please suggest a relaxing activity for a quiet evening."),
                        ("assistant", "Read a book in a comfortable chair.")]
        valid_legacy = [("user", "Please suggest a useful activity for a rainy morning."),
                        ("assistant", "Organize a shelf.")]
        eval_prompt = "Please write the exact greeting from the evaluation."
        eval_record = [("user", eval_prompt), ("assistant", "Hello.")]

        def build():
            return assemble(
                [story(n) for n in ("Maya", "Ellis", "Nora", "Arlo", "Owen", "Maya")],
                [story("Owen")],
                [train_qa, train_qa, valid_qa, leaked_context], [valid_qa],
                [train_legacy, valid_legacy, eval_record], [valid_legacy], [eval_prompt],
                story_limit=2, validation_caps=(1, 1, 1), seed=19)

        files, stats = build()
        self.assertEqual((files, stats), build())
        self.assertEqual(stats["stories_heldout_removed"], 1)
        self.assertEqual(stats["reading_heldout_removed"], 2)
        self.assertEqual(stats["reading_selected_unique"], 1)
        self.assertEqual(stats["legacy_reserved_removed"], 2)
        self.assertEqual(stats["training_records"], 5)
        train_records = files["train.txt"].decode().split("<eot>")[:-1]
        val_records = files["validation.txt"].decode().split("<eot>")[:-1]
        self.assertFalse({key(r) for r in train_records} & {key(r) for r in val_records})
        for record in train_records + val_records:
            self.assertLessEqual(raw_token_bound(parse(record)), 1024)
        self.assertEqual(files["train.txt"].decode().count(render(reading_turns(train_qa))), 2)
        self.assertNotIn(eval_prompt, files["train.txt"].decode())


if __name__ == "__main__":
    unittest.main()
