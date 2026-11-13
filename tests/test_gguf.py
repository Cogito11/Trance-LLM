"""GGUF header reading and discovery, using hand-built files (no gguf package needed)."""
from __future__ import annotations

import os
import struct
import tempfile
import unittest
from pathlib import Path

from gui.gguf import discover_gguf, read_gguf_info


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv_str(key, value): return _s(key) + struct.pack("<I", 8) + _s(value)
def kv_u32(key, value): return _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)
def kv_u32_array(key, values): return _s(key) + struct.pack("<I", 9) + struct.pack("<IQ", 4, len(values)) + b"".join(struct.pack("<I", v) for v in values)
def kv_str_array(key, values): return _s(key) + struct.pack("<I", 9) + struct.pack("<IQ", 8, len(values)) + b"".join(_s(v) for v in values)


def gguf_bytes(*kvs: bytes, version=3) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", version, 0, len(kvs)) + b"".join(kvs)


def write(path: Path, *kvs: bytes, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(gguf_bytes(*kvs, **kw)); return path


def chat_model(name="Test Model", arch="llama", ctx=4096, file_type=15, label=None):
    kvs = [kv_str("general.architecture", arch), kv_str("general.name", name), kv_u32("general.file_type", file_type),
           kv_u32(f"{arch}.context_length", ctx), kv_u32(f"{arch}.block_count", 4)]
    if label: kvs.append(kv_str("general.size_label", label))
    return kvs


class ReadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_the_fields_the_picker_needs(self):
        path = write(self.root / "m.gguf", *chat_model("Qwen Tiny", "qwen3", 32768, 15, "0.6B"),
                     kv_str_array("tokenizer.ggml.tokens", ["a", "b"]))
        info = read_gguf_info(path)
        self.assertEqual((info["title"], info["architecture"], info["context_length"], info["quantization"], info["size_label"]),
                         ("Qwen Tiny", "qwen3", 32768, "Q4_K_M", "0.6B"))
        self.assertEqual((info["kind"], info["name"], info["size_bytes"]), ("gguf", "m.gguf", path.stat().st_size))

    def test_skips_arrays_before_the_keys_it_wants_and_stops_at_the_tokenizer(self):
        path = write(self.root / "m.gguf", kv_u32_array("some.array", [1, 2, 3]), kv_str_array("some.strings", ["x", "yy"]),
                     *chat_model(), kv_str_array("tokenizer.ggml.tokens", ["zzzz"] * 1000))
        self.assertEqual(read_gguf_info(path)["context_length"], 4096)

    def test_unknown_quantization_and_missing_name_fall_back_gracefully(self):
        path = write(self.root / "weird-name.gguf", kv_str("general.architecture", "llama"), kv_u32("general.file_type", 999))
        info = read_gguf_info(path)
        self.assertIsNone(info["quantization"]); self.assertEqual(info["title"], "weird-name"); self.assertEqual(info["context_length"], 0)

    def test_rejects_things_that_are_not_usable_models(self):
        (self.root / "empty.gguf").write_bytes(b"")
        (self.root / "text.gguf").write_text("hello world, this is not a model file at all")
        (self.root / "trunc.gguf").write_bytes(gguf_bytes(*chat_model())[:40])
        write(self.root / "v1.gguf", *chat_model(), version=1)
        write(self.root / "noarch.gguf", kv_str("general.name", "x"))
        write(self.root / "proj.gguf", *chat_model(arch="clip"))
        for name in ("empty", "text", "trunc", "v1", "noarch", "proj"):
            with self.assertRaises(ValueError, msg=name): read_gguf_info(self.root / f"{name}.gguf")
        with self.assertRaises(ValueError): read_gguf_info(self.root / "missing.gguf")


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_nested_models_sorted_by_title_and_ignores_the_rest(self):
        write(self.root / "a" / "b" / "zeta.gguf", *chat_model("Zeta"))
        write(self.root / "alpha.gguf", *chat_model("alpha"))
        write(self.root / "mmproj-vision.gguf", *chat_model(arch="clip"))
        (self.root / "notes.txt").write_text("hi"); (self.root / "broken.gguf").write_bytes(b"nope")
        found = discover_gguf([self.root])
        self.assertEqual([m["title"] for m in found], ["alpha", "Zeta"])

    def test_only_the_first_part_of_a_split_model_is_listed(self):
        for part in (1, 2, 3):
            write(self.root / f"big-{part:05d}-of-00003.gguf", *chat_model("Big"))
        self.assertEqual([Path(m["path"]).name for m in discover_gguf([self.root])], ["big-00001-of-00003.gguf"])

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_symlinked_duplicates_are_listed_once(self):
        real = write(self.root / "blobs" / "real.gguf", *chat_model("Real"))
        (self.root / "snap").mkdir(); (self.root / "snap" / "link.gguf").symlink_to(real)
        self.assertEqual(len(discover_gguf([self.root])), 1)

    def test_missing_folders_limits_and_depth_are_handled(self):
        for n in range(5): write(self.root / f"m{n}.gguf", *chat_model(f"M{n}"))
        write(self.root / "1" / "2" / "3" / "deep.gguf", *chat_model("Deep"))
        self.assertEqual(discover_gguf([self.root / "nope"]), [])
        self.assertEqual(len(discover_gguf([self.root], limit=3)), 3)
        self.assertNotIn("Deep", [m["title"] for m in discover_gguf([self.root], max_depth=2)])


if __name__ == "__main__":
    unittest.main()
