#!/usr/bin/env python3
"""Independently validate final files and config without starting training."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "configs/trance1.2_30m.json").read_text())
manifest = json.loads((ROOT / "data/trance12/manifest.json").read_text())
assert config["vocab_size"] == 1024
assert config["context_length"] == manifest["context"] == 1024
assert config["embedding_dim"] % config["attention_heads"] == 0
assert "resume_model" not in config
assert config["output_model"] != "models/trance1.1_21m.bin"
prompts = {}
unique = {}
for split, key in (("train", "train_data"), ("validation", "validation_data")):
    path = ROOT / config[key]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["files"][split]["sha256"]
    assert len(data) == manifest["files"][split]["bytes"]
    records = data.decode("utf-8").split("<eot>\n")
    assert not records.pop().strip()
    assert len(records) == manifest["files"][split]["conversations"]
    prompts[split], unique[split] = set(), set()
    max_bound = 0
    for record in records:
        turns = re.split(r"(?m)^<(user|assistant)>\n", record)
        assert not turns.pop(0).strip()
        assert len(turns) >= 4 and len(turns) % 4 == 0
        roles = turns[::2]
        assert roles == ["user", "assistant"] * (len(roles)//2)
        for text in turns[1::2]:
            assert text.strip() and not any(tag in text.lower() for tag in ("<user>", "<assistant>", "<eot>"))
        raw = record.replace("<user>", "X").replace("<assistant>", "X").encode("utf-8")
        bound = len(raw) + 1  # EOT is one token; C loader skips its newline.
        assert bound <= config["context_length"], bound
        max_bound = max(max_bound, bound)
        prompt = " ".join(turns[1].split()).casefold()
        if len(prompt) >= 32 and len(prompt.split()) >= 6:
            prompts[split].add(prompt)
        unique[split].add(" ".join(record.split()).casefold())
    assert len(data) > config["context_length"]
    print(f"{split}: {len(records):,} records; {len(unique[split]):,} unique; {len(data):,} bytes; maximum raw token bound {max_bound}")
assert not (prompts["train"] & prompts["validation"]), "Substantive held-out first prompt leaked"
assert not (unique["train"] & unique["validation"]), "Conversation leaked"
assert sum(s["train_selected_unique"]*s["train_repeat"] for s in manifest["sources"].values()) == manifest["files"]["train"]["conversations"]
assert sum(s["validation_selected_unique"] for s in manifest["sources"].values()) == manifest["files"]["validation"]["conversations"]
print("Config, hashes, complete-turn structure, context fit and split isolation passed.")
