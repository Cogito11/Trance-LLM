#!/usr/bin/env python3
"""Build a pinned new-data curriculum; preserve every existing corpus/model.

Use the data environment (huggingface_hub and pyarrow):
  python tools/prepare_trance12_curriculum.py --fetch
Without --fetch, only already cached pinned source files are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from prepare_trance12 import clean, content_ok, heldout_prompt_key, iter_text, parse, raw_token_bound, render
from prepare_trance12_replay import publish, sha

ROOT = Path(__file__).resolve().parents[1]
SEED = 4107
SOURCES = {
    "stories": {
        "repo": "roneneldan/TinyStories",
        "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "license": "CDLA-Sharing-1.0",
        "train": "TinyStoriesV2-GPT4-train.txt",
        "validation": "TinyStoriesV2-GPT4-valid.txt",
    },
    "reading": {
        "repo": "rajpurkar/squad",
        "revision": "7b6d24c440a36b6815f21b70d25016731768db1f",
        "license": "CC-BY-SA-4.0",
        "train": "plain_text/train-00000-of-00001.parquet",
        "validation": "plain_text/validation-00000-of-00001.parquet",
    },
}


def key(text):
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def story_turns(raw):
    text = clean(raw)
    if (not content_ok(text) or not re.match(r'^[A-Z"“]', text) or
            any(x in text for x in ("â€", "Ã", "Â")) or
            len(text.split()) < 40 or not re.search(r"""[.!?]["”’')\]]?$""", text)):
        return None
    selector = int(key(text)[:8], 16)
    first = re.match(r"^(.{25,200}?[.!?])\s+(.+)$", text, flags=re.S)
    if selector % 4 and first and len(first[2].split()) >= 20:
        prefix = ("Continue this story:\n", "Finish this short story:\n",
                  "What happens next?\n", "Continue the story below:\n")[selector % 4]
        turns = [("user", prefix + first[1]), ("assistant", first[2])]
    else:
        prompt = ("Tell me a short story.", "Write a simple story.",
                  "Tell a short story.", "Please tell a short story.")[(selector // 4) % 4]
        turns = [("user", prompt), ("assistant", text)]
    # Reject over-length stories intact; never trim away an ending.
    return turns if raw_token_bound(turns) <= 1024 else None


def reading_turns(row):
    context, question = row.get("context", ""), row.get("question", "")
    answers = row.get("answers") or {}
    text, starts = answers.get("text", []), answers.get("answer_start", [])
    answer = None
    for value, start in zip(text, starts):
        if (isinstance(start, int) and start >= 0 and value and
                context[start:start + len(value)] == value and
                len(value.split()) <= 16 and len(value.encode()) <= 120):
            answer = clean(value)
            break
    if not answer or not all(content_ok(clean(v)) for v in (context, question, answer)):
        return None
    prompt = f"Read the passage and answer briefly.\nPassage: {clean(context)}\nQuestion: {clean(question)}"
    turns = [("user", prompt), ("assistant", answer)]
    return turns if raw_token_bound(turns) <= 1024 else None


def text_records(path):
    for index, raw in enumerate(iter_text(path), 1):
        turns = parse(raw)
        if not turns or raw_token_bound(turns) > 1024:
            raise ValueError(f"Malformed or over-length legacy record {index}: {path}")
        yield turns


def iter_stories(path):
    tail = ""
    with path.open(encoding="utf-8") as f:
        while chunk := f.read(1024 * 1024):
            parts = (tail + chunk).split("<|endoftext|>")
            tail = parts.pop()
            for text in parts:
                if text.strip():
                    yield text
        if tail.strip():
            # The pinned upstream files can end without a delimiter.
            # story_turns still rejects a visibly unfinished final sentence.
            yield tail


def iter_reading(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def assemble(story_train, story_validation, reading_train, reading_validation,
             legacy_train, legacy_validation, eval_prompts, *,
             story_limit=200000, validation_caps=(2500, 1000, 1500), seed=SEED):
    """Pure assembly entry point; upstream validation always reserved first."""
    stats = Counter()
    reserved_stories, reserved_contexts, reserved_questions = set(), set(), set()
    reserved_records, reserved_prompts = set(), set()
    evaluation_keys = {key(x) for x in eval_prompts}
    valid = {"stories": [], "reading": [], "legacy": []}
    valid_seen = set()

    def reserve(turns, source):
        if turns is None:
            return
        record = render(turns)
        identity = key(record)
        reserved_records.add(identity)
        prompt_identity = heldout_prompt_key(turns[0][1])
        if prompt_identity:
            reserved_prompts.add(prompt_identity)
        if identity not in valid_seen:
            valid_seen.add(identity)
            valid[source].append(record)

    for raw in story_validation:
        reserved_stories.add(key(raw))
        reserve(story_turns(raw), "stories")
    for row in reading_validation:
        reserved_contexts.add(key(row["context"]))
        reserved_questions.add(key(row["question"]))
        reserve(reading_turns(row), "reading")
    for turns in legacy_validation:
        reserve(turns, "legacy")

    def eligible(turns):
        if not turns:
            return False
        return (key(render(turns)) not in reserved_records and
                heldout_prompt_key(turns[0][1]) not in reserved_prompts and
                all(key(text) not in evaluation_keys for role, text in turns if role == "user"))

    stories, seen_stories, accepted = [], set(), 0
    rng = random.Random(f"stories:{seed}")
    for raw in story_train:
        stats["stories_input"] += 1
        if stats["stories_input"] % 250000 == 0:
            print(f"Scanned {stats['stories_input']:,} stories; eligible {accepted:,}", flush=True)
        identity = key(raw)
        if identity in reserved_stories:
            stats["stories_heldout_removed"] += 1
            continue
        turns = story_turns(raw)
        if not turns:
            stats["stories_quality_or_length_rejected"] += 1
            continue
        if not eligible(turns):
            stats["stories_reserved_prompt_or_record"] += 1
            continue
        if identity in seen_stories:
            stats["stories_duplicates"] += 1
            continue
        seen_stories.add(identity)
        accepted += 1
        record = render(turns)
        if len(stories) < story_limit:
            stories.append(record)
        else:
            index = rng.randrange(accepted)
            if index < story_limit:
                stories[index] = record
    stats["stories_eligible_unique"] = accepted
    if len(stories) < story_limit:
        raise ValueError(f"Need {story_limit} stories, found {len(stories)}")

    reading, reading_seen = [], set()
    for row in reading_train:
        stats["reading_input"] += 1
        if key(row["context"]) in reserved_contexts or key(row["question"]) in reserved_questions:
            stats["reading_heldout_removed"] += 1
            continue
        turns = reading_turns(row)
        if not turns:
            stats["reading_quality_or_length_rejected"] += 1
            continue
        identity = key(turns[0][1])
        if identity in reading_seen or not eligible(turns):
            stats["reading_duplicate_or_reserved"] += 1
            continue
        reading_seen.add(identity)
        reading.append(render(turns))
    if not reading:
        raise ValueError("No eligible reading examples")

    legacy = []
    for turns in legacy_train:
        stats["legacy_input"] += 1
        if eligible(turns):
            legacy.append(render(turns))
        else:
            stats["legacy_reserved_removed"] += 1
    if not legacy:
        raise ValueError("No eligible legacy examples")

    selected_validation = {}
    for (source, records), cap in zip(valid.items(), validation_caps):
        stats[f"{source}_validation_eligible"] = len(records)
        if len(records) < cap:
            raise ValueError(f"Not enough {source} validation records: {len(records)} < {cap}")
        selected_validation[source] = random.Random(f"validation:{source}:{seed}").sample(records, cap)
    train = stories + reading * 2 + legacy
    validation = [r for records in selected_validation.values() for r in records]
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(validation)
    stats.update(stories_selected=len(stories), reading_selected_unique=len(reading),
                 reading_weighted=2 * len(reading), legacy_selected_weighted=len(legacy),
                 training_records=len(train), validation_records=len(validation),
                 training_unique=len({key(r) for r in train}))
    if {key(r) for r in train} & {key(r) for r in validation}:
        raise AssertionError("Training/validation overlap")
    outputs = {"train.txt": "".join(train).encode(),
               "validation.txt": "".join(validation).encode()}
    for source, records in selected_validation.items():
        outputs[f"{source}_validation.txt"] = "".join(records).encode()
    return outputs, dict(stats)


def build(root, fetch=False):
    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download
    root = Path(root)
    source_paths, downloads = {}, []
    cards = {}
    for source, info in SOURCES.items():
        for split, filename in (("card", "README.md"), ("train", info["train"]),
                                ("validation", info["validation"])):
            path = Path(hf_hub_download(info["repo"], filename, repo_type="dataset",
                                       revision=info["revision"], cache_dir=root/"data/hf_cache_curriculum",
                                       local_files_only=not fetch))
            source_paths[source, split] = path
            downloads.append({"repo": info["repo"], "revision": info["revision"],
                              "file": filename, "bytes": path.stat().st_size, "sha256": sha(path)})
            if split == "card":
                cards[f"sources/{source}_README.md"] = path.read_bytes()
    legacy_paths = {s: root/f"data/trance12/{s}.txt" for s in ("train", "validation")}
    prompt_path = root/"tools/trance_generalization_prompts.json"
    prompts = [row["prompt"] for row in json.loads(prompt_path.read_text())]
    outputs, stats = assemble(
        iter_stories(source_paths["stories", "train"]),
        iter_stories(source_paths["stories", "validation"]),
        iter_reading(source_paths["reading", "train"]),
        iter_reading(source_paths["reading", "validation"]),
        text_records(legacy_paths["train"]), text_records(legacy_paths["validation"]), prompts)
    manifest = {
        "seed": SEED, "context_raw_bound": 1024, "sources": SOURCES,
        "downloads": downloads, "statistics": stats,
        "legacy_sha256": {s: sha(p) for s, p in legacy_paths.items()},
        "evaluation_prompt_sha256": sha(prompt_path),
        "files": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                  for name, content in outputs.items()},
        "format": "complete user/assistant conversations; assistant-target training",
        "weighting": "200000 unique stories; each selected SQuAD example twice; original instruction weights retained",
        "split_isolation": "official validation reserved first; normalized exact stories, passages, questions, conversations and substantive first prompts excluded; not a semantic near-duplicate audit",
        "preparation_only": True,
    }
    outputs.update(cards)
    outputs["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    out = root/"data/trance12_curriculum"
    # Check all existing paths before publishing any file; never overwrite differences.
    for name, content in outputs.items():
        path = out/name
        if path.exists() and sha(path) != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Existing output differs; preserved: {path}")
    for name, content in outputs.items():
        publish(out/name, content)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    result = build(args.project, args.fetch)
    print(json.dumps(result["statistics"], indent=2))
