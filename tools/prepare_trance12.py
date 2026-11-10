#!/usr/bin/env python3
r"""Fetch pinned HF additions and build a context-safe, reproducible Trance mix.

Run from any directory using the project environment:
  data/01\ Dataset\ Pulling\ Scripts/.venv-trance12/bin/python tools/prepare_trance12.py --fetch
  data/01\ Dataset\ Pulling\ Scripts/.venv-trance12/bin/python tools/prepare_trance12.py --build

The old data and models are never overwritten. Output paths are project-relative.
Complete conversation prefixes only: never truncate an answer or orphan a follow-up.
The UTF-8 byte bound guarantees context fit even before the new BPE is trained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/trance12"
NEW = ROOT / "data/trance12_sources"
CACHE = ROOT / "data/hf_cache_trance12"
SEED = 4004
CONTEXT = 1024
PINS = {
    "HuggingFaceTB/smoltalk": "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
    "OpenAssistant/oasst1": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
}
# Caps apply to UNIQUE training conversations before the specified repetition.
# The loader samples conversation starts uniformly, so repetition changes weight.
SOURCES = {
    "smoltalk": (60000, 1), "ultrachat": (15000, 1),
    "no_robots": (0, 3), "everyday_conversations": (0, 3),
    "smol_constraints": (15000, 1), "smol_rewrite": (15000, 1),
    "smol_summarize": (20000, 1), "oasst1_en": (0, 3),
}
RESERVED = ("<user>", "<assistant>", "<eot>", "<|endoftext|>")
ARTIFACTS = ("<|system|>", "<|user|>", "<|assistant|>", "[inst]", "[/inst]")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def heldout_prompt_key(text: str) -> str | None:
    value = " ".join(text.split()).casefold()
    # Common greetings (Hi/Hello/How are you?) legitimately start thousands
    # of different chats. Reserving them would erase almost all casual data.
    return digest(value) if len(value) >= 32 and len(value.split()) >= 6 else None


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def content_ok(text: str) -> bool:
    lower = text.lower()
    return bool(text) and not any(t in lower for t in RESERVED + ARTIFACTS) and not any(
        ord(c) < 32 and c not in "\n\t" for c in text
    ) and "\ufffd" not in text


def normalize(messages: list) -> list[tuple[str, str]] | None:
    turns = []
    systems = []
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            return None
        role, text = m.get("role"), clean(m["content"])
        if not content_ok(text):
            return None
        if role == "system":
            if turns:
                return None
            systems.append(text)
            continue
        if role not in ("user", "assistant"):
            return None
        if turns and role == turns[-1][0]:
            turns[-1] = (role, turns[-1][1] + "\n\n" + text)
        else:
            turns.append((role, text))
    if len(turns) < 2 or len(turns) % 2 or any(
        role != ("user" if i % 2 == 0 else "assistant") for i, (role, _) in enumerate(turns)
    ):
        return None
    if systems:
        # Trance has no system token. Keep the instruction explicitly in the
        # user request; silently dropping it makes rewrite/summary targets ambiguous.
        turns[0] = ("user", "\n\n".join(systems) + "\n\nInput:\n" + turns[0][1])
    return turns


def render(turns: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"<{r}>\n{t}" for r, t in turns) + "\n\n<eot>\n"


def raw_token_bound(turns: list[tuple[str, str]]) -> int:
    # UTF-8 bytes, collapsing each reserved marker to its single raw token,
    # and excluding the newline the C loader skips after EOT.
    # One marker + one newline per turn, two newlines between turns,
    # and two newlines plus EOT after the final turn.
    return sum(len(text.encode("utf-8")) for _, text in turns) + 4*len(turns) + 1


def parse(text: str) -> list[tuple[str, str]] | None:
    blocks = re.split(r"(?m)^<(user|assistant)>\n", text.strip())
    if blocks[0] or len(blocks) < 5:
        return None
    return normalize([{"role": blocks[i], "content": blocks[i+1]} for i in range(1, len(blocks), 2)])


def iter_text(path: Path):
    # Read bounded chunks instead of loading a 2.5GB file. An embedded EOT
    # tag makes a malformed fragment which parse() will reject.
    tail = ""
    with path.open(encoding="utf-8", errors="strict") as f:
        while chunk := f.read(1024*1024):
            parts = (tail + chunk).split("<eot>")
            tail = parts.pop()
            for text in parts:
                if text.strip():
                    yield text
        if tail.strip():
            raise ValueError(f"Unterminated conversation in {path}")


def path_for(source: str, split: str) -> Path:
    base = ROOT / "data" if source in ("smoltalk", "ultrachat", "no_robots", "everyday_conversations") else NEW
    return base / source / (split + ".txt")


def fetch() -> None:
    os.environ.setdefault("HF_HOME", str(CACHE))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    NEW.mkdir(parents=True, exist_ok=True)
    manifest = {"revisions": PINS, "downloads": [], "sources": {}}

    def download(repo, filename):
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=PINS[repo], cache_dir=str(CACHE)))
        sha = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1024*1024):
                sha.update(chunk)
        manifest["downloads"].append({"repo": repo, "file": filename, "bytes": path.stat().st_size, "sha256": sha.hexdigest()})
        return path

    for repo in PINS:
        card = download(repo, "README.md")
        (NEW / (repo.replace("/", "_") + "_README.md")).write_bytes(card.read_bytes())

    for config in ("smol-constraints", "smol-rewrite", "smol-summarize"):
        source = config.replace("-", "_")
        dest = NEW / source
        dest.mkdir(exist_ok=True)
        stats = Counter()
        for upstream, split in (("train", "train"), ("test", "validation")):
            path = download("HuggingFaceTB/smoltalk", f"data/{config}/{upstream}-00000-of-00001.parquet")
            temp = dest / (split + ".txt.tmp")
            with temp.open("w", encoding="utf-8") as f:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
                    for row in batch.to_pylist():
                        stats[split + "_rows"] += 1
                        turns = normalize(row["messages"])
                        if turns:
                            f.write(render(turns)); stats[split + "_accepted"] += 1
                        else:
                            stats[split + "_rejected"] += 1
            temp.replace(dest / (split + ".txt"))
        manifest["sources"][source] = dict(stats)
        print(source, dict(stats), flush=True)

    dest = NEW / "oasst1_en"
    dest.mkdir(exist_ok=True)
    stats = Counter()
    for filename, split in (("train-00000-of-00001-b42a775f407cee45.parquet", "train"),
                            ("validation-00000-of-00001-134b8fd0c89408b6.parquet", "validation")):
        path = download("OpenAssistant/oasst1", "data/" + filename)
        rows = pq.read_table(path).to_pylist()
        by_id = {r["message_id"]: r for r in rows}
        stats[split + "_rows"] = len(rows)

        def eligible(r):
            labels = dict(zip((r.get("labels") or {}).get("name", []), (r.get("labels") or {}).get("value", [])))
            return r["lang"] == "en" and not r["deleted"] and not r["synthetic"] and r["review_result"] is True and (
                r["tree_state"] == "ready_for_export"
            ) and labels.get("quality", 1) >= 0.5 and labels.get("spam", 0) < 0.5 and labels.get("lang_mismatch", 0) < 0.5

        temp = dest / (split + ".txt.tmp")
        with temp.open("w", encoding="utf-8") as f:
            for row in rows:
                if row["role"] != "assistant" or row["rank"] != 0 or not eligible(row):
                    continue
                chain, seen = [], set()
                current = row
                while current:
                    if current["message_id"] in seen or not eligible(current):
                        chain = []; break
                    seen.add(current["message_id"]); chain.append(current)
                    parent = current["parent_id"]
                    if parent and parent not in by_id:
                        chain = []; break
                    current = by_id.get(parent)
                turns = normalize([{"role": "user" if r["role"] == "prompter" else "assistant", "content": r["text"]}
                                   for r in reversed(chain)]) if chain else None
                if turns:
                    f.write(render(turns)); stats[split + "_accepted"] += 1
        temp.replace(dest / (split + ".txt"))
    manifest["sources"]["oasst1_en"] = dict(stats)
    print("oasst1_en", dict(stats), flush=True)
    (NEW / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        for split in ("train", "validation"):
            if not path_for(source, split).is_file():
                raise FileNotFoundError(path_for(source, split))
    stats = {source: Counter() for source in SOURCES}
    # Reserve substantive original held-out first prompts, even if their
    # answers are later length-filtered. Generic greetings are exempt.
    heldout_prompts = set()
    for source in SOURCES:
        for text in iter_text(path_for(source, "validation")):
            turns = parse(text)
            if turns:
                prompt = heldout_prompt_key(turns[0][1])
                if prompt:
                    heldout_prompts.add(prompt)
    seen = set()
    selected = {source: {"train": [], "validation": []} for source in SOURCES}
    # Held-out wins across sources, not just within each original dataset.
    for split in ("validation", "train"):
        for source, (cap, _) in SOURCES.items():
            rng = random.Random(f"{SEED}:{source}:{split}")
            accepted = 0
            bucket = selected[source][split]
            for text in iter_text(path_for(source, split)):
                st = stats[source]
                st[split + "_input"] += 1
                if st[split + "_input"] % 50000 == 0:
                    print(source, split, "processed", st[split + "_input"], flush=True)
                turns = parse(text)
                if turns is None:
                    st[split + "_malformed"] += 1; continue
                prompt = heldout_prompt_key(turns[0][1])
                if split == "train" and prompt is not None and prompt in heldout_prompts:
                    st["train_heldout_prompt_removed"] += 1; continue
                original_pairs = len(turns)//2
                while len(turns) > 2 and raw_token_bound(turns) > CONTEXT:
                    turns = turns[:-2]
                if raw_token_bound(turns) > CONTEXT:
                    st[split + "_too_long"] += 1; continue
                if any(len(t) < 2 for _, t in turns):
                    st[split + "_too_short"] += 1; continue
                text = render(turns)
                key = digest(" ".join(text.split()).casefold())
                if key in seen:
                    st[split + "_duplicate"] += 1; continue
                seen.add(key)
                if original_pairs > len(turns)//2:
                    st[split + "_trimmed_complete_pairs"] += 1
                accepted += 1
                # Uniform reservoir: a cap does not select a biased file prefix.
                if split == "train" and cap and len(bucket) >= cap:
                    j = rng.randrange(accepted)
                    if j < cap:
                        bucket[j] = text
                else:
                    bucket.append(text)
            st[split + "_eligible_unique"] = accepted
            st[split + "_selected_unique"] = len(bucket)
            print(source, split, dict(st), flush=True)

    manifest = {"seed": SEED, "context": CONTEXT, "filter": "complete prefixes; raw UTF-8 token bound <= context",
                "dedup": "casefolded whitespace-normalized SHA256, globally; substantive held-out first prompts reserved (>=32 chars and >=6 words); generic greetings exempt",
                "sources": {}, "files": {}, "source_files": {}}
    mixed = {"train": [], "validation": []}
    for source, (cap, repeat) in SOURCES.items():
        manifest["sources"][source] = {**dict(stats[source]), "train_repeat": repeat, "unique_train_cap": cap}
        for split in mixed:
            texts = selected[source][split]
            subdir = OUT / source
            subdir.mkdir(exist_ok=True)
            (subdir / (split + ".txt")).write_text("".join(texts), encoding="utf-8")
            mixed[split].extend(texts * (repeat if split == "train" else 1))
            path = path_for(source, split)
            sha = hashlib.sha256()
            with path.open("rb") as f:
                while chunk := f.read(1024*1024):
                    sha.update(chunk)
            manifest["source_files"][str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": sha.hexdigest()}
    for split, texts in mixed.items():
        if not texts:
            raise ValueError(f"Empty {split} output")
        random.Random(f"{SEED}:mix:{split}").shuffle(texts)
        path = OUT / (split + ".txt")
        temp = path.with_suffix(".txt.tmp")
        sha = hashlib.sha256()
        with temp.open("wb") as f:
            for text in texts:
                data = text.encode("utf-8"); f.write(data); sha.update(data)
        temp.replace(path)
        manifest["files"][split] = {"bytes": path.stat().st_size, "conversations": len(texts), "sha256": sha.hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("FINAL", manifest["files"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not (args.fetch or args.build):
        parser.error("specify --fetch and/or --build")
    if args.fetch:
        fetch()
    if args.build:
        build()
