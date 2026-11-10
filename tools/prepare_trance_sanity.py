#!/usr/bin/env python3
"""Prepare a small memorization diagnostic; never start training or replace data."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/"tools/trance_sanity_prompts.json"
    pairs=json.loads(source.read_text())
    conversations=[]
    for pair in pairs:
        for key in ("prompt","answer"):
            if not isinstance(pair.get(key),str) or not pair[key].strip(): raise ValueError(f"Missing {key}")
            if any(marker in pair[key] for marker in ("<user>","<assistant>","<eot>","<|endoftext|>")):
                raise ValueError("Reserved marker in diagnostic text")
        conversation=f"<user>\n{pair['prompt']}\n\n<assistant>\n{pair['answer']}\n\n<eot>\n"
        if len(conversation.encode())>1024: raise ValueError("Diagnostic conversation exceeds raw context bound")
        conversations.append(conversation)
    contents="".join(conversations)
    path=ROOT/"data/trance_sanity/train.txt"
    path.parent.mkdir(parents=True,exist_ok=True)
    # Idempotent preparation is allowed, but changed existing data is preserved.
    try:
        with path.open("x",encoding="utf-8") as output: output.write(contents)
    except FileExistsError:
        if path.read_text()!=contents: raise ValueError(f"Existing diagnostic data differs: {path}")
    print(json.dumps({"path":str(path),"conversations":len(pairs),"bytes":len(contents.encode()),
                      "sha256":hashlib.sha256(contents.encode()).hexdigest(),
                      "purpose":"training-set memorization test; not held-out quality evaluation"},indent=2))


if __name__=="__main__":main()
