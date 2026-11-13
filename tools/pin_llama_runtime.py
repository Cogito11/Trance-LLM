#!/usr/bin/env python3
"""Verify or update the pinned llama.cpp runtime builds used by gui/runtimes.py.

    python tools/pin_llama_runtime.py                 # re-check every pinned hash against the release (good for CI)
    python tools/pin_llama_runtime.py --tag b12400    # show the hashes a newer build would need
    python tools/pin_llama_runtime.py --tag b12400 --write   # also rewrite PINNED_TAG and the hashes in gui/runtimes.py

Downloads are streamed straight into SHA-256, so nothing is stored. The GitHub API is not used (it is rate limited).
After bumping, run the tests and try a real download on each platform you ship.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui import runtimes  # noqa: E402


def sha256_of(url: str) -> str:
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "TranceLLM-pin-tool"})
    with urllib.request.urlopen(request, timeout=60) as response:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=runtimes.PINNED_TAG, help="build tag to hash (default: the currently pinned one)")
    parser.add_argument("--write", action="store_true", help="rewrite gui/runtimes.py with the new tag and hashes")
    args = parser.parse_args()
    old_tag = runtimes.PINNED_TAG
    templates = sorted({a.name for flavors in runtimes.FLAVORS.values() for f in flavors for a in f.assets})
    pinned = {a.name: a.sha256 for flavors in runtimes.FLAVORS.values() for f in flavors for a in f.assets}

    def job(name: str):
        new_name = name.replace(old_tag, args.tag)
        try:
            return name, new_name, sha256_of(f"{runtimes.RELEASE_BASE}/{args.tag}/{new_name}")
        except Exception as error:   # report per asset instead of aborting the batch
            return name, new_name, f"ERROR {error}"

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(job, templates))
    bad = 0
    for name, new_name, sha in results:
        if sha.startswith("ERROR"): status, bad = sha, bad + 1
        elif args.tag == old_tag: status = "ok" if sha == pinned[name] else "MISMATCH"; bad += status != "ok"
        else: status = "new"
        print(f"{status:9s} {sha[:16]}  {new_name}")
    if bad:
        print(f"\n{bad} problem(s); nothing written."); return 1
    if args.write and args.tag != old_tag:
        path = Path(runtimes.__file__); text = path.read_text(encoding="utf-8")
        text = re.sub(r'PINNED_TAG = "b\d+"', f'PINNED_TAG = "{args.tag}"', text, count=1)
        for name, _new, sha in results:
            key = name.replace(old_tag, "{PINNED_TAG}")
            text, n = re.subn(re.escape(f'"{key}": "') + r"[0-9a-f]{64}", f'"{key}": "{sha}', text)
            if n != 1: print(f"Could not update the hash for {key}"); return 1
        path.write_text(text, encoding="utf-8"); print(f"\nUpdated {path} to {args.tag}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
