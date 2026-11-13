#!/usr/bin/env python3
"""Stand-in for llama-server: same flags, same endpoints, no model. Used by backend tests.

A model file starting with b"BAD" makes it exit with an error like a real load failure.
"""
import argparse
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-m"); parser.add_argument("--host"); parser.add_argument("--port", type=int)
parser.add_argument("--api-key"); parser.add_argument("-c"); parser.add_argument("-np"); parser.add_argument("--no-webui", action="store_true")
parser.add_argument("--version", action="store_true")
args = parser.parse_args()
if args.version:
    print("version: 0.9.0-fake (build 4242, commit abc)"); sys.exit(0)
if Path(args.m).read_bytes()[:3] == b"BAD":
    print("llama_model_load: error loading model: unknown model architecture: 'nonsense'", file=sys.stderr, flush=True)
    sys.exit(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_openai  # noqa: E402  (imported late so --version works from an installed copy)

print("loading", args.m, "ctx", args.c, flush=True)
time.sleep(0.4)   # a model takes a moment to load
server, port = fake_openai.start(token=args.api_key, delay=0.01, port=args.port)
print("listening on", port, flush=True)
while True:
    time.sleep(1)
