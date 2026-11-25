#!/usr/bin/env python3
"""Stand-in for `trance engine`: the same JSON-lines protocol (see native/engine.h), no GPU and no real model.

Like the real engine it holds at most one model, frees the previous one before reading a new one, and exits when its
input closes. A model path containing "BAD" fails to load (after the old model has been dropped, as the real one can).
If FAKE_ENGINE_LOG names a file, every command is appended to it as `command model`, so tests can see what was asked.
"""
import json
import os
import sys
import time

LOG = os.environ.get("FAKE_ENGINE_LOG")


def emit(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def main():
    loaded = None
    emit({"event": "ready", "protocol": 1, "gpu": "Fake GPU (compute capability 0.0)"})
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        rid, command = request["id"], request["command"]
        if LOG:
            with open(LOG, "a") as log:
                log.write(f"{command} {request.get('model', '')}\n")
        if command == "shutdown":
            emit({"event": "bye", "id": rid}); break
        if command == "unload":
            loaded = None
            emit({"event": "unloaded", "id": rid})
        elif command == "load":
            path = request["model"]
            loaded = None                                   # the previous model is freed before the new one is read
            time.sleep(float(os.environ.get("FAKE_ENGINE_LOAD_SECONDS", "0.15")))
            if "BAD" in os.path.basename(path):
                emit({"event": "error", "id": rid, "message": "Cannot load model"})
            else:
                loaded = path
                emit({"event": "loaded", "id": rid, "model": path, "parameters": 1_000_000.0, "step": 1,
                      "context_length": 256, "vocab_size": 259})
        elif command == "generate":
            if loaded is None:
                emit({"event": "error", "id": rid, "message": "Load a model first"}); continue
            emit({"event": "generation_started", "id": rid, "prompt_tokens": 4, "dropped_messages": 0, "context_length": 256})
            for ch in "Hi!":
                emit({"event": "delta", "id": rid, "bytes": ch.encode().hex()})
            emit({"event": "done", "id": rid, "reason": "stop", "generated_tokens": 3, "elapsed_seconds": 0.01})
        else:
            emit({"event": "error", "id": rid, "message": "Unknown command: " + command})


if __name__ == "__main__":
    main()
