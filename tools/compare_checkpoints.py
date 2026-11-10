#!/usr/bin/env python3
"""Compare saved models on fixed prompts/settings without training or Qt.

Reports contain raw responses for human review, not an automatic quality score.
The engine runs models sequentially and leaves all model files untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--prompts", type=Path, default=ROOT/"tools/trance12_eval_prompts.json")
    parser.add_argument("--limit", type=int, default=0, help="0 runs all prompts")
    parser.add_argument("--temperature", type=float, default=.5)
    parser.add_argument("--top-p", type=float, default=.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4004)
    parser.add_argument("--timeout", type=float, default=180, help="Total seconds allowed per request")
    parser.add_argument("--engine", type=Path, default=ROOT/"trance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Output already exists; choose a new report path")
    if args.limit < 0 or not math.isfinite(args.timeout) or args.timeout <= 0: parser.error("Invalid limit or timeout")
    if (not math.isfinite(args.temperature) or not 0 < args.temperature <= 5 or
            not math.isfinite(args.top_p) or not 0 < args.top_p <= 1 or
            not 0 <= args.top_k <= 1024 or not 1 <= args.max_tokens <= 1023 or
            not 0 <= args.seed <= 2**31-1): parser.error("Invalid generation settings")
    models = [path.resolve() for path in args.models]
    for model in models:
        if not model.is_file() or not Path(str(model)+".tok").is_file():
            parser.error(f"Missing model/tokenizer: {model}")
    prompts = json.loads(args.prompts.read_text())
    if not isinstance(prompts,list) or not prompts or any(not isinstance(p,dict) or not p.get("prompt") for p in prompts):
        parser.error("Prompts must be a nonempty list of objects with prompt text")
    if args.limit: prompts = prompts[:args.limit]
    settings = {"temperature":args.temperature,"top_p":args.top_p,"top_k":args.top_k,
                "max_new_tokens":args.max_tokens,"seed":args.seed}
    report = {"created_at":datetime.now(timezone.utc).isoformat(),"settings":settings,
              "fresh_context_per_prompt":True,"results":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    events = queue.Queue()
    # A file for stderr avoids a full stderr pipe blocking the native engine.
    with tempfile.TemporaryFile(mode="w+b") as errors:
        process = subprocess.Popen([str(args.engine.resolve()),"engine"],cwd=ROOT,
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True)
        def reader():
            for line in process.stdout: events.put(line)
            events.put(None)
        thread = threading.Thread(target=reader,daemon=True);thread.start()
        deadline = time.monotonic()+args.timeout
        def event():
            try: line = events.get(timeout=max(0,deadline-time.monotonic()))
            except queue.Empty: raise TimeoutError("Native engine response timed out")
            if line is None:
                errors.seek(0)
                raise RuntimeError("Engine exited: " + errors.read().decode("utf-8","replace"))
            value = json.loads(line)
            if value.get("event") == "error": raise RuntimeError(value.get("message",str(value)))
            return value
        serial = 0
        def send(command, **fields):
            nonlocal serial, deadline
            serial += 1
            deadline = time.monotonic()+args.timeout
            process.stdin.write(json.dumps({"id":serial,"command":command,**fields})+"\n")
            process.stdin.flush()
        try:
            ready = event()
            if ready.get("event") != "ready" or ready.get("protocol") != 1:
                raise RuntimeError(f"Unexpected engine protocol: {ready}")
            report["gpu"] = ready.get("gpu")
            for model in models:
                send("load",model=str(model))
                loaded = event()
                if loaded.get("event") != "loaded": raise RuntimeError(str(loaded))
                result = {"model":str(model),"step":loaded["step"],"responses":[]}
                report["results"].append(result)
                for prompt in prompts:
                    send("generate",messages=[{"role":"user","content":prompt["prompt"]}],**settings)
                    response = {**prompt};raw = bytearray()
                    while True:
                        value = event()
                        if value.get("id") != serial: raise RuntimeError("Mismatched response ID")
                        kind = value.get("event")
                        if kind == "generation_started": response["prompt_tokens"] = value["prompt_tokens"]
                        elif kind == "delta": raw.extend(bytes.fromhex(value["bytes"]))
                        elif kind == "done":
                            response.update(content=raw.decode("utf-8","replace"),
                                            reason=value["reason"],generated_tokens=value["generated_tokens"],
                                            elapsed_seconds=value["elapsed_seconds"])
                            result["responses"].append(response)
                            print(f"{model.name}: {len(result['responses'])}/{len(prompts)}",flush=True)
                            break
                        else: raise RuntimeError(f"Unexpected event: {value}")
            send("shutdown")
            if event().get("event") != "bye": raise RuntimeError("Shutdown was not acknowledged")
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill();process.wait(timeout=10)
            process.stdin.close();thread.join(timeout=2);process.stdout.close()
    # Create-only: a comparison never overwrites an earlier report.
    with args.output.open("x",encoding="utf-8") as output:
        json.dump(report,output,ensure_ascii=False,indent=2);output.write("\n")
    print(f"Responses saved in {args.output}")


if __name__ == "__main__": main()
