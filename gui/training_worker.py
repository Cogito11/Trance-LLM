"""Detached training supervisor. Closing the GUI does not terminate this process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .storage import ROOT, atomic_json
from .runtime import ENGINE


def run(directory: Path, binary: Path | None = None):
    directory = directory.resolve()
    status_path = directory / "status.json"
    status = {"status": "starting", "runner_pid": os.getpid(), "started_at": time.time(), "run_dir": str(directory)}
    atomic_json(status_path, status)
    command = [str(binary or ENGINE), "train",
               "--config", str(directory / "config.json"), "--json-events", "--stop-file", str(directory / "stop.request")]
    result = "failed"
    try:
        with (directory / "events.jsonl").open("a", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            status.update(status="running", engine_pid=process.pid, command=command)
            atomic_json(status_path, status)
            for line in process.stdout:
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("Expected an event object")
                except ValueError:
                    event = {"event": "log", "message": line.rstrip()}
                log.write(json.dumps(event, ensure_ascii=True) + "\n")
                if event.get("event") == "stage" and not status.get("native_ready"):
                    status["native_ready"] = True
                    atomic_json(status_path, status)
                if event.get("event") == "training_started":
                    status.update(base_step=event["step"],end_step=event["end_step"])
                    atomic_json(status_path,status)
                # The engine checks --stop-file at each safe step boundary,
                # including after startup. Avoid a redundant signal racing
                # with its terminal event and successful process shutdown.
                if event.get("event") in ("complete", "stopped"):
                    result = "complete" if event["event"] == "complete" else "stopped"
                    status.update(checkpoint=event.get("path"), step=event.get("step"))
            code = process.wait()
            status.update(status=result if code == 0 else "failed", exit_code=code, finished_at=time.time())
    except Exception as error:
        status.update(status="failed", error=str(error), finished_at=time.time())
    atomic_json(status_path, status)
    return 0 if status["status"] in ("complete", "stopped") else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m gui.training_worker RUN_DIRECTORY")
    raise SystemExit(run(Path(sys.argv[1])))
