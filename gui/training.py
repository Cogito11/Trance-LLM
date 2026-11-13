"""Detached run management and JSON-lines progress tracking."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import uuid

from .storage import ROOT, STATE, atomic_json, read_json, validate_config
from .runtime import FROZEN,BUNDLE,supervisor_program,supervisor_arguments


def run_alive(status: dict) -> bool:
    if status.get("status") not in ("running", "starting"):
        return False
    pid = status.get("runner_pid")
    if not pid:
        return status.get("status") == "starting" and time.time() - status.get("started_at", 0) < 15
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    if sys.platform.startswith("linux"):
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
            # Avoid treating a reused PID as this training supervisor.
            return (b"gui.training_worker" in cmdline or b"--training-worker" in cmdline) and str(status["run_dir"]).encode() in cmdline
        except OSError:
            return False
    return True


def latest_run() -> Path | None:
    runs = STATE / "runs"
    candidates = sorted(runs.glob("*/status.json"), key=lambda p: p.parent.name, reverse=True)
    active=[]
    for path in candidates:
        try:
            if run_alive(read_json(path, {})): active.append(path)
        except (ValueError,OSError):
            continue
    return (active or candidates)[0].parent if candidates else None


def recent_runs(limit: int = 12) -> list[dict]:
    """Newest-first run summaries for the sidebar; unreadable runs are skipped."""
    found = []
    paths = sorted((STATE / "runs").glob("*/status.json"), key=lambda p: p.parent.name, reverse=True)
    for path in paths[:limit]:
        try:
            status = read_json(path, {}); config = read_json(path.parent / "config.json", {})
        except (ValueError, OSError):
            continue
        recorded = status.get("status")
        if run_alive(status): state = "running"
        elif recorded in ("complete", "stopped", "failed"): state = recorded
        elif recorded in ("running", "starting"): state = "failed"  # supervisor vanished
        else: state = "unknown"
        found.append({"dir": path.parent, "model": str(config.get("output_model", "")), "state": state})
    return found


def launch_training(config: dict) -> Path:
    validate_config(config)
    active = latest_run()
    if active and run_alive(read_json(active / "status.json", {})):
        raise ValueError("A training run is already active")
    directory = STATE / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    (ROOT / config["output_model"]).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "config.json", config)
    # The snapshot is passed to the engine; later edits to the original config
    # cannot change a running job. Initial status also makes early Stop reliable.
    atomic_json(directory / "status.json", {"status": "starting", "started_at": time.time(), "run_dir": str(directory)})
    # Qt performs a true detached launch, with no owned child handle that
    # would be destroyed (or leave a Popen ResourceWarning) on GUI shutdown.
    from PySide6.QtCore import QProcess, QIODevice,QProcessEnvironment
    worker=QProcess()
    worker.setProgram(supervisor_program())
    worker.setArguments(supervisor_arguments(directory))
    environment=QProcessEnvironment.systemEnvironment()
    if FROZEN:
        environment.insert("PYINSTALLER_RESET_ENVIRONMENT","1")
    else:
        previous=environment.value("PYTHONPATH","")
        environment.insert("PYTHONPATH",str(BUNDLE)+(os.pathsep+previous if previous else ""))
    worker.setProcessEnvironment(environment)
    worker.setWorkingDirectory(str(ROOT))
    worker.setStandardInputFile(QProcess.nullDevice())
    worker.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
    worker.setStandardOutputFile(str(directory/"supervisor.log"),QIODevice.OpenModeFlag.Append)
    result=worker.startDetached()
    started=result[0] if isinstance(result,tuple) else bool(result)
    if not started:
        atomic_json(directory/"status.json",{"status":"failed","error":"Cannot launch training supervisor","run_dir":str(directory)})
        raise OSError("Cannot launch the training supervisor")
    return directory


def stop_training(directory: Path):
    status = read_json(directory / "status.json", {})
    if not run_alive(status):
        raise ValueError("This training run is no longer active")
    (directory / "stop.request").touch()


class EventTail:
    def __init__(self, directory: Path):
        self.directory = directory
        self.offset = 0
        self.buffer = b""

    def read(self) -> list[dict]:
        path = self.directory / "events.jsonl"
        if not path.exists():
            return []
        with path.open("rb") as f:
            f.seek(self.offset)
            self.buffer += f.read()
            self.offset = f.tell()
        events = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            try:
                event=json.loads(line)
                if not isinstance(event,dict): raise ValueError("Expected event object")
                events.append(event)
            except (ValueError, UnicodeError):
                events.append({"event": "log", "message": "Unreadable training event"})
        return events
