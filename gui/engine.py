"""Asynchronous Qt client for the persistent native JSON-lines engine."""
from __future__ import annotations

import codecs
import json
import os
from pathlib import Path
import signal

from PySide6.QtCore import QObject, QProcess, Signal

from .storage import ROOT
from .runtime import ENGINE


class EngineClient(QObject):
    delta = Signal(str)
    started = Signal(dict)
    finished = Signal(dict)
    failed = Signal(str)
    state_changed = Signal(str)
    ready = Signal(str)
    loaded = Signal(dict)

    def __init__(self, parent=None, binary: Path | None = None):
        super().__init__(parent)
        self.binary = binary or ENGINE
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.readyReadStandardError.connect(self._stderr)
        self.process.errorOccurred.connect(self._process_error)
        self.process.finished.connect(self._exited)
        self._buffer = b""
        self._errors = ""
        self._serial = 0
        self._id = None
        self._pending = None
        self._loaded = ""
        self._ready = False
        self._phase = "idle"
        self._cancelled = False
        self._decoder = None
        self._closed = False

    @property
    def busy(self):
        return self._pending is not None

    def generate(self, model: str, messages: list[dict], settings: dict):
        if self.busy:
            raise RuntimeError("The engine is already working")
        self._pending = {"model": str(Path(model).resolve()), "messages": messages, **settings}
        self._cancelled = False
        self._closed = False
        self._errors = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._phase = "starting"
            self.state_changed.emit("Starting engine…")
            self._ready = False
            self._buffer = b""
            self.process.start(str(self.binary), ["engine"])
        elif self._ready:
            self._prepare()

    def _send(self, command, **fields):
        self._serial += 1
        self._id = self._serial
        request = {"id": self._id, "command": command, **fields}
        self.process.write((json.dumps(request, ensure_ascii=True) + "\n").encode())

    def _prepare(self):
        if not self._pending:
            return
        if self._loaded != self._pending["model"]:
            self._phase = "loading"
            self.state_changed.emit("Loading model…")
            self._send("load", model=self._pending["model"])
        else:
            self._generate()

    def _generate(self):
        self._phase = "preparing"
        self.state_changed.emit("Preparing context…")
        fields = {k: v for k, v in self._pending.items() if k != "model"}
        self._send("generate", **fields)

    def _read(self):
        self._buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                self._fail("Engine returned an invalid protocol message. Rebuild the native binary.")
                self.process.kill()
                return
            kind = event.get("event")
            if kind == "ready":
                if event.get("protocol") != 1:
                    self._fail("Unsupported engine protocol")
                    self.process.kill()
                    return
                self._ready = True
                self.ready.emit(event.get("gpu", "GPU ready"))
                self._prepare()
                continue
            if event.get("id") != self._id:
                continue
            if kind == "loaded":
                self._loaded = event["model"]
                self.loaded.emit(event)
                self._generate()
            elif kind == "generation_started":
                self._phase = "generating"
                self.state_changed.emit("Generating…")
                self.started.emit(event)
            elif kind == "delta":
                try:
                    text = self._decoder.decode(bytes.fromhex(event["bytes"]))
                except (ValueError, KeyError):
                    self._fail("Engine returned invalid streamed bytes")
                    self.process.kill()
                    return
                if text:
                    self.delta.emit(text)
            elif kind == "done":
                tail = self._decoder.decode(b"", final=True)
                if tail:
                    self.delta.emit(tail)
                self._pending = None
                self._phase = "idle"
                self.finished.emit(event)
            elif kind == "error":
                self._fail(event.get("message", "Engine error"))

    def _stderr(self):
        self._errors = (self._errors + bytes(self.process.readAllStandardError()).decode("utf-8", "replace"))[-8192:]

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._fail(f"Cannot start {self.binary.name}. Build the CUDA engine first.")

    def _exited(self, code, status):
        self._stderr()
        self._ready = False
        self._loaded = ""
        self._buffer = b""
        if self._pending:
            if self._cancelled:
                self._pending = None
                self._phase = "idle"
                self.finished.emit({"reason": "cancelled", "generated_tokens": 0, "elapsed_seconds": 0})
            else:
                self._fail(self._errors.strip() or f"Engine exited with code {code}")

    def _fail(self, message):
        self._pending = None
        self._phase = "idle"
        self.failed.emit(message)

    def cancel(self):
        if not self.busy:
            return
        self._cancelled = True
        if self._phase == "generating" and hasattr(signal, "SIGUSR1"):
            try:
                os.kill(self.process.processId(), signal.SIGUSR1)
                self.state_changed.emit("Stopping response…")
                return
            except ProcessLookupError:
                pass
        self.process.kill()

    def release(self):
        """Release all GPU memory before a training run; callers must be idle."""
        if self.busy:
            raise RuntimeError("Stop generation before starting training")
        self._closed = True
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(3000)
        self._loaded = ""
        self._ready = False

    def close(self):
        self._closed = True
        self._pending = None
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(3000)
