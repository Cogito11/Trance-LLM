"""Chat with GGUF models through a managed local llama.cpp ``llama-server`` process.

One server runs at a time, bound to 127.0.0.1 on a free port and protected by a
random per-launch API key. It stays loaded between messages for speed and is
stopped when another backend needs the GPU, before training, and on exit.
"""
from __future__ import annotations

import http.client
import secrets
import socket
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, QTimer, Signal

from ..gguf import read_gguf_info
from ..runtime import external_environment
from .openai_stream import ChatStream, build_payload

NO_RUNTIME = ("GGUF models need the llama.cpp runtime. Install it in Tools \u2192 Settings \u2192 Runtimes, "
              "or choose an existing llama-server there.")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _HealthWaiter(QThread):
    ready = Signal()
    failed = Signal(str)

    def __init__(self, port: int, timeout: float, parent=None):
        super().__init__(parent)
        self.port, self.timeout = port, timeout
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        deadline = time.monotonic() + self.timeout
        while not self._cancel.is_set() and time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1.5)
                conn.request("GET", "/health")
                status = conn.getresponse().status
                conn.close()
                if status == 200:
                    self.ready.emit(); return
            except (OSError, http.client.HTTPException):
                pass   # not listening yet, or still loading (503)
            self._cancel.wait(0.25)
        if not self._cancel.is_set():
            self.failed.emit("The model took too long to load.")


class LlamaServerBackend(QObject):
    delta = Signal(str)
    started = Signal(dict)
    finished = Signal(dict)
    failed = Signal(str)
    state_changed = Signal(str)
    ready = Signal(str)
    loaded = Signal(dict)

    def __init__(self, resolver, parent=None, *, context_cap: int = 8192, load_timeout: float = 900):
        super().__init__(parent)
        self._resolver = resolver            # () -> RuntimeInfo | None
        self.context_cap, self.load_timeout = context_cap, load_timeout
        self.process: QProcess | None = None
        self._port = 0
        self._token = ""
        self._model = ""
        self._pending: dict | None = None
        self._phase = "idle"
        self._cancelled = False
        self._stopping = False
        self._errors = ""
        self._stream: ChatStream | None = None
        self._waiter: _HealthWaiter | None = None
        self._runtime_label = ""
        self._context = 0

    @property
    def busy(self) -> bool:
        return self._pending is not None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    # ---- public surface (mirrors EngineClient)
    def generate(self, model: str, messages: list[dict], settings: dict):
        if self.busy:
            raise RuntimeError("The engine is already working")
        model = str(Path(model).resolve())
        self._pending = {"model": model, "messages": messages, "settings": settings}
        self._cancelled, self._errors = False, ""
        info = self._resolver()
        if info is None:
            QTimer.singleShot(0, lambda: self._fail(NO_RUNTIME)); return
        if self.running and self._model == model and self._phase == "idle":
            self._request()
        else:
            self._start_server(info, model)

    def cancel(self):
        if not self.busy: return
        self._cancelled = True
        if self._stream is not None and self._phase == "generating":
            self.state_changed.emit("Stopping response\u2026")
            self._stream.cancel()
        else:                                   # still loading: abandon the server
            self._stop_process()
            self._finish({"reason": "cancelled", "generated_tokens": 0, "elapsed_seconds": 0})

    def release(self):
        """Free GPU memory (before training or when another backend needs it); callers must be idle."""
        if self.busy:
            raise RuntimeError("Stop generation before starting training")
        self._stop_process()

    def close(self):
        self._pending = None
        if self._stream is not None:
            self._stream.cancel(); self._stream.wait(2000)
        self._stop_process()

    # ---- server lifecycle
    def _context_for(self, model: str) -> int:
        try: native = read_gguf_info(Path(model))["context_length"]
        except ValueError: native = 0
        return min(self.context_cap, native) if native else self.context_cap

    def _start_server(self, info, model: str):
        self._stop_process()
        self._port, self._token = free_port(), secrets.token_urlsafe(18)
        self._context = self._context_for(model)
        args = ["-m", model, "--host", "127.0.0.1", "--port", str(self._port), "--api-key", self._token,
                "-c", str(self._context), "-np", "1", "--no-webui"]
        proc = QProcess(self)
        proc.setProgram(str(info.path)); proc.setArguments(args); proc.setWorkingDirectory(str(info.path.parent))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment()
        for key, value in external_environment().items(): environment.insert(key, value)
        proc.setProcessEnvironment(environment)
        proc.readyReadStandardOutput.connect(self._collect)
        proc.errorOccurred.connect(self._process_error)
        proc.finished.connect(self._exited)
        self.process, self._stopping, self._phase = proc, False, "loading"
        self._runtime_label = "llama.cpp " + " \u00b7 ".join(p for p in (info.version, info.flavor) if p)
        self.state_changed.emit("Loading model\u2026")
        proc.start()
        self._waiter = _HealthWaiter(self._port, self.load_timeout, self)
        self._waiter.ready.connect(lambda m=model: self._server_ready(m))
        self._waiter.failed.connect(self._load_failed)
        self._waiter.start()

    def _server_ready(self, model: str):
        if not self.busy or self._phase != "loading": return
        self._model, self._phase = model, "idle"
        self.loaded.emit({"model": model, "parameters": 0})
        self.ready.emit(self._runtime_label.strip())
        self._request()

    def _request(self):
        pending = self._pending
        if pending is None: return
        self._phase = "preparing"
        self.state_changed.emit("Preparing context\u2026")
        payload = build_payload(Path(pending["model"]).name, pending["messages"], pending["settings"])
        stream = ChatStream(f"http://127.0.0.1:{self._port}/v1/chat/completions", payload,
                            {"Authorization": f"Bearer {self._token}"}, parent=self)
        stream.started_stream.connect(self._stream_started)
        stream.delta.connect(self.delta)
        stream.done.connect(self._stream_done)
        stream.failed.connect(self._stream_failed)
        self._stream = stream
        stream.start()

    def _stream_started(self):
        self._phase = "generating"
        self.state_changed.emit("Generating\u2026")
        self.started.emit({})

    def _stream_done(self, info: dict):
        self._reap_stream()
        self._finish({**info, "context_length": self._context})

    def _stream_failed(self, message: str):
        self._reap_stream()
        if not self.running:
            message = self._server_message("The llama.cpp server stopped unexpectedly.")
        self._fail(message)

    def _reap_stream(self):
        if self._stream is not None:
            self._stream.wait(2000); self._stream = None

    def _finish(self, info: dict):
        if self._pending is None: return
        self._pending, self._phase = None, "idle"
        self.finished.emit(info)

    def _fail(self, message: str):
        was_busy = self._pending is not None
        self._pending, self._phase = None, "idle"
        if was_busy: self.failed.emit(message)

    # ---- process events
    def _collect(self):
        if self.process is not None:
            text = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
            self._errors = (self._errors + text)[-12000:]

    def _server_message(self, prefix: str) -> str:
        self._collect()
        lines = [l.strip() for l in self._errors.splitlines() if l.strip()]
        important = [l for l in lines if "error" in l.lower() or "failed" in l.lower() or "unknown" in l.lower()]
        detail = " ".join((important or lines)[-2:])
        return f"{prefix} {detail}".strip()

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._fail(f"Cannot start the llama.cpp server at {self.process.program()}.")

    def _exited(self, code, status):
        if self._stopping: return
        self._model = ""
        if self._waiter is not None: self._waiter.cancel()
        if self.busy and self._phase == "loading" and not self._cancelled:
            self._fail(self._server_message(f"llama-server could not load this model (exit {code})."))

    def _load_failed(self, message: str):
        if self.busy and self._phase == "loading":
            self._stop_process(); self._fail(message)

    def _stop_process(self):
        if self._waiter is not None:
            self._waiter.cancel(); self._waiter.wait(2000); self._waiter = None
        proc, self.process, self._model = self.process, None, ""
        if proc is not None and proc.state() != QProcess.ProcessState.NotRunning:
            self._stopping = True
            proc.terminate()
            if not proc.waitForFinished(2500):
                proc.kill(); proc.waitForFinished(2500)
        if proc is not None:
            proc.deleteLater()
