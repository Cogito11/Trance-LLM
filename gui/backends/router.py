"""Pick the right backend for a model and present them to the window as one engine."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..engine import EngineClient
from .llama_server import LlamaServerBackend

_SIGNALS = ("delta", "started", "finished", "failed", "state_changed", "ready", "loaded")


def backend_kind(model: str) -> str:
    """"gguf" for llama.cpp models, "trance" for the native engine's checkpoints."""
    return "gguf" if str(model).lower().endswith(".gguf") else "trance"


class BackendRouter(QObject):
    delta = Signal(str)
    started = Signal(dict)
    finished = Signal(dict)
    failed = Signal(str)
    state_changed = Signal(str)
    ready = Signal(str)
    loaded = Signal(dict)

    def __init__(self, parent=None, *, binary: Path | None = None, resolver=lambda: None):
        super().__init__(parent)
        self.trance = EngineClient(self, binary=binary)
        self.llama = LlamaServerBackend(resolver, self)
        self._active = self.trance
        for backend in (self.trance, self.llama):
            for name in _SIGNALS:
                getattr(backend, name).connect(getattr(self, name))

    @property
    def backends(self):
        return (self.trance, self.llama)

    @property
    def busy(self) -> bool:
        return any(b.busy for b in self.backends)

    def generate(self, model: str, messages: list[dict], settings: dict):
        if self.busy:
            raise RuntimeError("The engine is already working")
        chosen = self.llama if backend_kind(model) == "gguf" else self.trance
        for other in self.backends:           # only one backend may hold the GPU at a time
            if other is not chosen: other.release()
        self._active = chosen
        chosen.generate(model, messages, settings)

    def cancel(self):
        self._active.cancel()

    def release(self):
        """Free all GPU memory (before training); callers must be idle."""
        if self.busy:
            raise RuntimeError("Stop generation before starting training")
        for backend in self.backends: backend.release()

    def close(self):
        for backend in self.backends: backend.close()
