"""Pick the right backend for a model and present them to the window as one engine."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..engine import EngineClient
from .llama_server import LlamaServerBackend

_SIGNALS = ("delta", "thinking", "started", "finished", "failed", "state_changed", "ready", "loaded")
GGUF_ONLY_SETTINGS = ("thinking",)


def backend_kind(model: str) -> str:
    """"gguf" for llama.cpp models, "trance" for the native engine's checkpoints."""
    return "gguf" if str(model).lower().endswith(".gguf") else "trance"


class BackendRouter(QObject):
    delta = Signal(str)
    thinking = Signal(str)
    started = Signal(dict)
    finished = Signal(dict)
    failed = Signal(str)
    state_changed = Signal(str)
    ready = Signal(str)
    loaded = Signal(dict)
    unloaded = Signal()

    def __init__(self, parent=None, *, binary: Path | None = None, resolver=lambda: None):
        super().__init__(parent)
        self.trance = EngineClient(self, binary=binary)
        self.llama = LlamaServerBackend(resolver, self)
        self._active = self.trance
        self.loaded_model = ""          # path of the model currently held in memory, or ""
        self._loaded_backend = None
        for backend in (self.trance, self.llama):
            backend.loaded.connect(self._note_loaded)   # connected first, so listeners of `loaded` already see loaded_model
            for name in _SIGNALS:
                if hasattr(backend, name):               # the native engine has no thinking channel
                    getattr(backend, name).connect(getattr(self, name))
        self.llama.unloaded.connect(lambda: self._clear_loaded(self.llama))
        self.trance.unloaded.connect(lambda: self._clear_loaded(self.trance))

    def _note_loaded(self, info: dict):
        self.loaded_model = info["model"]
        self._loaded_backend = self.llama if backend_kind(info["model"]) == "gguf" else self.trance

    def _clear_loaded(self, backend=None):
        if self.loaded_model and (backend is None or backend is self._loaded_backend):
            self.loaded_model, self._loaded_backend = "", None
            self.unloaded.emit()

    @property
    def backends(self):
        return (self.trance, self.llama)

    @property
    def busy(self) -> bool:
        return any(b.busy for b in self.backends)

    def _claim(self, model: str):
        """Make `model`'s backend the only one holding memory, and forget a model that is about to be replaced."""
        chosen = self.llama if backend_kind(model) == "gguf" else self.trance
        for other in self.backends:           # only one backend may hold the GPU at a time
            if other is not chosen:
                self._clear_loaded(other)
                other.release()
        if self.loaded_model and self.loaded_model != str(Path(model).resolve()):
            self._clear_loaded()              # one model at a time: the old one is unloaded, even if the new one then fails
        self._active = chosen
        return chosen

    def load(self, model: str):
        """Load a model now, replacing whatever is loaded; `loaded` fires when it is ready, `failed` if it cannot be."""
        if self.busy:
            raise RuntimeError("The engine is already working")
        self._claim(model).load(model)

    def generate(self, model: str, messages: list[dict], settings: dict):
        if self.busy:
            raise RuntimeError("The engine is already working")
        chosen = self._claim(model)
        if chosen is self.trance: settings = {k: v for k, v in settings.items() if k not in GGUF_ONLY_SETTINGS}   # the native engine takes every field literally
        chosen.generate(model, messages, settings)

    def cancel(self):
        self._active.cancel()

    def release(self):
        """Free all GPU memory (before training); callers must be idle."""
        if self.busy:
            raise RuntimeError("Stop generation before starting training")
        for backend in self.backends: backend.release()
        self._clear_loaded()

    def close(self):
        for backend in self.backends: backend.close()
