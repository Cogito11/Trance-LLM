"""The popup shown when a GGUF model is picked but no llama.cpp runtime is installed.

It offers to download the recommended runtime right there (and the window then loads the model that was picked), or to
go to the Runtimes page. The window decides what each button does; this widget only shows the choices and the download's progress.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from .modal import ModalOverlay


class RuntimePrompt(ModalOverlay):
    install_requested = Signal()
    cancel_install_requested = Signal()
    runtimes_requested = Signal()

    CARD_WIDTH = 480

    def __init__(self, parent):
        super().__init__(parent)
        self.installing = False
        self._model = ""
        box = QVBoxLayout(self.card); box.setContentsMargins(24, 22, 24, 20); box.setSpacing(12)
        self.title = QLabel(); self.title.setObjectName("modalTitle"); self.title.setWordWrap(True); box.addWidget(self.title)
        self.body = QLabel(); self.body.setWordWrap(True); box.addWidget(self.body)
        self.detail = QLabel(); self.detail.setObjectName("muted"); self.detail.setWordWrap(True); box.addWidget(self.detail)
        self.progress = QProgressBar(); self.progress.setVisible(False); box.addWidget(self.progress)
        self.error = QLabel(); self.error.setObjectName("configHint"); self.error.setProperty("bad", True)
        self.error.setWordWrap(True); self.error.setVisible(False); box.addWidget(self.error)
        box.addSpacing(4)
        row = QHBoxLayout(); row.setSpacing(8)
        self.runtimes_button = QPushButton("Open Runtimes"); self.cancel_button = QPushButton("Cancel")
        self.install_button = QPushButton("Download"); self.install_button.setObjectName("primary")
        for b in (self.runtimes_button, self.cancel_button, self.install_button): b.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addWidget(self.runtimes_button); row.addStretch(1); row.addWidget(self.cancel_button); row.addWidget(self.install_button)
        box.addLayout(row)
        self.install_button.clicked.connect(self.install_requested)
        self.runtimes_button.clicked.connect(self._runtimes)
        self.cancel_button.clicked.connect(self._cancel)

    # ---- states
    def offer(self, model: str, *, flavor: str = "", size_mb: int = 0):
        """Explain what is missing. With no `flavor` there is nothing to download on this system, only the Runtimes page."""
        self.installing, self._model = False, model
        self.title.setText("Install the llama.cpp runtime")
        self.body.setText(f"\u201c{model}\u201d is a GGUF model. GGUF models run on the llama.cpp runtime, which isn\u2019t installed yet.")
        if flavor:
            self.detail.setText(f"Download it now ({flavor}, about {size_mb} MB) and {model} will load as soon as it\u2019s ready. "
                                "This contacts github.com; the files are checked against SHA-256 hashes built into the app before they are unpacked.")
        else:
            self.detail.setText("No prebuilt runtime is available for this system. On the Runtimes page you can choose a llama-server you already have.")
        self.install_button.setText(f"Download ({size_mb} MB)" if size_mb else "Download")
        self._show(downloadable=bool(flavor))
        self.error.setVisible(False)

    def set_installing(self):
        self.installing = True
        self.title.setText("Downloading the llama.cpp runtime")
        self.body.setText(f"Installing it now so \u201c{self._model}\u201d can load." if self._model else "Installing the runtime.")
        self.error.setVisible(False)
        self.progress.setRange(0, 0); self.progress.setFormat("Starting\u2026"); self.progress.setVisible(True)
        self.cancel_button.setText("Cancel download")
        for b in (self.install_button, self.runtimes_button): b.setVisible(False)
        self.detail.setText("The model loads as soon as the runtime is ready.")

    def show_progress(self, done: int, total: int, text: str):
        if not self.installing: return
        if total > 0:
            self.progress.setRange(0, total); self.progress.setValue(done)
            self.progress.setFormat(f"{text}  \u00b7  {done / 1e6:.0f} / {total / 1e6:.0f} MB")
        else:
            self.progress.setRange(0, 0); self.progress.setFormat(text)

    def show_error(self, message: str):
        """The download failed: say why and let the person try again. (A cancelled download reports no message.)"""
        if not message or not self.isVisible(): return
        self.installing = False
        self.title.setText("Couldn\u2019t install the runtime")
        self.progress.setVisible(False)
        self.error.setText(message); self.error.setVisible(True)
        self.error.style().unpolish(self.error); self.error.style().polish(self.error)
        self.install_button.setText("Try again")
        self._show(downloadable=True)

    def finish(self):
        """The runtime is installed: close without asking anything."""
        self.installing = False
        self._close()

    def _show(self, *, downloadable: bool):
        self.progress.setVisible(False)
        self.cancel_button.setText("Cancel")
        self.install_button.setVisible(downloadable)
        self.runtimes_button.setVisible(True)
        self.runtimes_button.setObjectName("primary" if not downloadable else "")
        self.runtimes_button.style().unpolish(self.runtimes_button); self.runtimes_button.style().polish(self.runtimes_button)

    # ---- buttons
    def _runtimes(self):
        self._close()
        self.runtimes_requested.emit()

    def _cancel(self):
        if self.installing:
            self.installing = False
            self.cancel_install_requested.emit()
        self._close()

    # ---- showing
    def can_dismiss(self):
        return not self.installing                    # Escape and the backdrop must not abandon a download by accident

    def _card_height(self, width: int) -> int:
        layout = self.card.layout()
        wanted = layout.totalHeightForWidth(width) if layout.hasHeightForWidth() else layout.sizeHint().height()
        return max(180, min(wanted, self.height() - 2 * self.MARGIN))
