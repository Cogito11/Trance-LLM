"""Settings page: the color theme, the model loaded at launch, and the runtimes (the Trance engine and llama.cpp)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from .theme import PALETTES, PALETTE_INFO


class ThemeCard(QPushButton):
    """A selectable card previewing a palette's accent colors."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.theme_name = name
        title, blurb = PALETTE_INFO[name]
        self.setObjectName("themeCard"); self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor); self.setMinimumSize(190, 120)
        self.setAccessibleName(f"{title} theme")
        box = QVBoxLayout(self); box.setContentsMargins(16, 14, 16, 14); box.setSpacing(8)
        swatches = QHBoxLayout(); swatches.setSpacing(6)
        for key in ("accent", "accent_hi", "accent_2"):
            chip = QFrame(); chip.setFixedSize(26, 26)
            chip.setStyleSheet(f"background:{PALETTES[name][key]};border-radius:13px;border:1px solid rgba(255,255,255,0.14);")
            swatches.addWidget(chip)
        swatches.addStretch(); box.addLayout(swatches)
        name_label = QLabel(title); name_label.setObjectName("themeName"); box.addWidget(name_label)
        blurb_label = QLabel(blurb); blurb_label.setObjectName("muted"); blurb_label.setWordWrap(True); box.addWidget(blurb_label)
        box.addStretch(1)
        for child in self.findChildren(QWidget):  # clicks anywhere on the card select it
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


class RuntimeRow(QFrame):
    """One installed runtime: what it is, where it lives, and what you can do with it."""
    open_requested = Signal(str)
    use_requested = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, entry, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.setObjectName("runtimeRow")
        box = QHBoxLayout(self); box.setContentsMargins(14, 10, 12, 10); box.setSpacing(10)
        text = QVBoxLayout(); text.setSpacing(2)
        top = QHBoxLayout(); top.setSpacing(8)
        self.name_label = QLabel(entry.name); self.name_label.setObjectName("rowName"); top.addWidget(self.name_label)
        self.badge = QLabel("In use"); self.badge.setObjectName("badge"); self.badge.setVisible(entry.active); top.addWidget(self.badge)
        top.addStretch(1); text.addLayout(top)
        size = f"  \u00b7  {entry.size_bytes / 1e6:,.0f} MB" if entry.size_bytes else ""
        self.path_label = QLabel(str(entry.path) + size); self.path_label.setObjectName("muted"); self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); self.path_label.setToolTip(str(entry.path))
        text.addWidget(self.path_label); box.addLayout(text, 1)
        self.use_button = QPushButton("Use"); self.open_button = QPushButton("Open folder")
        self.remove_button = QPushButton("Remove" if entry.removable else "Stop using")
        for b in (self.use_button, self.open_button, self.remove_button):
            b.setProperty("compact", True); box.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
        self.use_button.setVisible(entry.source == "managed" and not entry.active)
        self.remove_button.setVisible(entry.removable or entry.source == "custom")
        self.use_button.clicked.connect(lambda: self.use_requested.emit(entry.key))
        self.open_button.clicked.connect(lambda: self.open_requested.emit(str(entry.folder)))
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(entry.key))


class ElidedLabel(QLabel):
    """A one-line label that cuts long text (a file path) with an ellipsis instead of widening the page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def set_full_text(self, text: str, tooltip: str = ""):
        self._full = text; self.setToolTip(tooltip or text); self._elide()

    def full_text(self) -> str:
        return self._full

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, max(20, self.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()


class EngineCard(QFrame):
    """The built-in Trance CUDA engine. Read-only: it ships with the app, so there is nothing to install or choose."""
    GPU_UNKNOWN = "Detected the first time a Trance model loads"

    def __init__(self, binary: Path, bundled: bool = False, parent=None):
        super().__init__(parent)
        self.binary, self.bundled, self._gpu = Path(binary), bundled, ""
        self.setObjectName("settingsCard")
        box = QVBoxLayout(self); box.setContentsMargins(18, 16, 18, 16); box.setSpacing(10)
        head = QHBoxLayout(); head.setSpacing(10)
        title = QLabel("Trance engine"); title.setObjectName("themeName"); head.addWidget(title)
        badge = QLabel("Built in"); badge.setObjectName("badge"); head.addWidget(badge); head.addStretch(1); box.addLayout(head)
        blurb = QLabel("Runs Trance models on your NVIDIA GPU with CUDA. It comes with the app, so there is nothing to download or choose.")
        blurb.setObjectName("muted"); blurb.setWordWrap(True); box.addWidget(blurb)
        grid = QGridLayout(); grid.setHorizontalSpacing(16); grid.setVerticalSpacing(6); grid.setColumnStretch(1, 1)
        self.status = QLabel(); self.location = ElidedLabel(); self.gpu = QLabel(); self.gpu.setWordWrap(True)
        for row, (name, widget) in enumerate((("Status", self.status), ("Location", self.location), ("GPU", self.gpu))):
            caption = QLabel(name); caption.setObjectName("muted"); grid.addWidget(caption, row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(widget, row, 1)
        box.addLayout(grid)
        self.refresh()

    def set_gpu(self, text: str):
        """The engine reports its GPU when it starts, e.g. `NVIDIA GeForce RTX 4060 (compute capability 8.9)`."""
        self._gpu = text.split(" (compute")[0].strip()
        self.refresh()

    def refresh(self):
        found = self.binary.is_file()
        self.status.setText("Ready to use" if found else "Not found. Build it first: nvcc -O3 -o trance trance.cu -lcublas (see the README).")
        self.status.setObjectName("" if found else "configHint"); self.status.setProperty("bad", not found)
        self.status.style().unpolish(self.status); self.status.style().polish(self.status)
        self.location.set_full_text("Bundled with the app" if self.bundled and found else str(self.binary), str(self.binary))
        self.gpu.setText(self._gpu or (self.GPU_UNKNOWN if found else "Unavailable"))

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()                                  # the binary may have been built since the app started


class RuntimeCard(QFrame):
    """Download a llama.cpp runtime, and see and manage every runtime the app can use."""

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.rows: dict = {}
        self.setObjectName("settingsCard")
        box = QVBoxLayout(self); box.setContentsMargins(18, 16, 18, 16); box.setSpacing(10)
        title = QLabel("llama.cpp runtime"); title.setObjectName("themeName"); box.addWidget(title)
        blurb = QLabel("Runs GGUF models with the same engine LM Studio uses. Trance models run on the built-in CUDA engine and do not need this.")
        blurb.setObjectName("muted"); blurb.setWordWrap(True); box.addWidget(blurb)
        row = QHBoxLayout(); row.setSpacing(8)
        self.flavor = QComboBox(); row.addWidget(self.flavor, 1)
        self.install_button = QPushButton("Download"); self.install_button.setObjectName("primary"); row.addWidget(self.install_button)
        box.addLayout(row)
        self.flavor_detail = QLabel(); self.flavor_detail.setObjectName("muted"); self.flavor_detail.setWordWrap(True); box.addWidget(self.flavor_detail)
        self.progress = QProgressBar(); self.progress.setVisible(False); box.addWidget(self.progress)
        self.error = QLabel(); self.error.setObjectName("configHint"); self.error.setProperty("bad", True); self.error.setWordWrap(True)
        self.error.setVisible(False); box.addWidget(self.error)
        header = QHBoxLayout(); header.setSpacing(8)
        heading = QLabel("INSTALLED"); heading.setObjectName("eyebrow"); header.addWidget(heading); header.addStretch(1)
        self.choose_button = QPushButton("Choose existing\u2026"); self.choose_button.setProperty("compact", True); header.addWidget(self.choose_button)
        box.addSpacing(6); box.addLayout(header)
        self.list_box = QVBoxLayout(); self.list_box.setSpacing(8); box.addLayout(self.list_box)
        self.empty = QLabel(); self.empty.setObjectName("muted"); self.empty.setWordWrap(True); box.addWidget(self.empty)
        note = QLabel("Downloading contacts github.com. Nothing else in this app uses the network. Files are checked against "
                      "SHA-256 hashes built into the app before they are unpacked.")
        note.setObjectName("muted"); note.setWordWrap(True); box.addWidget(note)
        self.flavor.currentIndexChanged.connect(self._flavor_changed)
        self.install_button.clicked.connect(self._install_clicked)
        self.choose_button.clicked.connect(self._choose)
        manager.changed.connect(self.refresh); manager.progress.connect(self._progress); manager.install_failed.connect(self._failed)
        for flavor in manager.flavors:
            self.flavor.addItem(f"{flavor.label}  \u00b7  ~{flavor.approx_mb} MB", flavor.id)
        recommended = self.flavor.findData(manager.recommended()) if manager.flavors else -1
        if recommended >= 0:
            self.flavor.setCurrentIndex(recommended); self.flavor.setItemText(recommended, self.flavor.itemText(recommended) + "  (recommended)")
        self.refresh()

    def _flavor_changed(self):
        flavor = next((f for f in self.manager.flavors if f.id == self.flavor.currentData()), None)
        self.flavor_detail.setText(flavor.detail if flavor else "")

    def refresh(self):
        installing = self.manager.installing
        self.install_button.setText("Cancel" if installing else "Download")
        self.install_button.setEnabled(bool(self.manager.flavors))
        self.flavor.setEnabled(not installing and bool(self.manager.flavors))
        self.choose_button.setEnabled(not installing)
        self.progress.setVisible(installing)
        if installing: self.error.setVisible(False)
        self._flavor_changed()
        while self.list_box.count():
            item = self.list_box.takeAt(0)
            if item.widget() is not None: item.widget().setParent(None)
        self.rows = {}
        entries = self.manager.installed()
        for entry in entries:
            widget = RuntimeRow(entry)
            widget.open_requested.connect(self._open); widget.use_requested.connect(self.manager.use)
            widget.remove_requested.connect(self._remove); widget.setEnabled(not installing)
            self.list_box.addWidget(widget); self.rows[entry.key] = widget
        self.empty.setVisible(not entries)
        self.empty.setText("Nothing installed yet. GGUF models need a llama.cpp runtime: download one above or choose a llama-server you already have."
                           if self.manager.flavors else "No prebuilt runtime is available for this system. Choose a llama-server you already have.")

    def _install_clicked(self):
        if self.manager.installing: self.manager.cancel_install(); return
        self.error.setVisible(False); self.progress.setRange(0, 0)
        self.manager.start_install(self.flavor.currentData())

    def _progress(self, done, total, text):
        if total > 0:
            self.progress.setRange(0, total); self.progress.setValue(done)
            self.progress.setFormat(f"{text}  \u00b7  {done/1e6:.0f} / {total/1e6:.0f} MB")
        else:
            self.progress.setRange(0, 0); self.progress.setFormat(text)

    def _failed(self, message):
        self.error.setText(message); self.error.setVisible(True)
        self.error.style().unpolish(self.error); self.error.style().polish(self.error)

    def _choose(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose llama-server", str(Path.home()), "")
        if path: self.manager.set_custom(path)

    def _open(self, folder):
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _remove(self, key):
        if key == "custom": self.manager.clear_custom()
        else: self.manager.remove_install(key)


class DefaultModelCard(QFrame):
    """Choose the model that loads by itself when the app starts."""
    selected = Signal(str)                      # a model path, or "" for none
    NONE = "None  \u00b7  start with no model loaded"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsCard")
        box = QVBoxLayout(self); box.setContentsMargins(18, 16, 18, 16); box.setSpacing(10)
        title = QLabel("Default model"); title.setObjectName("themeName"); box.addWidget(title)
        blurb = QLabel("Loaded automatically each time the app starts, so it is ready to chat straight away. "
                       "A loaded model uses memory (GPU memory for most models); choose None to start empty and pick one yourself.")
        blurb.setObjectName("muted"); blurb.setWordWrap(True); box.addWidget(blurb)
        self.combo = QComboBox(); self.combo.setAccessibleName("Default model"); box.addWidget(self.combo)
        self.combo.activated.connect(lambda _index: self.selected.emit(self.combo.currentData() or ""))

    def set_models(self, models: list[tuple[str, str]], selected: str = ""):
        """`models` is a list of (path, label). A saved default whose file has gone is still shown, so merely opening
        Settings never forgets it."""
        self.combo.blockSignals(True); self.combo.clear()
        self.combo.addItem(self.NONE, "")
        for path, text in models: self.combo.addItem(text, path)
        if selected and self.combo.findData(selected) < 0:
            self.combo.addItem(f"{Path(selected).name}  \u00b7  not found", selected)
        self.combo.setCurrentIndex(max(0, self.combo.findData(selected)))
        self.combo.blockSignals(False)


class SettingsPage(QWidget):
    theme_selected = Signal(str)
    default_model_selected = Signal(str)

    def __init__(self, current: str = "teal", parent=None, *, runtimes=None, engine_binary=None, engine_bundled=False):
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setObjectName("formScroll"); scroll.setWidgetResizable(True); outer.addWidget(scroll)
        content = QWidget(); scroll.setWidget(content)
        layout = QVBoxLayout(content); layout.setContentsMargins(30, 28, 30, 24); layout.setSpacing(14)
        title = QLabel("Settings"); title.setObjectName("title"); layout.addWidget(title)
        eyebrow = QLabel("APPEARANCE"); eyebrow.setObjectName("eyebrow"); layout.addWidget(eyebrow)
        intro = QLabel("Choose a color theme. It applies immediately and is remembered next time."); intro.setObjectName("muted"); intro.setWordWrap(True)
        layout.addWidget(intro)
        row = QHBoxLayout(); row.setSpacing(12)
        self.group = QButtonGroup(self); self.group.setExclusive(True); self.cards = {}
        for name in PALETTES:
            card = ThemeCard(name); self.group.addButton(card); self.cards[name] = card; row.addWidget(card, 1)
            card.clicked.connect(lambda checked=False, n=name: self.theme_selected.emit(n))
        layout.addLayout(row)
        heading = QLabel("MODELS"); heading.setObjectName("eyebrow"); layout.addSpacing(8); layout.addWidget(heading)
        self.default_model_card = DefaultModelCard(); layout.addWidget(self.default_model_card)
        self.default_model_card.selected.connect(self.default_model_selected)
        self.engine_card = self.runtime_card = None
        if runtimes is not None or engine_binary is not None:
            heading = QLabel("RUNTIMES"); heading.setObjectName("eyebrow"); layout.addSpacing(8); layout.addWidget(heading)
        if engine_binary is not None:
            self.engine_card = EngineCard(engine_binary, engine_bundled); layout.addWidget(self.engine_card)
        if runtimes is not None:
            self.runtime_card = RuntimeCard(runtimes); layout.addWidget(self.runtime_card)
        layout.addStretch(1)
        self.select(current)

    def set_models(self, models: list[tuple[str, str]], default: str = ""):
        """Refresh the models offered as the launch default."""
        self.default_model_card.set_models(models, default)

    def select(self, name: str):
        """Mark a theme as current without emitting theme_selected."""
        if name in self.cards: self.cards[name].setChecked(True)
