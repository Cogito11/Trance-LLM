"""The Settings page (color theme and the model loaded at launch) and the Runtimes page (the Trance engine and llama.cpp)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QButtonGroup, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from .theme import PALETTES, PALETTE_INFO
from .widgets import NoWheelComboBox


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
    """One llama.cpp runtime. Installed, it shows where it lives and offers Use / Open folder / Remove; not installed
    yet, it shows what it is and offers only Download (there is no folder to open and nothing to remove)."""
    open_requested = Signal(str)
    use_requested = Signal(str)
    remove_requested = Signal(str)
    download_requested = Signal(str)            # a flavor id
    cancel_requested = Signal()

    def __init__(self, name, *, entry=None, flavor=None, recommended=False, parent=None):
        """`entry` is the InstalledRuntime when it is on disk; `flavor` is the downloadable catalog entry, if any."""
        super().__init__(parent)
        self.entry, self.flavor = entry, flavor
        self.installed = entry is not None
        self.setObjectName("runtimeRow")
        box = QHBoxLayout(self); box.setContentsMargins(14, 10, 12, 10); box.setSpacing(10)
        text = QVBoxLayout(); text.setSpacing(2)
        top = QHBoxLayout(); top.setSpacing(8)
        self.name_label = QLabel(name); self.name_label.setObjectName("rowName"); top.addWidget(self.name_label)
        self.badge = QLabel("In use"); self.badge.setObjectName("badge"); self.badge.setVisible(bool(entry and entry.active)); top.addWidget(self.badge)
        self.recommended_badge = QLabel("Recommended"); self.recommended_badge.setObjectName("badge"); self.recommended_badge.setVisible(recommended)
        top.addWidget(self.recommended_badge)
        top.addStretch(1); text.addLayout(top)
        if entry is not None:
            size = f"  \u00b7  {entry.size_bytes / 1e6:,.0f} MB" if entry.size_bytes else ""
            detail, tip = str(entry.path) + size, str(entry.path)
        else:
            detail = f"{flavor.detail}  \u00b7  ~{flavor.approx_mb} MB download"; tip = detail
        self.path_label = QLabel(detail); self.path_label.setObjectName("muted"); self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); self.path_label.setToolTip(tip)
        text.addWidget(self.path_label)
        self.progress = QProgressBar(); self.progress.setVisible(False); text.addWidget(self.progress)
        box.addLayout(text, 1)
        self.download_button = QPushButton("Download"); self.download_button.setObjectName("primary")
        self.use_button = QPushButton("Use"); self.open_button = QPushButton("Open folder")
        self.remove_button = QPushButton("Remove" if entry and entry.removable else "Stop using")
        for b in (self.download_button, self.use_button, self.open_button, self.remove_button):
            b.setProperty("compact", True); box.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
        self.download_button.setVisible(not self.installed)
        self.use_button.setVisible(bool(entry and entry.source == "managed" and not entry.active))
        self.open_button.setVisible(self.installed)
        self.remove_button.setVisible(bool(entry and (entry.removable or entry.source == "custom")))
        if entry is not None:
            self.use_button.clicked.connect(lambda: self.use_requested.emit(entry.key))
            self.open_button.clicked.connect(lambda: self.open_requested.emit(str(entry.folder)))
            self.remove_button.clicked.connect(lambda: self.remove_requested.emit(entry.key))
        else:
            self.download_button.clicked.connect(self._download_clicked)
        self._downloading = False

    def _download_clicked(self):
        if self._downloading: self.cancel_requested.emit()
        else: self.download_requested.emit(self.flavor.id)

    def set_download_state(self, *, downloading: bool, locked: bool):
        """`downloading`: this row is the one being fetched (its button becomes Cancel). `locked`: another download is
        running, so everything else waits."""
        self._downloading = downloading
        self.download_button.setText("Cancel" if downloading else "Download")
        self.progress.setVisible(downloading)
        self.setEnabled(downloading or not locked)

    def set_progress(self, done: int, total: int, text: str):
        if total > 0:
            self.progress.setRange(0, total); self.progress.setValue(done)
            self.progress.setFormat(f"{text}  \u00b7  {done/1e6:.0f} / {total/1e6:.0f} MB")
        else:
            self.progress.setRange(0, 0); self.progress.setFormat(text)


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
    """The llama.cpp runtimes: every build this system can download is listed, installed or not, and runtimes that
    already exist on this computer can be located and used too."""

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.rows: dict = {}                    # row key -> RuntimeRow (catalog rows are keyed like their installed form)
        self._last_progress = (0, 0, "")
        self.setObjectName("settingsCard")
        box = QVBoxLayout(self); box.setContentsMargins(18, 16, 18, 16); box.setSpacing(10)
        title = QLabel("llama.cpp runtime"); title.setObjectName("themeName"); box.addWidget(title)
        blurb = QLabel("Runs GGUF models with the same engine LM Studio uses. Trance models run on the built-in CUDA engine and do not need this.")
        blurb.setObjectName("muted"); blurb.setWordWrap(True); box.addWidget(blurb)
        heading = QLabel("AVAILABLE RUNTIMES"); heading.setObjectName("eyebrow"); box.addSpacing(6); box.addWidget(heading)
        self.unavailable = QLabel("No prebuilt runtime is available for this system. Choose a llama-server you already have below.")
        self.unavailable.setObjectName("muted"); self.unavailable.setWordWrap(True); box.addWidget(self.unavailable)
        self.list_box = QVBoxLayout(); self.list_box.setSpacing(8); box.addLayout(self.list_box)
        self.error = QLabel(); self.error.setObjectName("configHint"); self.error.setProperty("bad", True); self.error.setWordWrap(True)
        self.error.setVisible(False); box.addWidget(self.error)
        header = QHBoxLayout(); header.setSpacing(8)
        found = QLabel("ALREADY ON THIS COMPUTER"); found.setObjectName("eyebrow"); header.addWidget(found); header.addStretch(1)
        self.choose_button = QPushButton("Choose existing\u2026"); self.choose_button.setProperty("compact", True); header.addWidget(self.choose_button)
        box.addSpacing(6); box.addLayout(header)
        self.external_box = QVBoxLayout(); self.external_box.setSpacing(8); box.addLayout(self.external_box)
        self.external_empty = QLabel("No existing runtime in use. If you already have llama.cpp, choose its llama-server file and the app will use it.")
        self.external_empty.setObjectName("muted"); self.external_empty.setWordWrap(True); box.addWidget(self.external_empty)
        note = QLabel("Downloading contacts github.com. Nothing else in this app uses the network. Files are checked against "
                      "SHA-256 hashes built into the app before they are unpacked.")
        note.setObjectName("muted"); note.setWordWrap(True); box.addWidget(note)
        self.choose_button.clicked.connect(self._choose)
        manager.changed.connect(self.refresh); manager.progress.connect(self._progress); manager.install_failed.connect(self._failed)
        self.refresh()

    @staticmethod
    def _clear(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None: item.widget().setParent(None)

    def _connect(self, row):
        row.open_requested.connect(self._open); row.use_requested.connect(self.manager.use); row.remove_requested.connect(self._remove)
        row.download_requested.connect(self._download); row.cancel_requested.connect(self.manager.cancel_install)

    def refresh(self):
        manager = self.manager
        installing, flavor_id = manager.installing, manager.installing_flavor
        self.choose_button.setEnabled(not installing)
        if installing: self.error.setVisible(False)
        self._clear(self.list_box); self._clear(self.external_box)
        self.rows = {}
        entries = {entry.key: entry for entry in manager.installed()}
        recommended = manager.recommended()
        used = set()
        for flavor in manager.flavors:                              # every build on offer, whether or not it is installed
            key = manager.install_key(flavor.id); entry = entries.get(key); used.add(key)
            row = RuntimeRow(flavor.label, entry=entry, flavor=flavor, recommended=flavor.id == recommended)
            row.name_label.setToolTip(flavor.detail)
            self._connect(row)
            if entry is None:
                downloading = installing and flavor.id == flavor_id
                row.set_download_state(downloading=downloading, locked=installing)
                if downloading: row.set_progress(*self._last_progress)
            else:
                row.setEnabled(not installing)
            self.list_box.addWidget(row); self.rows[key] = row
        for key, entry in entries.items():                          # other downloads (an older build, say) stay manageable
            if key in used or entry.source != "managed": continue
            row = RuntimeRow(entry.name, entry=entry); self._connect(row); row.setEnabled(not installing)
            self.list_box.addWidget(row); self.rows[key] = row
        for key, entry in entries.items():                          # runtimes that were not downloaded here
            if entry.source == "managed": continue
            row = RuntimeRow(entry.name, entry=entry); self._connect(row); row.setEnabled(not installing)
            self.external_box.addWidget(row); self.rows[key] = row
        self.unavailable.setVisible(not manager.flavors)
        self.external_empty.setVisible(not any(e.source != "managed" for e in entries.values()))

    def _download(self, flavor_id):
        self.error.setVisible(False); self._last_progress = (0, 0, "Starting\u2026")
        self.manager.start_install(flavor_id)

    def _progress(self, done, total, text):
        self._last_progress = (done, total, text)
        row = self.rows.get(self.manager.install_key(self.manager.installing_flavor)) if self.manager.installing_flavor else None
        if row is not None: row.set_progress(done, total, text)

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
        self.combo = NoWheelComboBox(); self.combo.setAccessibleName("Default model"); box.addWidget(self.combo)
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


def _scrolling_page(page: QWidget, title: str) -> QVBoxLayout:
    """Give a page the standard scrolling form layout with its title; returns the layout to fill."""
    outer = QVBoxLayout(page); outer.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea(); scroll.setObjectName("formScroll"); scroll.setWidgetResizable(True); outer.addWidget(scroll)
    content = QWidget(); scroll.setWidget(content)
    layout = QVBoxLayout(content); layout.setContentsMargins(30, 28, 30, 24); layout.setSpacing(14)
    heading = QLabel(title); heading.setObjectName("title"); layout.addWidget(heading)
    return layout


class SettingsPage(QWidget):
    """Appearance and the model that loads at launch."""
    theme_selected = Signal(str)
    default_model_selected = Signal(str)

    def __init__(self, current: str = "teal", parent=None):
        super().__init__(parent)
        layout = _scrolling_page(self, "Settings")
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
        layout.addStretch(1)
        self.select(current)

    def set_models(self, models: list[tuple[str, str]], default: str = ""):
        """Refresh the models offered as the launch default."""
        self.default_model_card.set_models(models, default)

    def select(self, name: str):
        """Mark a theme as current without emitting theme_selected."""
        if name in self.cards: self.cards[name].setChecked(True)


class RuntimesPage(QWidget):
    """The engines that run models: the built-in Trance engine and the llama.cpp runtimes."""

    def __init__(self, runtimes, engine_binary, parent=None, *, engine_bundled=False):
        super().__init__(parent)
        layout = _scrolling_page(self, "Runtimes")
        intro = QLabel("The engines that run your models."); intro.setObjectName("muted"); intro.setWordWrap(True); layout.addWidget(intro)
        self.engine_card = EngineCard(engine_binary, engine_bundled); layout.addWidget(self.engine_card)
        self.runtime_card = RuntimeCard(runtimes); layout.addWidget(self.runtime_card)
        layout.addStretch(1)
