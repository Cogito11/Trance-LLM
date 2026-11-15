"""Settings page: currently the color theme."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

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


class SettingsPage(QWidget):
    theme_selected = Signal(str)

    def __init__(self, current: str = "teal", parent=None, *, runtimes=None):
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
        self.runtime_card = None
        if runtimes is not None:
            heading = QLabel("RUNTIMES"); heading.setObjectName("eyebrow"); layout.addSpacing(8); layout.addWidget(heading)
            self.runtime_card = RuntimeCard(runtimes); layout.addWidget(self.runtime_card)
        layout.addStretch(1)
        self.select(current)

    def select(self, name: str):
        """Mark a theme as current without emitting theme_selected."""
        if name in self.cards: self.cards[name].setChecked(True)
