"""Settings page: currently the color theme."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
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


class RuntimeCard(QFrame):
    """Shows which llama.cpp runtime is in use and lets the user download or choose one."""

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setObjectName("settingsCard")
        box = QVBoxLayout(self); box.setContentsMargins(18, 16, 18, 16); box.setSpacing(10)
        title = QLabel("llama.cpp runtime"); title.setObjectName("themeName"); box.addWidget(title)
        blurb = QLabel("Runs GGUF models with the same engine LM Studio uses. Trance models run on the built-in CUDA engine and do not need this.")
        blurb.setObjectName("muted"); blurb.setWordWrap(True); box.addWidget(blurb)
        self.status = QLabel(); self.status.setObjectName("runtimeStatus"); self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); box.addWidget(self.status)
        row = QHBoxLayout(); row.setSpacing(8)
        self.flavor = QComboBox(); row.addWidget(self.flavor, 1)
        self.install_button = QPushButton("Download"); self.install_button.setObjectName("primary"); row.addWidget(self.install_button)
        box.addLayout(row)
        self.flavor_detail = QLabel(); self.flavor_detail.setObjectName("muted"); self.flavor_detail.setWordWrap(True); box.addWidget(self.flavor_detail)
        self.progress = QProgressBar(); self.progress.setVisible(False); box.addWidget(self.progress)
        self.error = QLabel(); self.error.setObjectName("configHint"); self.error.setProperty("bad", True); self.error.setWordWrap(True)
        self.error.setVisible(False); box.addWidget(self.error)
        row2 = QHBoxLayout(); row2.setSpacing(8)
        self.choose_button = QPushButton("Choose existing\u2026"); self.remove_button = QPushButton("Remove download")
        row2.addWidget(self.choose_button); row2.addWidget(self.remove_button); row2.addStretch(1); box.addLayout(row2)
        note = QLabel("Downloading contacts github.com. Nothing else in this app uses the network. Files are checked against "
                      "SHA-256 hashes built into the app before they are unpacked.")
        note.setObjectName("muted"); note.setWordWrap(True); box.addWidget(note)
        self.flavor.currentIndexChanged.connect(self._flavor_changed)
        self.install_button.clicked.connect(self._install_clicked)
        self.choose_button.clicked.connect(self._choose)
        self.remove_button.clicked.connect(self._remove)
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
        self.status.setText(self.manager.status())
        self.install_button.setText("Cancel" if installing else "Download")
        self.install_button.setEnabled(bool(self.manager.flavors))
        self.flavor.setEnabled(not installing and bool(self.manager.flavors))
        self.choose_button.setEnabled(not installing)
        custom = bool(self.manager.custom_path())
        self.remove_button.setVisible(custom or self.manager.has_download())
        self.remove_button.setText("Stop using this path" if custom else "Remove download")
        self.remove_button.setEnabled(not installing)
        self.progress.setVisible(installing)
        if installing: self.error.setVisible(False)
        self._flavor_changed()

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

    def _remove(self):
        if self.manager.custom_path(): self.manager.clear_custom()
        else: self.manager.remove_managed()


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
