"""The model picker: a modal card over the window with a searchable list of models.

It is drawn inside the main window (a dimmed backdrop and a centered card) rather than as a separate operating
system dialog. The window owns the model list and does the loading; this widget only shows the choices and reports
what was chosen.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QStyle, QStyledItemDelegate, QStyleOptionViewItem, QVBoxLayout)

from .modal import ModalOverlay
from .theme import COLORS as C

PATH_ROLE = Qt.ItemDataRole.UserRole
DETAIL_ROLE = Qt.ItemDataRole.UserRole + 1
LOADED_ROLE = Qt.ItemDataRole.UserRole + 2
NEEDS_ROLE = Qt.ItemDataRole.UserRole + 3


class ModelRowDelegate(QStyledItemDelegate):
    """A two-line row: the model name, a quieter line of details beneath it, and a badge on the loaded model."""
    HEIGHT = 58

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), self.HEIGHT)

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""                                           # the style draws the hover/selected background; the text is ours
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        rect = opt.rect.adjusted(14, 0, -14, 0)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        right = rect.right()
        if index.data(LOADED_ROLE):
            font = QFont(opt.font); font.setPixelSize(11); font.setWeight(QFont.Weight.DemiBold)
            width = QFontMetrics(font).horizontalAdvance("Loaded") + 20
            badge = QRect(rect.right() - width + 1, rect.center().y() - 10, width, 20)
            wash = QColor(C["accent"]); wash.setAlphaF(0.16)
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(wash); painter.drawRoundedRect(badge, 10, 10)
            painter.setFont(font); painter.setPen(QColor(C["accent"]))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, "Loaded")
            right = badge.left() - 12
        elif index.data(NEEDS_ROLE):                            # a GGUF model while no llama.cpp runtime is installed
            font = QFont(opt.font); font.setPixelSize(11); font.setWeight(QFont.Weight.DemiBold)
            width = QFontMetrics(font).horizontalAdvance("Needs runtime") + 20
            chip = QRect(rect.right() - width + 1, rect.center().y() - 10, width, 20)
            painter.setPen(QPen(QColor(C["warning"]), 1)); painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(chip.adjusted(0, 0, -1, -1), 10, 10)
            painter.setFont(font); painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, "Needs runtime")
            right = chip.left() - 12
        room = max(40, right - rect.left())
        title_font = QFont(opt.font); title_font.setPixelSize(13); title_font.setWeight(QFont.Weight.DemiBold)
        detail_font = QFont(opt.font); detail_font.setPixelSize(11)
        painter.setFont(title_font); painter.setPen(QColor(C["text"]))
        title = QFontMetrics(title_font).elidedText(str(index.data(Qt.ItemDataRole.DisplayRole) or ""), Qt.TextElideMode.ElideRight, room)
        painter.drawText(QRect(rect.left(), rect.top() + 9, room, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, title)
        painter.setFont(detail_font); painter.setPen(QColor(C["text_muted"]))
        detail = QFontMetrics(detail_font).elidedText(str(index.data(DETAIL_ROLE) or ""), Qt.TextElideMode.ElideRight, room)
        painter.drawText(QRect(rect.left(), rect.top() + 29, room, 18), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, detail)
        painter.restore()


class _SearchField(QLineEdit):
    """A search box whose arrow keys move through the list below it, so a model can be chosen without the mouse."""
    step = Signal(int)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            self.step.emit(1 if event.key() == Qt.Key.Key_Down else -1)
            event.accept()
        else:
            super().keyPressEvent(event)


class ModelPicker(ModalOverlay):
    picked = Signal(str)             # the path of the chosen model
    eject_requested = Signal()
    browse_requested = Signal()
    refresh_requested = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self._entries: list[dict] = []
        self._loaded = ""
        box = QVBoxLayout(self.card); box.setContentsMargins(22, 18, 22, 18); box.setSpacing(12)
        head = QHBoxLayout(); head.setSpacing(8)
        self.title = QLabel("Select a model"); self.title.setObjectName("modalTitle"); head.addWidget(self.title); head.addStretch(1)
        self.close_button = QPushButton("\u2715"); self.close_button.setObjectName("iconButton")
        self.close_button.setFixedSize(30, 30); self.close_button.setToolTip("Close"); self.close_button.setAccessibleName("Close")
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor); self.close_button.clicked.connect(self.dismiss)
        head.addWidget(self.close_button); box.addLayout(head)
        self.search = _SearchField(); self.search.setPlaceholderText("Search models"); self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Search models")
        self.search.textChanged.connect(lambda _text: self._populate())
        self.search.returnPressed.connect(self._pick_current); self.search.step.connect(self._step)
        box.addWidget(self.search)
        self.list = QListWidget(); self.list.setObjectName("modelList"); self.list.setItemDelegate(ModelRowDelegate(self.list))
        self.list.setMouseTracking(True); self.list.setSpacing(2); self.list.setUniformItemSizes(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.itemClicked.connect(self._pick); self.list.itemActivated.connect(self._pick)
        self.empty = QLabel(); self.empty.setObjectName("muted"); self.empty.setWordWrap(True); self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(self.list, 1); box.addWidget(self.empty, 1)
        rule = QFrame(); rule.setObjectName("sidebarRule"); rule.setFixedHeight(1); box.addWidget(rule)
        foot = QHBoxLayout(); foot.setSpacing(6)
        self.browse_button = QPushButton("Browse\u2026"); self.browse_button.setToolTip("Add a model file from anywhere on this computer")
        self.refresh_button = QPushButton("\u21bb  Refresh list"); self.refresh_button.setToolTip("Look for newly added or newly trained models")
        self.eject_button = QPushButton("Eject"); self.eject_button.setToolTip("Unload the loaded model and free its memory")
        for b in (self.browse_button, self.refresh_button, self.eject_button):
            b.setProperty("compact", True); b.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_button.clicked.connect(self._browse)
        self.refresh_button.clicked.connect(self.refresh_requested)
        self.eject_button.clicked.connect(self.eject_requested)
        foot.addWidget(self.browse_button); foot.addWidget(self.refresh_button); foot.addStretch(1); foot.addWidget(self.eject_button)
        box.addLayout(foot)

    # ---- content
    def set_models(self, entries: list[dict], loaded: str = "", *, can_eject: bool = False):
        """`entries` are dicts with `path`, `title`, `details` and optionally `needs_runtime`; the loaded model is listed first."""
        self._entries, self._loaded = list(entries), loaded
        self.eject_button.setEnabled(bool(loaded) and can_eject)
        self._populate()

    def _populate(self):
        terms = self.search.text().casefold().split()
        self.list.clear()
        for entry in sorted(self._entries, key=lambda e: e["path"] != self._loaded):          # stable: loaded first
            haystack = f"{entry['title']} {entry['details']} {Path(entry['path']).name}".casefold()
            if not all(term in haystack for term in terms): continue
            item = QListWidgetItem(entry["title"])
            item.setData(PATH_ROLE, entry["path"]); item.setData(DETAIL_ROLE, entry["details"])
            item.setData(LOADED_ROLE, entry["path"] == self._loaded); item.setToolTip(entry["path"])
            item.setData(NEEDS_ROLE, bool(entry.get("needs_runtime")))
            self.list.addItem(item)
        if self.list.count(): self.list.setCurrentRow(0)
        shown = self.list.count() > 0
        self.list.setVisible(shown); self.empty.setVisible(not shown)
        self.empty.setText("No models match your search." if self._entries else
                           "No models found yet. Use Browse\u2026 to add a model file, or Refresh list after training one.")

    def visible_paths(self) -> list[str]:
        return [self.list.item(i).data(PATH_ROLE) for i in range(self.list.count())]

    # ---- choosing
    def _step(self, delta: int):
        if self.list.count():
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, self.list.currentRow() + delta)))

    def _pick_current(self):
        item = self.list.currentItem()
        if item is not None: self._pick(item)

    def _pick(self, item):
        path = item.data(PATH_ROLE)
        self.dismiss()
        self.picked.emit(path)

    def _browse(self):
        self.dismiss()
        self.browse_requested.emit()

    # ---- showing and hiding
    def open(self):
        self.search.clear()
        self._populate()
        super().open()
        self.search.setFocus()
