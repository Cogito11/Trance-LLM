"""A modal drawn inside the main window: a dimmed backdrop over everything with one card in the middle.

Subclasses fill `card`. The window sizes the overlay to itself; the overlay centers and sizes the card.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame


class ModalOverlay(QFrame):
    closed = Signal()
    CARD_WIDTH, CARD_HEIGHT, MARGIN = 540, 560, 28

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("modalOverlay")
        self.card = QFrame(self); self.card.setObjectName("modalCard")
        self.hide()

    def can_dismiss(self) -> bool:
        """Subclasses return False while something must finish first (Escape and the backdrop then do nothing)."""
        return True

    def open(self):
        self._fit()
        self.show(); self.raise_()

    def dismiss(self):
        if not self.can_dismiss(): return
        self._close()

    def _close(self):
        was_open = self.isVisible()
        self.hide()
        if was_open: self.closed.emit()

    def _card_height(self, width: int) -> int:
        return max(300, min(self.CARD_HEIGHT, self.height() - 2 * self.MARGIN))

    def _fit(self):
        width = max(300, min(self.CARD_WIDTH, self.width() - 2 * self.MARGIN))
        height = self._card_height(width)
        self.card.setGeometry((self.width() - width) // 2, (self.height() - height) // 2, width, height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape: self.dismiss()      # children pass unhandled keys up to here
        else: super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if not self.card.geometry().contains(event.position().toPoint()): self.dismiss()      # a click on the backdrop closes it
        event.accept()

    def wheelEvent(self, event):
        event.accept()                                          # nothing behind the backdrop scrolls while it is open
