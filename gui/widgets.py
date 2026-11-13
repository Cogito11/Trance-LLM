"""Desktop presentation widgets; charts use QPainter, with no web renderer."""
from __future__ import annotations

import html
import math

from PySide6.QtCore import Qt, Signal, QRect, QRectF
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QListWidget, QPlainTextEdit, QStyle, QStyledItemDelegate, QTextBrowser, QWidget

from .theme import STYLE, COLORS as C  # STYLE re-exported for existing imports


class ChatItemDelegate(QStyledItemDelegate):
    """Draws a ⋯ affordance on the hovered or active chat so rename/delete is discoverable."""
    DOTS = 28

    def _marked(self, option):
        return bool(option.state & (QStyle.StateFlag.State_MouseOver | QStyle.StateFlag.State_Selected)) \
            and bool(option.state & QStyle.StateFlag.State_Enabled)

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if self._marked(option):  # leave room so long titles never run under the dots
            option.text = option.fontMetrics.elidedText(option.text, Qt.TextElideMode.ElideRight, option.rect.width() - 16 - self.DOTS)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if self._marked(option):
            painter.save()
            painter.setPen(QColor(C["text_muted"]))
            painter.drawText(QRect(option.rect.right() - self.DOTS, option.rect.top(), self.DOTS, option.rect.height()),
                             Qt.AlignmentFlag.AlignCenter, "\u22ef")
            painter.restore()


class ConversationList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setItemDelegate(ChatItemDelegate(self))

    def mousePressEvent(self, event):
        # A context-menu action should not switch away from an unsent draft
        # or clear a temporary chat before the user chooses an action.
        if event.button() == Qt.MouseButton.RightButton:
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            position = event.position().toPoint()
            item = self.itemAt(position)
            if item is not None and position.x() >= self.visualItemRect(item).right() - ChatItemDelegate.DOTS:
                # The ⋯ button opens the same menu as right-click.
                self.customContextMenuRequested.emit(position)
                event.accept()
                return
        super().mousePressEvent(event)


class Composer(QPlainTextEdit):
    submitted = Signal()

    def __init__(self, parent=None, *, auto_size=False):
        super().__init__(parent)
        self._auto_size = auto_size
        self._sizing = False
        if auto_size:
            self.textChanged.connect(self._fit_content)
            self.document().documentLayout().documentSizeChanged.connect(self._fit_content)
            self._fit_content()

    def _fit_content(self):
        if not self._auto_size or self._sizing: return
        self._sizing = True
        try:
            lines = 0
            block = self.document().begin()
            while block.isValid() and lines < 7:
                # Force layout so wrapped lines count even after a large paste.
                self.document().documentLayout().blockBoundingRect(block)
                lines += max(1,block.layout().lineCount())
                block = block.next()
            height = math.ceil(lines*self.fontMetrics().lineSpacing()+2*self.document().documentMargin()+12)
            self.setFixedHeight(max(40,min(136,height)))
        finally:
            self._sizing = False

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if getattr(self,"_auto_size",False): self._fit_content()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class Transcript(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.setObjectName("transcript")
        self.setReadOnly(True)

    def restyle(self):
        """Redraw with the current theme colors (the HTML bakes them in)."""
        messages, title, text = getattr(self, "_last", ([], "", ""))
        self.render(messages, title, text)

    def render(self, messages: list[dict], empty_title="", empty_text=""):
        self._last = (messages, empty_title, empty_text)
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum()-24
        scroll = bar.value()
        if not messages:
            body = f'<h2 style="color:{C["text"]}">{html.escape(empty_title)}</h2><p style="color:{C["text_muted"]}">{html.escape(empty_text)}</p>' if empty_title or empty_text else ""
        else:
            blocks = []
            for m in messages:
                role = "You" if m["role"] == "user" else "Trance"
                color = C["text_subtle"] if m["role"] == "user" else C["accent"]
                text = html.escape(m.get("content", "")).replace("\n", "<br>")
                if not text:
                    text = f'<span style="color:{C["text_muted"]}">Thinking…</span>' if m.get("status") == "generating" else f'<span style="color:{C["text_muted"]}">No response text.</span>'
                note = html.escape(m.get("note", ""))
                blocks.append(f'<p style="color:{color};font-size:12px"><b>{role.upper()}</b></p>'
                              f'<p style="color:{C["text"]};line-height:150%">{text}</p>'
                              f'<p style="color:{C["text_muted"]};font-size:11px">{note}</p><br>')
            body = "".join(blocks)
        self.setHtml('<html><body style="font-family:DejaVu Sans;font-size:14px;margin:24px">' + body + '</body></html>')
        bar.setValue(bar.maximum() if at_bottom else scroll)


class LossChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(225)
        self.train = []
        self.valid = []

    def clear(self):
        self.train.clear(); self.valid.clear(); self.update()

    def add(self, event):
        for field, series in (("train_loss", self.train), ("valid_loss", self.valid)):
            value = event.get(field)
            if isinstance(value, (int, float)) and math.isfinite(value):
                series.append((event["step"], value))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(C["surface"]))
        area = QRectF(54, 25, max(1, self.width()-80), max(1, self.height()-64))
        all_points = self.train + self.valid
        p.setPen(QColor(C["text_muted"]))
        if not all_points:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Loss history will appear when training starts")
            return
        lo = min(v for _, v in all_points); hi = max(v for _, v in all_points)
        pad = max(.1, (hi-lo)*.12); lo = max(0, lo-pad); hi += pad
        max_step = max(1, max(s for s, _ in all_points))
        for i in range(5):
            y = area.top() + area.height()*i/4
            p.setPen(QColor(C["border"])); p.drawLine(int(area.left()), int(y), int(area.right()), int(y))
            p.setPen(QColor(C["text_muted"])); p.drawText(QRectF(0,y-9,46,18), Qt.AlignmentFlag.AlignRight, f"{hi-(hi-lo)*i/4:.2f}")
        p.drawText(QRectF(area.left(),area.bottom()+10,area.width(),22), Qt.AlignmentFlag.AlignRight, f"Step {max_step:,}")
        for series, color in ((self.train,C["accent_2"]),(self.valid,C["accent"])):
            if not series:
                continue
            path = QPainterPath()
            sampled = series[::max(1,len(series)//1500)]
            if sampled[-1] != series[-1]:
                sampled.append(series[-1])
            for i, (step, loss) in enumerate(sampled):
                x = area.left()+step/max_step*area.width(); y = area.bottom()-(loss-lo)/(hi-lo)*area.height()
                if i == 0: path.moveTo(x,y)
                else: path.lineTo(x,y)
            p.setPen(QPen(QColor(color),2)); p.drawPath(path)
            if len(sampled)==1:
                p.drawEllipse(path.currentPosition(),3,3)


def dot_icon(color, size=10):
    """A small filled circle, used as a status marker in lists."""
    pixmap = QPixmap(size, size); pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor(color)); painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return QIcon(pixmap)
