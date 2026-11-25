"""Desktop presentation widgets; charts use QPainter, with no web renderer."""
from __future__ import annotations

import html
import math

from PySide6.QtCore import Qt, Signal, QPointF, QRect, QRectF, QSize
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QListWidget, QPlainTextEdit, QPushButton, QSizePolicy, QStyle, QStyledItemDelegate,
                               QTextBrowser, QWidget)

from .markdown import to_html as to_markdown_html
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
            hovered = getattr(option.widget, "dots_row", -1) == index.row()
            area = QRect(option.rect.right() - self.DOTS, option.rect.top(), self.DOTS, option.rect.height())
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if hovered:                                   # the pointer is over the dots themselves: a small rounded button
                button = QRect(area.center().x() - 11, area.center().y() - 11, 22, 22)
                painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor(C["pressed"])); painter.drawRoundedRect(button, 6, 6)
            painter.setPen(QColor(C["text"] if hovered else C["text_muted"]))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "\u22ef")
            painter.restore()


class ConversationList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setItemDelegate(ChatItemDelegate(self))
        self.dots_row = -1                      # the row whose ⋯ button the pointer is over, or -1

    def _set_dots_row(self, row):
        if row != self.dots_row:
            self.dots_row = row
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor if row >= 0 else Qt.CursorShape.ArrowCursor)
            self.viewport().update()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        position = event.position().toPoint(); item = self.itemAt(position)
        over = item is not None and position.x() >= self.visualItemRect(item).right() - ChatItemDelegate.DOTS
        self._set_dots_row(self.row(item) if over else -1)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._set_dots_row(-1)

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


class DropdownButton(QPushButton):
    """A button that looks like a drop-down: the label on the left (elided when long) and a chevron on the right.

    It opens whatever the owner connects to `clicked`. The label is painted rather than set as the button text so
    that a long model name is cut with an ellipsis instead of widening the header.
    """
    CHEVRON = 30                                   # room kept on the right for the chevron

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropdown")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._label, self._placeholder = "", False

    def set_label(self, text: str, *, placeholder: bool = False):
        self._label, self._placeholder = text, placeholder
        self.setAccessibleName(text)
        self.updateGeometry(); self.update()

    def label(self) -> str:
        return self._label

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(self.fontMetrics().horizontalAdvance(self._label) + 18 + self.CHEVRON, max(hint.height(), 38))

    def minimumSizeHint(self) -> QSize:
        return QSize(140, max(super().minimumSizeHint().height(), 38))

    def paintEvent(self, event):
        super().paintEvent(event)                  # no button text is set: the style sheet draws just the box
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.isEnabled(): color = C["text_disabled"]
        elif self._placeholder: color = C["text_subtle"]
        else: color = C["text"]
        painter.setPen(QColor(color))
        area = self.rect().adjusted(14, 0, -self.CHEVRON, 0)
        painter.drawText(area, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         self.fontMetrics().elidedText(self._label, Qt.TextElideMode.ElideRight, area.width()))
        cx, cy = self.width() - 17, self.height() / 2                       # chevron
        painter.setPen(QPen(QColor(C["text_disabled"] if not self.isEnabled() else C["text_muted"]), 1.6,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPolyline([QPointF(cx - 4, cy - 2), QPointF(cx, cy + 2), QPointF(cx + 4, cy - 2)])


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
    """The conversation view: plain text for you, rendered markdown for the model, and a collapsible thinking block."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)           # links never navigate this view; see _link_clicked
        self.setOpenExternalLinks(False)
        self.setObjectName("transcript")
        self.setReadOnly(True)
        self.anchorClicked.connect(self._link_clicked)
        # Scroll-follow: stay glued to the newest text while the reader is at the end, and leave the view alone once
        # they scroll up. Qt lays a redrawn document out lazily, so the scroll range changes *after* setHtml returns;
        # position is therefore decided by remembered intent (_stick) and re-applied whenever the range changes.
        self._stick, self._programmatic, self._restore = True, False, None
        bar = self.verticalScrollBar()
        bar.valueChanged.connect(self._value_changed)
        bar.rangeChanged.connect(self._range_changed)

    def follow_end(self):
        """Resume following the newest text (a new chat was opened, or a message was just sent)."""
        self._stick, self._restore = True, None

    def _value_changed(self, value):
        if self._programmatic: return
        self._restore = None                                   # the reader moved: forget any position we were holding for them
        self._stick = value >= self.verticalScrollBar().maximum() - 6

    def _range_changed(self, _low, high):
        if self._programmatic: return
        bar = self.verticalScrollBar()
        self._programmatic = True
        try:
            if self._stick: bar.setValue(high)
            elif self._restore is not None:
                bar.setValue(min(self._restore, high))
                if high >= self._restore: self._restore = None
        finally: self._programmatic = False

    def restyle(self):
        """Redraw with the current theme colors (the HTML bakes them in)."""
        messages, title, text = getattr(self, "_last", ([], "", ""))
        self.render(messages, title, text)

    def _link_clicked(self, url):
        if url.scheme() == "think":
            messages = getattr(self, "_last", ([],))[0]
            try: message = messages[int(url.path())]
            except (ValueError, IndexError): return
            message["thinking_open"] = not message.get("thinking_open", False)
            self.restyle()
        elif url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)        # only when the reader clicks; nothing is ever fetched automatically

    @staticmethod
    def _thinking_html(index: int, message: dict) -> str:
        thinking = message.get("thinking", "")
        if not thinking: return ""
        waiting = message.get("status") == "generating" and not message.get("content")
        opened = waiting or message.get("thinking_open", False)
        seconds = message.get("thinking_seconds")
        title = "Thinking\u2026" if waiting else (f"Thought for {seconds:.0f} s" if seconds and seconds >= 1 else "Thoughts")
        muted = C["text_muted"]
        arrow = "\u25be" if opened else "\u25b8"
        head = (f'<p style="font-size:12px;margin-top:0px;margin-bottom:2px"><a href="think:{index}" style="color:{muted};text-decoration:none">'
                f'{arrow} {title}</a></p>')
        if not opened: return head
        return head + to_markdown_html(thinking, "thinking") + '<p style="margin-top:0px;margin-bottom:0px;font-size:8px">&nbsp;</p>'

    def render(self, messages: list[dict], empty_title="", empty_text=""):
        self._last = (messages, empty_title, empty_text)
        bar = self.verticalScrollBar()
        target = self._restore if self._restore is not None else bar.value()      # where the reader was, before this redraw
        if not messages:
            body = f'<h2 style="color:{C["text"]}">{html.escape(empty_title)}</h2><p style="color:{C["text_muted"]}">{html.escape(empty_text)}</p>' if empty_title or empty_text else ""
        else:
            blocks, muted = [], C["text_muted"]
            for index, m in enumerate(messages):
                user = m["role"] == "user"
                role = "You" if user else (m.get("model_name") or "Trance")
                color = C["text_subtle"] if user else C["accent"]
                content = m.get("content", "")
                if user: text = f'<p style="color:{C["text"]};line-height:150%;margin-top:0px;margin-bottom:0px">{html.escape(content).replace(chr(10), "<br>")}</p>'
                elif content.strip(): text = to_markdown_html(content)
                elif m.get("status") == "generating": text = "" if m.get("thinking") else f'<p style="color:{muted};margin-top:0px;margin-bottom:0px">Generating\u2026</p>'
                else: text = f'<p style="color:{muted};margin-top:0px;margin-bottom:0px">No response text.</p>'
                note = html.escape(m.get("note", ""))
                note_html = f'<p style="color:{muted};font-size:11px;margin-top:2px;margin-bottom:0px">{note}</p>' if note else ""
                blocks.append(f'<p style="color:{color};font-size:12px;margin-top:0px;margin-bottom:4px"><b>{html.escape(role.upper())}</b></p>'
                              + ("" if user else self._thinking_html(index, m)) + text + note_html
                              + '<p style="margin-top:0px;margin-bottom:0px;font-size:18px">&nbsp;</p>')     # the gap between messages
            body = "".join(blocks)
        self._programmatic = True
        try:
            self.setHtml('<html><body style="font-family:DejaVu Sans;font-size:14px;margin:24px">' + body + '</body></html>')
            if self._stick:
                bar.setValue(bar.maximum()); self._restore = None
            else:
                bar.setValue(min(target, bar.maximum()))
                self._restore = target if target > bar.maximum() else None            # layout is not finished yet: finish restoring later
        finally: self._programmatic = False


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


def round_popup(popup):
    """Give a popup (a menu or a combo list) real rounded corners.

    The style sheet can round the *drawing*, but the popup window itself is rectangular; making it frameless and
    transparent lets the corners show what is behind them. (Without a compositing window manager the corners
    can look square or dark; nothing breaks.)
    """
    popup.setWindowFlags(popup.windowFlags() | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint)
    popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def style_combo_popup(combo: QComboBox):
    """Polished drop-down list: rounded, padded rows with hover and selection effects (see theme.py)."""
    popup = combo.view().window()
    popup.setObjectName("comboPopup")
    round_popup(popup)
