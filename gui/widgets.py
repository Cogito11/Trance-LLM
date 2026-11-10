"""Desktop presentation widgets; charts use QPainter, with no web renderer."""
from __future__ import annotations

import html
import math

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QPlainTextEdit, QTextBrowser, QWidget, QListWidget


class ConversationList(QListWidget):
    def mousePressEvent(self, event):
        # A context-menu action should not switch away from an unsent draft
        # or clear a temporary chat before the user chooses an action.
        if event.button() == Qt.MouseButton.RightButton:
            event.accept()
        else:
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

    def render(self, messages: list[dict], empty_title="", empty_text=""):
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum()-24
        scroll = bar.value()
        if not messages:
            body = f'<h2 style="color:#e5edf7">{html.escape(empty_title)}</h2><p style="color:#8294ab">{html.escape(empty_text)}</p>' if empty_title or empty_text else ""
        else:
            blocks = []
            for m in messages:
                role = "You" if m["role"] == "user" else "Trance"
                color = "#91a5c1" if m["role"] == "user" else "#63dacb"
                text = html.escape(m.get("content", "")).replace("\n", "<br>")
                if not text:
                    text = '<span style="color:#8294ab">Thinking…</span>' if m.get("status") == "generating" else '<span style="color:#8294ab">No response text.</span>'
                note = html.escape(m.get("note", ""))
                blocks.append(f'<p style="color:{color};font-size:12px"><b>{role.upper()}</b></p>'
                              f'<p style="color:#e5edf7;line-height:150%">{text}</p>'
                              f'<p style="color:#8294ab;font-size:11px">{note}</p><br>')
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
        p.fillRect(self.rect(), QColor("#111b29"))
        area = QRectF(54, 25, max(1, self.width()-80), max(1, self.height()-64))
        all_points = self.train + self.valid
        p.setPen(QColor("#8294ab"))
        if not all_points:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Loss history will appear when training starts")
            return
        lo = min(v for _, v in all_points); hi = max(v for _, v in all_points)
        pad = max(.1, (hi-lo)*.12); lo = max(0, lo-pad); hi += pad
        max_step = max(1, max(s for s, _ in all_points))
        for i in range(5):
            y = area.top() + area.height()*i/4
            p.setPen(QColor("#253348")); p.drawLine(int(area.left()), int(y), int(area.right()), int(y))
            p.setPen(QColor("#8294ab")); p.drawText(QRectF(0,y-9,46,18), Qt.AlignmentFlag.AlignRight, f"{hi-(hi-lo)*i/4:.2f}")
        p.drawText(QRectF(area.left(),area.bottom()+10,area.width(),22), Qt.AlignmentFlag.AlignRight, f"Step {max_step:,}")
        for series, color in ((self.train,"#7fa7ff"),(self.valid,"#63dacb")):
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


STYLE = """
QWidget { background: #0b121c; color: #e5edf7; font-family: 'DejaVu Sans'; font-size: 13px; }
QMainWindow { background: #0b121c; }
QFrame#sidebar { background: #101a27; border-right: 1px solid #253348; }
QFrame#settings { background: #101a27; border-left: 1px solid #253348; }
QLabel { background: transparent; }
QLabel#brand { color: #e5edf7; font-size: 24px; font-weight: 700; letter-spacing: 3px; }
QLabel#brandAccent { color: #63dacb; font-size: 20px; font-weight: 700; letter-spacing: 1px; }
QLabel#title { font-size: 25px; font-weight: 600; }
QLabel#muted { color: #8294ab; font-size: 12px; }
QLabel#eyebrow { color: #63dacb; font-size: 11px; font-weight: 600; }
QLabel#parameter { font-size: 12px; font-weight: 600; }
QLabel#parameterDescription { color: #8294ab; font-size: 12px; }
QLabel#temporaryNotice { color: #8294ab; font-size: 14px; }
QLabel#composerHint { color: #8294ab; font-size: 11px; }
QScrollArea#generationScroll { background: transparent; border: none; }
QScrollArea#formScroll { background: transparent; border: none; }
QWidget#generationContent { background: #101a27; }
QFrame#composerPanel { background: #101a27; border: 1px solid #2a3b52; border-radius: 12px; }
QFrame#modelControls { background: #101a27; border: 1px solid #2a3b52; border-radius: 10px; }
QWidget#responseControls { background: transparent; }
QPlainTextEdit#messageInput { background: transparent; border: none; padding: 4px; }
QPushButton { background: #192638; border: 1px solid #2a3b52; padding: 10px 16px; border-radius: 7px; }
QPushButton:hover { background: #24364c; border-color: #496486; }
QPushButton:pressed { background: #2a4059; }
QPushButton:disabled { color: #52647c; background: #142030; border-color: #253348; }
QPushButton#primary { background: #63dacb; color: #09251f; border: none; font-weight: 700; }
QPushButton#primary:hover { background: #8be8dd; }
QPushButton#primary:disabled { background: #264b4b; color: #768f94; }
QPushButton#nav { text-align: left; background: transparent; border: none; padding: 12px; }
QPushButton#nav:checked { background: #223247; color: #8fe4d9; }
QPushButton[compact="true"] { padding: 8px 10px; font-size: 12px; }
QComboBox, QSpinBox, QDoubleSpinBox { background: #142030; border: 1px solid #2a3b52; border-radius: 5px; padding: 8px; }
QLineEdit { background: #142030; border: 1px solid #2a3b52; border-radius: 5px; padding: 8px; }
QComboBox::drop-down { width: 24px; border: none; }
QComboBox QAbstractItemView { background: #142030; selection-background-color: #2a4059; }
QPlainTextEdit, QTextBrowser { background: #101a27; border: 1px solid #2a3b52; border-radius: 7px; padding: 10px; selection-background-color: #315985; }
QTextBrowser#transcript { background: #0b121c; border: none; padding: 0; }
QTextBrowser#articleReader { background: #101a27; border: 1px solid #2a3b52; padding: 18px; font-size: 14px; }
QTabWidget::pane { border: 1px solid #2a3b52; border-radius: 5px; }
QTabBar::tab { background: #142030; color: #9db0c9; padding: 12px 16px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { background: #192638; color: #8fe4d9; border-bottom: 2px solid #63dacb; }
QTreeWidget { background: #101a27; border: 1px solid #2a3b52; border-radius: 5px; outline: none; }
QTreeWidget::item { padding: 10px 4px; }
QTreeWidget::item:selected { background: #223247; color: #8fe4d9; }
QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { padding: 11px 8px; border-radius: 5px; color: #9db0c9; }
QListWidget::item:selected { background: #223247; color: #e5edf7; }
QProgressBar { background: #142030; border: none; border-radius: 5px; min-height: 12px; text-align: center; }
QProgressBar::chunk { background: #63dacb; border-radius: 5px; }
QStatusBar { background: #101a27; color: #8294ab; }
QSplitter::handle { background: #253348; width: 1px; }
QScrollBar:vertical { background: transparent; width: 9px; }
QScrollBar::handle:vertical { background: #2a3b52; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""
