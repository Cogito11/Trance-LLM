"""Single source of truth for the desktop app's colors and stylesheet.

A near-black blue base with one primary accent and one secondary accent.
Change a value in COLORS and the whole app follows; widgets read colors
from COLORS at paint time, never from literals.
"""
from __future__ import annotations

from pathlib import Path

COLORS = {
    # Surfaces, darkest to lightest
    "base": "#070b14",
    "surface": "#0e1624",
    "raised": "#141e30",
    "control": "#101a2a",
    "hover": "#1b2840",
    "pressed": "#233453",
    # Lines
    "border": "#1f2b40",
    "border_strong": "#2c3d5a",
    "border_hover": "#4a6490",
    # Text
    "text": "#e6edf7",
    "text_muted": "#8294ab",
    "text_subtle": "#9db0c9",
    "text_disabled": "#4f6079",
    # Accent: one primary color, one secondary used for data and the wordmark
    "accent": "#4fd1c5",
    "accent_hi": "#7ae0d6",
    "accent_2": "#7da2ff",
    "accent_wash": "rgba(79, 209, 197, 0.10)",
    "selected_bg": "#0f2b2c",
    "on_accent": "#04201d",
    "accent_disabled": "#1b4543",
    "accent_disabled_text": "#6c9390",
    # Status
    "danger": "#f87171",
    # Text selection
    "selection": "#2b4a7a",
}

C = COLORS

# Palettes only differ in the accent tokens; surfaces and text stay shared.
PALETTES = {
    "teal": {
        "accent": "#4fd1c5", "accent_hi": "#7ae0d6", "accent_2": "#7da2ff",
        "accent_wash": "rgba(79, 209, 197, 0.10)", "selected_bg": "#0f2b2c", "on_accent": "#04201d",
        "accent_disabled": "#1b4543", "accent_disabled_text": "#6c9390",
    },
    "indigo": {
        "accent": "#8b9bff", "accent_hi": "#aab5ff", "accent_2": "#4fd1c5",
        "accent_wash": "rgba(139, 155, 255, 0.12)", "selected_bg": "#1a2146", "on_accent": "#0b1030",
        "accent_disabled": "#2a3166", "accent_disabled_text": "#8089b8",
    },
    "amber": {
        "accent": "#f2b45a", "accent_hi": "#f7c97f", "accent_2": "#6aa9ff",
        "accent_wash": "rgba(242, 180, 90, 0.10)", "selected_bg": "#2b2110", "on_accent": "#241705",
        "accent_disabled": "#4a3a1d", "accent_disabled_text": "#9a8760",
    },
}
PALETTE_INFO = {
    "teal": ("Teal", "Calm and cool. The default."),
    "indigo": ("Indigo", "Cooler and a little more formal."),
    "amber": ("Amber", "Warm and distinctive."),
}
DEFAULT_PALETTE = "teal"
ASSETS = Path(__file__).resolve().parent / "assets"


def _url(name: str) -> str:
    # Forward slashes and quotes keep paths with spaces working in Qt styles.
    return '"' + (ASSETS / name).as_posix() + '"'


def build_stylesheet() -> str:
    c = dict(C)
    c["arrow_down"] = _url("arrow_down.svg")
    c["arrow_up"] = _url("arrow_up.svg")
    c["grad_progress"] = ("qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                          f"stop:0 {C['accent']},stop:1 {C['accent_2']})")
    return _TEMPLATE.format(**c)


_TEMPLATE = """
QWidget {{ background: {base}; color: {text}; font-family: 'DejaVu Sans'; font-size: 13px; }}
QMainWindow {{ background: {base}; }}
QFrame#sidebar {{ background: {surface}; border-right: 1px solid {border}; }}
QFrame#settings {{ background: {surface}; border-left: 1px solid {border}; }}
QLabel {{ background: transparent; }}
QLabel#brand {{ color: {accent}; font-size: 24px; font-weight: 700; letter-spacing: 3px; }}
QLabel#brandAccent {{ color: {accent_2}; font-size: 20px; font-weight: 700; letter-spacing: 1px; }}
QLabel#title {{ font-size: 25px; font-weight: 600; }}
QLabel#muted {{ color: {text_muted}; font-size: 12px; }}
QLabel#eyebrow {{ color: {accent}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QFrame#sidebar QLabel#eyebrow {{ color: {text_muted}; }}
QLabel#parameter {{ font-size: 12px; font-weight: 600; }}
QLabel#parameterDescription {{ color: {text_muted}; font-size: 12px; }}
QLabel#temporaryNotice {{ color: {text_muted}; font-size: 14px; }}
QLabel#composerHint {{ color: {text_muted}; font-size: 11px; }}
QScrollArea#generationScroll {{ background: transparent; border: none; }}
QScrollArea#formScroll {{ background: transparent; border: none; }}
QWidget#generationContent {{ background: {surface}; }}
QFrame#composerPanel {{ background: {surface}; border: 1px solid {border_strong}; border-radius: 12px; }}
QFrame#modelControls {{ background: {surface}; border: 1px solid {border_strong}; border-radius: 10px; }}
QWidget#responseControls {{ background: transparent; }}
QPlainTextEdit#messageInput {{ background: transparent; border: none; padding: 4px; }}
QPushButton {{ background: {raised}; border: 1px solid {border_strong}; padding: 10px 16px; border-radius: 7px; }}
QPushButton:hover {{ background: {hover}; border-color: {border_hover}; }}
QPushButton:pressed {{ background: {pressed}; }}
QPushButton:disabled {{ color: {text_disabled}; background: {control}; border-color: {border}; }}
QPushButton#primary {{ background: {accent}; color: {on_accent}; border: none; font-weight: 700; }}
QPushButton#primary:hover {{ background: {accent_hi}; }}
QPushButton#primary:disabled {{ background: {accent_disabled}; color: {accent_disabled_text}; }}
QPushButton#nav {{ text-align: left; background: transparent; border: none; border-left: 3px solid transparent; border-radius: 0 7px 7px 0; padding: 12px 12px 12px 10px; color: {text_subtle}; }}
QPushButton#nav:hover {{ background: {hover}; color: {text}; }}
QPushButton#nav:checked {{ background: {accent_wash}; color: {accent}; border-left: 3px solid {accent}; font-weight: 600; }}
QPushButton[compact="true"] {{ padding: 8px 10px; font-size: 12px; }}
QComboBox, QSpinBox, QDoubleSpinBox {{ background: {control}; border: 1px solid {border_strong}; border-radius: 6px; padding: 8px; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {accent}; }}
QLineEdit {{ background: {control}; border: 1px solid {border_strong}; border-radius: 6px; padding: 8px; }}
QComboBox {{ padding-right: 26px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 26px; border: none; }}
QComboBox::down-arrow {{ image: url({arrow_down}); width: 10px; height: 6px; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 22px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 20px; border: none; border-left: 1px solid {border}; border-top-right-radius: 6px; background: transparent; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 20px; border: none; border-left: 1px solid {border}; border-bottom-right-radius: 6px; background: transparent; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {hover}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({arrow_up}); width: 9px; height: 5px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({arrow_down}); width: 9px; height: 5px; }}
QComboBox QAbstractItemView {{ background: {control}; border: 1px solid {border_strong}; selection-background-color: {pressed}; outline: none; }}
QPlainTextEdit, QTextBrowser {{ background: {surface}; border: 1px solid {border_strong}; border-radius: 7px; padding: 10px; selection-background-color: {selection}; }}
QTextBrowser#transcript {{ background: {base}; border: none; padding: 0; }}
QTextBrowser#articleReader {{ background: {surface}; border: 1px solid {border_strong}; padding: 18px; font-size: 14px; }}
QTabWidget::pane {{ border: 1px solid {border_strong}; border-radius: 6px; }}
QTabBar::tab {{ background: {control}; color: {text_subtle}; padding: 12px 16px; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {text}; }}
QTabBar::tab:selected {{ background: {raised}; color: {accent}; border-bottom: 2px solid {accent}; }}
QTreeWidget {{ background: {surface}; border: 1px solid {border_strong}; border-radius: 6px; outline: none; selection-background-color: {selected_bg}; }}
QTreeWidget::item {{ padding: 10px 4px; }}
QTreeWidget::item:hover {{ background: {hover}; }}
QTreeWidget::item:selected {{ background: {selected_bg}; color: {accent}; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ padding: 11px 8px; border-radius: 6px; color: {text_subtle}; }}
QListWidget::item:hover {{ background: {hover}; }}
QListWidget::item:selected {{ background: {accent_wash}; color: {text}; }}
QProgressBar {{ background: {control}; border: none; border-radius: 6px; min-height: 12px; text-align: center; }}
QProgressBar::chunk {{ background: {grad_progress}; border-radius: 6px; }}
QStatusBar {{ background: {surface}; color: {text_muted}; }}
QSplitter::handle {{ background: {border}; width: 1px; }}
QToolTip {{ background: {raised}; color: {text}; border: 1px solid {border_strong}; padding: 6px; }}
QPushButton#outline {{ background: transparent; border: 1px dashed {border_hover}; border-radius: 7px; color: {text_subtle}; padding: 10px 16px; text-align: center; }}
QPushButton#outline:hover {{ border-color: {accent}; color: {accent}; background: transparent; }}
QPushButton#outline:checked {{ border: 1px dashed {accent}; color: {accent}; background: transparent; }}
QPushButton#outline:disabled {{ border-color: {border}; color: {text_disabled}; background: transparent; }}
QPushButton#link {{ background: transparent; border: none; color: {accent}; text-align: left; padding: 4px 2px; font-size: 12px; font-weight: 600; }}
QPushButton#link:hover {{ color: {accent_hi}; background: transparent; }}
QFrame#statTile {{ background: {control}; border: 1px solid {border}; border-radius: 9px; }}
QLabel#statValue {{ font-size: 20px; font-weight: 700; }}
QLabel#statLabel {{ color: {text_muted}; font-size: 11px; }}
QLabel#configHint {{ color: {accent}; font-size: 12px; }}
QLabel#configHint[bad="true"] {{ color: {danger}; }}
QLabel#welcomeTitle {{ font-size: 40px; font-weight: 700; }}
QLabel#welcomeText {{ color: {text_muted}; font-size: 14px; }}
QLabel#welcomeHint {{ color: {accent}; font-size: 12px; font-weight: 600; }}
QPushButton#chip {{ background: {surface}; border: 1px solid {border_strong}; border-radius: 10px; padding: 12px 14px; text-align: left; color: {text_subtle}; }}
QPushButton#chip:hover {{ border-color: {accent}; color: {text}; background: {accent_wash}; }}
QPushButton#themeCard {{ background: {surface}; border: 1px solid {border_strong}; border-radius: 12px; text-align: left; padding: 0; }}
QPushButton#themeCard:hover {{ border-color: {border_hover}; background: {surface}; }}
QPushButton#themeCard:checked {{ border: 2px solid {accent}; background: {accent_wash}; }}
QLabel#themeName {{ font-size: 15px; font-weight: 700; }}
QFrame#settingsCard {{ background: {surface}; border: 1px solid {border_strong}; border-radius: 12px; }}
QLabel#runtimeStatus {{ color: {text_subtle}; font-size: 12px; }}
QFrame#segBar {{ background: {control}; border: 1px solid {border}; border-radius: 9px; }}
QPushButton#seg {{ background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 7px 4px; color: {text_muted}; font-weight: 600; font-size: 12px; }}
QPushButton#seg:hover {{ color: {text}; }}
QPushButton#seg:checked {{ background: {raised}; border: 1px solid {border_strong}; color: {accent}; }}
QPushButton#seg[running="true"] {{ color: {accent_2}; }}
QFrame#sidebarRule {{ background: {border}; border: none; }}
QWidget#sidebarPanel, QStackedWidget#sectionStack {{ background: transparent; }}
QPushButton#ghost {{ background: transparent; border: 1px solid transparent; border-radius: 7px; color: {text_muted}; text-align: left; padding: 7px 10px; font-size: 12px; }}
QPushButton#ghost:hover {{ background: {hover}; color: {text}; }}
QPushButton#ghost:checked {{ background: {accent_wash}; color: {accent}; }}
QPushButton#ghost:disabled {{ color: {text_disabled}; background: transparent; border-color: transparent; }}
QLineEdit#chatSearch {{ padding: 7px 10px; border-radius: 8px; font-size: 12px; }}
QListWidget#runList::item {{ padding: 8px 8px; }}
QFrame#statusCard {{ background: {control}; border: 1px solid {border}; border-radius: 9px; }}
QLabel#statusTitle {{ font-weight: 600; font-size: 12px; }}
QLabel#statusDot {{ font-size: 11px; color: {text_disabled}; }}
QLabel#statusDot[state="ok"] {{ color: {accent}; }}
QLabel#statusDot[state="busy"] {{ color: {accent_2}; }}
QLabel#statusDot[state="error"] {{ color: {danger}; }}
QScrollBar:vertical {{ background: transparent; width: 9px; }}
QScrollBar::handle:vertical {{ background: {border_strong}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {border_hover}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; }}
QScrollBar::handle:horizontal {{ background: {border_strong}; border-radius: 4px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {border_hover}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""

def apply_palette(name: str) -> str:
    """Switch the shared COLORS to a palette and return the matching stylesheet."""
    COLORS.update(PALETTES[name if name in PALETTES else DEFAULT_PALETTE])
    return build_stylesheet()


STYLE = build_stylesheet()
