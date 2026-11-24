"""Turn chat-model markdown into themed rich text, using Qt's own markdown reader (no extra packages).

Qt produces usable HTML but with hard-coded light-theme colors, no code-block background and tables that
vanish on a dark theme, so the output is restyled from the app's color tokens. Beyond plain markdown this adds:

* callouts: GitHub alerts (``> [!NOTE]``, TIP, IMPORTANT, WARNING, CAUTION) and the common Obsidian types
  become tinted boxes;
* math: ``$...$``, ``$$...$$``, ``\\(...\\)`` and ``\\[...\\]`` are shown as readable Unicode (see gui/latex.py);
* a quieter "thinking" style for a model's reasoning.

Raw HTML in the model's text is shown as text instead of being interpreted, and the model's own single line
breaks are kept (small models often write poems and lists that way).
"""
from __future__ import annotations

import html
import re
from functools import lru_cache

from PySide6.QtGui import QTextDocument

from .latex import to_text
from .theme import COLORS as C

_OPEN, _CLOSE = "\ue000", "\ue001"            # private-use markers that carry a placeholder through the markdown reader
_FENCE = re.compile(r"^\s*(```+|~~~+)")
_CODE_SPAN = re.compile(r"(`+[^`\n]*`+)")
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_START = re.compile(r"<(?=[A-Za-z/!?])")
_PRE_RUN = re.compile(r"(?:<pre\b[^>]*>.*?</pre>\s*)+", re.DOTALL)
_QUOTE = re.compile(r"^\s{0,3}>")
_CALLOUT_HEAD = re.compile(r"^\s{0,3}>\s*\[!([A-Za-z]+)\][+-]?[ \t]*(.*)$")
_DISPLAY_MATH = (re.compile(r"\$\$(.+?)\$\$", re.DOTALL), re.compile(r"\\\[(.+?)\\\]", re.DOTALL))
_INLINE_MATH = (re.compile(r"\\\((.+?)\\\)"), re.compile(r"(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<![\s\\])\$(?!\d)"))
_PLACEHOLDER_BLOCK = re.compile(r"<p[^>]*>\s*" + _OPEN + r"([BC])(\d+)" + _CLOSE + r"\s*</p>")
_PLACEHOLDER_ANY = re.compile(_OPEN + r"([BCI])(\d+)" + _CLOSE)

_HEADINGS = {
    "reply": {"xx-large": "24px", "x-large": "20px", "large": "17px", "medium": "15px", "small": "13px"},
    "thinking": {"xx-large": "16px", "x-large": "15px", "large": "14px", "medium": "13px", "small": "12px"},
}

# type -> (color token, default title, icon). Colors follow GitHub's alerts where the two systems disagree.
_CALLOUTS = {
    "note": ("info", "Note", "\u2139"), "info": ("info", "Info", "\u2139"), "todo": ("info", "To do", "\u2610"),
    "abstract": ("info", "Summary", "\u2261"), "summary": ("info", "Summary", "\u2261"), "tldr": ("info", "TL;DR", "\u2261"),
    "tip": ("success", "Tip", "\u2726"), "hint": ("success", "Hint", "\u2726"), "success": ("success", "Success", "\u2714"),
    "check": ("success", "Check", "\u2714"), "done": ("success", "Done", "\u2714"),
    "important": ("important", "Important", "\u2605"), "example": ("important", "Example", "\u25b8"),
    "question": ("important", "Question", "?"), "help": ("important", "Help", "?"), "faq": ("important", "FAQ", "?"),
    "warning": ("warning", "Warning", "\u26a0"), "attention": ("warning", "Attention", "\u26a0"),
    "caution": ("danger", "Caution", "\u2716"), "danger": ("danger", "Danger", "\u2716"), "error": ("danger", "Error", "\u2716"),
    "failure": ("danger", "Failure", "\u2716"), "fail": ("danger", "Failure", "\u2716"), "missing": ("danger", "Missing", "\u2716"),
    "bug": ("danger", "Bug", "\u2716"), "quote": ("text_muted", "Quote", "\u275d"), "cite": ("text_muted", "Quote", "\u275d"),
}
_MATH_FONT = "font-family:'DejaVu Serif','Times New Roman',serif; font-style:italic"


class _Extras:
    """Things pulled out of the text before the markdown reader sees it, restored afterwards."""

    def __init__(self):
        self.inline: list[str] = []
        self.block: list[str] = []
        self.callouts: list[tuple[str, str, str]] = []


def _token(kind: str, n: int) -> str:
    return f"{_OPEN}{kind}{n}{_CLOSE}"


def _escape_line(line: str, extras: _Extras) -> str:
    """Math to placeholders; tag-like text shown literally (outside inline code); <br> becomes a break."""
    is_row = line.lstrip().startswith("|")
    parts = _CODE_SPAN.split(line)
    for n in range(0, len(parts), 2):                      # even parts are outside `code spans`
        piece = parts[n]
        for pattern in _INLINE_MATH:
            def stash(match):
                extras.inline.append(match.group(1)); return _token("I", len(extras.inline) - 1)
            piece = pattern.sub(stash, piece)
        piece = _BR.sub(" " if is_row else "  \n", piece)
        parts[n] = _TAG_START.sub("&lt;", piece)
    return "".join(parts)


def _extract_callouts(lines: list[str], extras: _Extras) -> list[str]:
    out, n = [], 0
    while n < len(lines):
        head = _CALLOUT_HEAD.match(lines[n])
        if head and head.group(1).lower() in _CALLOUTS:
            body, n = [], n + 1
            while n < len(lines) and _QUOTE.match(lines[n]):
                body.append(re.sub(r"^\s{0,3}>\s?", "", lines[n])); n += 1
            extras.callouts.append((head.group(1).lower(), head.group(2).strip(), "\n".join(body).strip("\n")))
            out += ["", _token("C", len(extras.callouts) - 1), ""]
        else:
            out.append(lines[n]); n += 1
    return out


def _prepare_normal(lines: list[str], extras: _Extras, allow_callouts: bool) -> list[str]:
    if allow_callouts: lines = _extract_callouts(lines, extras)
    text = "\n".join(lines)
    for pattern in _DISPLAY_MATH:
        def stash(match):
            extras.block.append(match.group(1)); return f"\n\n{_token('B', len(extras.block) - 1)}\n\n"
        text = pattern.sub(stash, text)
    lines = text.split("\n")
    out = []
    for n, line in enumerate(lines):
        line = _escape_line(line, extras)
        following = lines[n + 1] if n + 1 < len(lines) else ""
        if line.strip() and following.strip() and not line.endswith("  ") and not _FENCE.match(following):
            line += "  "                                    # a single newline in the model's text is a line break
        out.append(line)
    return out


def prepare(text: str, extras: _Extras | None = None, allow_callouts: bool = True) -> str:
    """Make model text safe and faithful before it goes to the markdown reader."""
    extras = extras if extras is not None else _Extras()
    out, normal, fence = [], [], ""
    for line in text.replace("\r\n", "\n").split("\n"):
        marker = _FENCE.match(line)
        if marker:
            token = marker.group(1)[:3]
            if not fence:
                out += _prepare_normal(normal, extras, allow_callouts); normal = []
                fence = token; out.append(line); continue
            if token == fence:
                fence = ""; out.append(line); continue
        if fence: out.append(line)                          # code is left exactly as written
        else: normal.append(line)
    out += _prepare_normal(normal, extras, allow_callouts)
    return "\n".join(out)


def _mix(foreground: str, background: str, weight: float) -> str:
    f, b = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (foreground, background))
    return "#" + "".join(f"{round(x * weight + y * (1 - weight)):02x}" for x, y in zip(f, b))


def _signature() -> tuple:
    return tuple(C[k] for k in ("text", "text_muted", "text_subtle", "accent", "control", "border", "border_strong", "surface",
                                "info", "success", "warning", "important", "danger"))


def _math_html(source: str, block: bool, style: str) -> str:
    text = to_text(source)
    if block:
        size = "14px" if style == "thinking" else "17px"
        lines = "<br>".join(html.escape(line) for line in text.split("\n"))
        return f'<p align="center" style="margin-top:8px;margin-bottom:8px;font-size:{size}"><span style="{_MATH_FONT}">{lines}</span></p>'
    return f'<span style="{_MATH_FONT}">{html.escape(text.replace(chr(10), " "))}</span>'


def _callout_html(kind: str, title: str, body: str, style: str, depth: int) -> str:
    token, default_title, icon = _CALLOUTS[kind]
    color = C.get(token, C["text_muted"])
    tint = _mix(color, C["surface"], 0.14)
    inner = _body(body, style, depth + 1) if body.strip() else ""
    heading = html.escape(title or default_title)
    return (f'<table width="100%" cellspacing="0" cellpadding="0" style="margin-top:6px;margin-bottom:6px"><tr>'
            f'<td width="4" bgcolor="{color}"></td><td bgcolor="{tint}" style="padding:10px 14px">'
            f'<p style="margin-top:0px;margin-bottom:4px;color:{color}"><b>{icon} {heading}</b></p>{inner}</td></tr></table>')


def _style(body: str, style: str) -> str:
    blocks: list[str] = []

    def stash_code(match: re.Match) -> str:
        blocks.append(f'<table width="100%" cellspacing="0" cellpadding="10" style="background-color:{C["control"]};'
                      f' border:1px solid {C["border"]}; font-style:normal"><tr><td>{match.group(0)}</td></tr></table>')
        return f"@@CODE{len(blocks) - 1}@@"

    body = _PRE_RUN.sub(stash_code, body)
    for name, size in _HEADINGS[style].items():
        body = body.replace(f"font-size:{name};", f"font-size:{size};")
    body = body.replace("font-family:'monospace';", f"font-family:'monospace'; background-color:{C['control']}; font-style:normal;")
    body = body.replace("color:#0000ff;", f"color:{C['accent']};")
    body = body.replace('<table border="1" style=" border-collapse:collapse;" cellpadding="4">',
                        f'<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; border-color:{C["border_strong"]};">')
    for n, block in enumerate(blocks):
        body = body.replace(f"@@CODE{n}@@", block)
    return body


def _body(text: str, style: str, depth: int = 0) -> str:
    """The themed HTML for markdown text, without its outer wrapper (callouts render their contents with this)."""
    extras = _Extras()
    document = QTextDocument()
    document.setMarkdown(prepare(text, extras, allow_callouts=depth < 2), QTextDocument.MarkdownFeature.MarkdownDialectGitHub)
    match = re.search(r"<body[^>]*>(.*)</body>", document.toHtml(), re.DOTALL)
    result = _style(match.group(1) if match else "", style)

    def block(m: re.Match) -> str:
        n = int(m.group(2))
        return _callout_html(*extras.callouts[n], style, depth) if m.group(1) == "C" else _math_html(extras.block[n], True, style)

    def anything(m: re.Match) -> str:                       # placeholders that did not end up alone in a paragraph (inside a list, say)
        n = int(m.group(2))
        if m.group(1) == "C": return _callout_html(*extras.callouts[n], style, depth)
        return _math_html(extras.block[n] if m.group(1) == "B" else extras.inline[n], False, style)

    return _PLACEHOLDER_ANY.sub(anything, _PLACEHOLDER_BLOCK.sub(block, result))


@lru_cache(maxsize=256)
def _render(text: str, signature: tuple, style: str) -> str:
    inner = _body(text, style)
    if style == "thinking":
        wrap = f"color:{C['text_muted']}; font-size:12px; line-height:140%; margin-left:14px; font-style:italic"
    else:
        wrap = f"color:{C['text']}; line-height:150%"
    return f'<div style="{wrap}">{inner}</div>'


def to_html(text: str, style: str = "reply") -> str:
    """Themed HTML for a model reply ("reply") or its reasoning ("thinking"); cached, since a transcript is
    re-drawn for every streamed chunk."""
    return _render(text, _signature(), style) if text.strip() else ""
