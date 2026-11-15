"""Turn chat-model markdown into themed rich text, using Qt's own markdown reader (no extra packages).

Qt produces usable HTML but with hard-coded light-theme colors, no code-block background and tables that
vanish on a dark theme, so the output is restyled from the app's color tokens. Raw HTML in the model's text is
shown as text instead of being interpreted, and the model's own single line breaks are kept (small models
often write poems and lists that way).
"""
from __future__ import annotations

import re
from functools import lru_cache

from PySide6.QtGui import QTextDocument

from .theme import COLORS as C

_FENCE = re.compile(r"^\s*(```+|~~~+)")
_CODE_SPAN = re.compile(r"(`+[^`\n]*`+)")
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_START = re.compile(r"<(?=[A-Za-z/!?])")
_PRE_RUN = re.compile(r"(?:<pre\b[^>]*>.*?</pre>\s*)+", re.DOTALL)
_HEADING_SIZES = {"xx-large": "24px", "x-large": "20px", "large": "17px", "medium": "15px", "small": "13px"}


def _escape_line(line: str) -> str:
    """Show tag-like text literally (outside inline code); <br> becomes a break, or a space inside a table row."""
    is_row = line.lstrip().startswith("|")
    parts = _CODE_SPAN.split(line)
    for n in range(0, len(parts), 2):                      # even parts are outside `code spans`
        parts[n] = _BR.sub(" " if is_row else "  \n", parts[n])
        parts[n] = _TAG_START.sub("&lt;", parts[n])
    return "".join(parts)


def prepare(text: str) -> str:
    """Make model text safe and faithful before it goes to the markdown reader."""
    lines = text.replace("\r\n", "\n").split("\n")
    out, fence = [], ""
    for n, line in enumerate(lines):
        marker = _FENCE.match(line)
        if marker:
            token = marker.group(1)[:3]
            if not fence: fence = token
            elif token == fence: fence = ""
            out.append(line); continue
        if fence:
            out.append(line); continue                      # code is left exactly as written
        line = _escape_line(line)
        following = lines[n + 1] if n + 1 < len(lines) else ""
        if line.strip() and following.strip() and not line.endswith("  ") and not _FENCE.match(following):
            line += "  "                                    # a single newline in the model's text is a line break
        out.append(line)
    return "\n".join(out)


def _signature() -> tuple:
    return tuple(C[k] for k in ("text", "text_subtle", "accent", "control", "border", "border_strong"))


def _style(body: str) -> str:
    blocks: list[str] = []

    def stash_code(match: re.Match) -> str:
        blocks.append(f'<table width="100%" cellspacing="0" cellpadding="10" style="background-color:{C["control"]};'
                      f' border:1px solid {C["border"]};"><tr><td>{match.group(0)}</td></tr></table>')
        return f"@@CODE{len(blocks) - 1}@@"

    body = _PRE_RUN.sub(stash_code, body)
    for name, size in _HEADING_SIZES.items():
        body = body.replace(f"font-size:{name};", f"font-size:{size};")
    body = body.replace("font-family:'monospace';", f"font-family:'monospace'; background-color:{C['control']};")
    body = body.replace("color:#0000ff;", f"color:{C['accent']};")
    body = body.replace('<table border="1" style=" border-collapse:collapse;" cellpadding="4">',
                        f'<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; border-color:{C["border_strong"]};">')
    for n, block in enumerate(blocks):
        body = body.replace(f"@@CODE{n}@@", block)
    return f'<div style="color:{C["text"]}; line-height:150%">{body}</div>'


@lru_cache(maxsize=256)
def _render(text: str, signature: tuple) -> str:
    document = QTextDocument()
    document.setMarkdown(prepare(text), QTextDocument.MarkdownFeature.MarkdownDialectGitHub)
    match = re.search(r"<body[^>]*>(.*)</body>", document.toHtml(), re.DOTALL)
    return _style(match.group(1) if match else "")


def to_html(text: str) -> str:
    """Themed HTML for a model reply; cached, since a transcript is re-drawn for every streamed chunk."""
    return _render(text, _signature()) if text.strip() else ""
