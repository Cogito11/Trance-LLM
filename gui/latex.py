"""Turn the LaTeX that chat models write into readable Unicode text (no extra packages).

This is a text approximation, not a typesetter: Greek letters, operators, roots, super- and subscripts, accents
and simple fractions become Unicode ("x = (-b ± √(b² - 4ac))/2a"). Stacked fractions are written inline,
and anything it does not understand (matrices, alignment environments, unknown commands) is left as the
original LaTeX instead of being guessed at.
"""
from __future__ import annotations

import re

SYMBOLS = {
    # Greek
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    # operators and relations
    "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓", "ast": "∗", "star": "⋆", "circ": "∘", "bullet": "•",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "approx": "≈", "sim": "∼", "simeq": "≃",
    "cong": "≅", "equiv": "≡", "propto": "∝", "ll": "≪", "gg": "≫", "prec": "≺", "succ": "≻",
    "in": "∈", "notin": "∉", "ni": "∋", "subset": "⊂", "supset": "⊃", "subseteq": "⊆", "supseteq": "⊇",
    "cup": "∪", "cap": "∩", "setminus": "∖", "emptyset": "∅", "varnothing": "∅", "forall": "∀", "exists": "∃",
    "nexists": "∄", "neg": "¬", "lnot": "¬", "land": "∧", "wedge": "∧", "lor": "∨", "vee": "∨", "oplus": "⊕",
    "otimes": "⊗", "perp": "⊥", "parallel": "∥", "angle": "∠", "triangle": "△", "therefore": "∴", "because": "∵",
    # calculus and big operators
    "infty": "∞", "partial": "∂", "nabla": "∇", "sum": "∑", "prod": "∏", "coprod": "∐", "int": "∫", "iint": "∬",
    "iiint": "∭", "oint": "∮", "bigcup": "⋃", "bigcap": "⋂", "sqrt": "√", "hbar": "ħ", "ell": "ℓ", "Re": "ℜ",
    "Im": "ℑ", "aleph": "ℵ", "wp": "℘", "degree": "°", "prime": "′",
    # arrows
    "to": "→", "rightarrow": "→", "leftarrow": "←", "gets": "←", "leftrightarrow": "↔", "Rightarrow": "⇒",
    "Leftarrow": "⇐", "Leftrightarrow": "⇔", "implies": "⟹", "impliedby": "⟸", "iff": "⟺", "mapsto": "↦",
    "uparrow": "↑", "downarrow": "↓", "longrightarrow": "⟶", "longleftarrow": "⟵", "rightleftharpoons": "⇌",
    # dots and delimiters
    "ldots": "…", "dots": "…", "cdots": "⋯", "vdots": "⋮", "ddots": "⋱", "langle": "⟨", "rangle": "⟩",
    "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉", "vert": "|", "Vert": "‖", "lbrace": "{", "rbrace": "}",
    "lbrack": "[", "rbrack": "]",
    # named functions keep their name
    "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot", "sec": "sec", "csc": "csc", "arcsin": "arcsin",
    "arccos": "arccos", "arctan": "arctan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "log": "log", "ln": "ln",
    "lg": "lg", "exp": "exp", "min": "min", "max": "max", "sup": "sup", "inf": "inf", "lim": "lim", "limsup": "limsup",
    "liminf": "liminf", "det": "det", "dim": "dim", "ker": "ker", "deg": "deg", "gcd": "gcd", "arg": "arg",
    "Pr": "Pr", "mod": "mod", "bmod": "mod",
}
ESCAPED = {"%": "%", "$": "$", "&": "&", "#": "#", "_": "_", "{": "{", "}": "}", ",": " ", ";": " ", ":": " ", "!": "",
           " ": " ", "|": "‖", ">": " ", "<": " "}
SPACING = {"quad": "  ", "qquad": "    ", "enspace": " ", "thinspace": " ", "medspace": " ", "thickspace": " "}
DROPPED = {"left", "right", "big", "Big", "bigg", "Bigg", "displaystyle", "textstyle", "scriptstyle", "limits", "nolimits",
           "mathstrut", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "relax", "nonumber", "notag"}
TEXTLIKE = {"text", "textrm", "textit", "textbf", "mathrm", "mathit", "mathbf", "mathsf", "mathtt", "mathcal", "mathscr",
            "mathfrak", "boldsymbol", "bm", "operatorname", "mbox", "textsf", "texttt", "emph"}
BLACKBOARD = {"R": "ℝ", "Z": "ℤ", "N": "ℕ", "Q": "ℚ", "C": "ℂ", "P": "ℙ", "H": "ℍ", "E": "𝔼"}
ACCENTS = {"vec": "\u20d7", "hat": "\u0302", "widehat": "\u0302", "bar": "\u0304", "overline": "\u0305", "tilde": "\u0303",
           "widetilde": "\u0303", "dot": "\u0307", "ddot": "\u0308", "check": "\u030c", "acute": "\u0301", "grave": "\u0300",
           "breve": "\u0306", "underline": "\u0332"}
VULGAR = {("1", "2"): "½", ("1", "3"): "⅓", ("2", "3"): "⅔", ("1", "4"): "¼", ("3", "4"): "¾", ("1", "5"): "⅕",
          ("2", "5"): "⅖", ("3", "5"): "⅗", ("4", "5"): "⅘", ("1", "6"): "⅙", ("5", "6"): "⅚", ("1", "8"): "⅛",
          ("3", "8"): "⅜", ("5", "8"): "⅝", ("7", "8"): "⅞"}

_SUP = dict(zip("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"))
_SUP.update(zip("abcdefghijklmnoprstuvwxyz", "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ"))
_SUP.update(zip("ABDEGHIJKLMNOPRTUVW", "ᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ"))
_SUP.update({"′": "′", "*": "*", "∗": "*", "°": "°", "−": "⁻", " ": " "})
_SUB = dict(zip("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"))
_SUB.update(zip("aehijklmnoprstuvx", "ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
_SUB.update({"β": "ᵦ", "γ": "ᵧ", "ρ": "ᵨ", "φ": "ᵩ", "χ": "ᵪ", "−": "₋", " ": " "})

_SIMPLE = re.compile(r"^[\w.′°]+$")


def _atom(text: str) -> str:
    """Wrap a multi-part expression in parentheses so it can sit next to / and √ unambiguously."""
    text = text.strip()
    if not text or _SIMPLE.match(text): return text
    if text[0] == "(" and text[-1] == ")" and text.count("(") == text.count(")"): return text
    return f"({text})"


def _map(text: str, table: dict, fallback_mark: str) -> str:
    if text and all(ch in table for ch in text): return "".join(table[ch] for ch in text)
    return f"{fallback_mark}({text})" if len(text) > 1 else f"{fallback_mark}{text}"


class _Parser:
    def __init__(self, source: str):
        self.s, self.i = source, 0

    def eof(self) -> bool:
        return self.i >= len(self.s)

    def group(self) -> str:
        """The next argument: a {braced group}, a \\command, or a single character."""
        while not self.eof() and self.s[self.i] == " ": self.i += 1
        if self.eof(): return ""
        if self.s[self.i] == "{":
            depth, start = 0, self.i + 1
            for j in range(self.i, len(self.s)):
                if self.s[j] == "{": depth += 1
                elif self.s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        self.i = j + 1
                        return self.s[start:j]
            body, self.i = self.s[start:], len(self.s)     # unbalanced: take the rest
            return body
        if self.s[self.i] == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", self.s[self.i:])
            self.i += m.end()
            return m.group(0)
        ch = self.s[self.i]; self.i += 1
        return ch

    def optional(self) -> str | None:
        j = self.i
        while j < len(self.s) and self.s[j] == " ": j += 1
        if j < len(self.s) and self.s[j] == "[":
            end = self.s.find("]", j)
            if end != -1:
                self.i = end + 1
                return self.s[j + 1:end]
        return None

    def parse(self) -> str:
        out: list[str] = []
        restore_space = False                    # "sin" + superscript: the space after the name moves after the script
        while not self.eof():
            ch = self.s[self.i]
            if restore_space and ch not in "^_":
                out.append(" "); restore_space = False
            if ch == "\\":
                out.append(self.command())
            elif ch in "^_":
                if out and out[-1].endswith(" ") and out[-1].strip() in _SPACE_AFTER:
                    out[-1] = out[-1].rstrip(); restore_space = True
                self.i += 1
                arg = convert(self.group())
                out.append(_map(arg, _SUP if ch == "^" else _SUB, ch))
            elif ch == "{":
                out.append(convert(self.group()))
            elif ch == "}":
                self.i += 1
            elif ch == "~":
                out.append(" "); self.i += 1
            elif ch == "&":
                out.append(" "); self.i += 1          # alignment marker in aligned/array environments
            elif ch == "-":
                out.append("\u2212"); self.i += 1
            else:
                out.append(ch); self.i += 1
        if restore_space: out.append(" ")
        return "".join(out)

    def command(self) -> str:
        m = re.match(r"\\([A-Za-z]+|.)", self.s[self.i:])
        name = m.group(1); self.i += m.end()
        if name in ("frac", "dfrac", "tfrac", "cfrac"):
            num, den = convert(self.group()), convert(self.group())
            return VULGAR.get((num, den)) or f"{_atom(num)}/{_atom(den)}"
        if name == "binom":
            return f"C({convert(self.group())}, {convert(self.group())})"
        if name == "sqrt":
            root = self.optional(); body = _atom(convert(self.group()))
            index = _map(convert(root), _SUP, "^") if root else ""
            return f"{index}√{body}"
        if name in ACCENTS:
            inner = convert(self.group())
            return inner + ACCENTS[name] if len(inner) == 1 else f"{inner}{ACCENTS[name]}" if name in ("overline", "underline") else inner
        if name == "mathbb":
            inner = self.group()
            return "".join(BLACKBOARD.get(c, c) for c in inner)
        if name in TEXTLIKE:
            return self.group()
        if name in SPACING: return SPACING[name]
        if name in DROPPED: return ""
        if name in SYMBOLS:
            return SYMBOLS[name] + (" " if name in _SPACE_AFTER else "")
        if name == "\\": return "\n"
        if name in ESCAPED: return ESCAPED[name]
        if self.i < len(self.s) and self.s[self.i] == "{":      # not understood: show it, arguments and all, as written
            return f"\\{name}{{{self.group()}}}"
        return "\\" + name


_SPACE_AFTER = {"sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "log", "ln", "lg",
                "exp", "min", "max", "lim", "limsup", "liminf", "det", "dim", "ker", "deg", "gcd", "arg", "Pr", "mod", "bmod"}


def convert(source: str) -> str:
    return _Parser(source).parse()


def to_text(latex: str) -> str:
    """Readable Unicode for a LaTeX math fragment (see the module docstring for what is and is not handled)."""
    source = re.sub(r"\\(begin|end)\{(aligned|align\*?|gather\*?|split|equation\*?)\}", " ", latex)
    text = convert(source.strip())
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\(\s+", "(", text); text = re.sub(r"\s+\)", ")", text)      # \left( x \right) leaves padding inside
    return "\n".join(line.strip() for line in text.strip().split("\n") if line.strip())
