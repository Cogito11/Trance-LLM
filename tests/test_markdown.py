"""Markdown rendering for model replies: callouts, math, the thinking style, and what must not be touched."""
from __future__ import annotations

import unittest

from PySide6.QtWidgets import QApplication

from gui import markdown, theme
from gui.markdown import to_html

C = theme.COLORS


class MarkdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self):
        theme.apply_palette("teal")

    def clean(self, html):
        self.assertNotIn("\ue000", html); self.assertNotIn("\ue001", html)      # placeholders never leak into the output
        return html

    # ---- callouts
    def test_github_alerts_become_tinted_boxes_in_their_own_colors(self):
        for kind, token, title in (("NOTE", "info", "Note"), ("TIP", "success", "Tip"), ("IMPORTANT", "important", "Important"),
                                   ("WARNING", "warning", "Warning"), ("CAUTION", "danger", "Caution")):
            html = self.clean(to_html(f"> [!{kind}]\n> Body text here"))
            self.assertIn(f'bgcolor="{C[token]}"', html, kind)
            self.assertIn(title, html); self.assertIn("Body text here", html)
            self.assertNotIn(f"[!{kind}]", html)

    def test_a_custom_title_replaces_the_default_and_lowercase_obsidian_types_work(self):
        html = to_html("> [!IMPORTANT] Remember this\n> Details")
        self.assertIn("Remember this", html); self.assertNotIn(">★ Important", html)
        self.assertIn(f'bgcolor="{C["success"]}"', to_html("> [!success]\n> done"))
        self.assertIn(f'bgcolor="{C["danger"]}"', to_html("> [!bug]\n> broken"))

    def test_callout_bodies_are_markdown_with_math(self):
        html = self.clean(to_html("> [!TIP]\n> Use **bold**, a list:\n> - one\n> - two\n> and $x^2$"))
        for expected in ("font-weight:700", "<li", "x²"):
            self.assertIn(expected, html)

    def test_consecutive_and_nested_callouts_all_render(self):
        two = to_html("> [!NOTE]\n> first\n\n> [!WARNING]\n> second")
        self.assertIn(f'bgcolor="{C["info"]}"', two); self.assertIn(f'bgcolor="{C["warning"]}"', two)
        nested = to_html("> [!NOTE]\n> outer\n> > [!TIP]\n> > inner")
        self.assertIn(f'bgcolor="{C["info"]}"', nested); self.assertIn(f'bgcolor="{C["success"]}"', nested)
        self.assertIn("outer", nested); self.assertIn("inner", nested)

    def test_unknown_types_plain_quotes_and_code_are_left_alone(self):
        unknown = to_html("> [!FOO]\n> text")
        self.assertNotIn("bgcolor", unknown); self.assertIn("[!FOO]", unknown)
        self.assertNotIn("bgcolor", to_html("> just a quote\n> second line"))
        fenced = to_html("```\n> [!NOTE]\n> inside code\n```")
        self.assertNotIn(f'bgcolor="{C["info"]}"', fenced); self.assertIn("[!NOTE]", fenced)

    def test_a_callout_with_no_body_still_shows_its_title(self):
        html = self.clean(to_html("> [!WARNING]"))
        self.assertIn("Warning", html); self.assertIn(f'bgcolor="{C["warning"]}"', html)

    # ---- math
    def test_inline_and_display_math_are_converted(self):
        html = self.clean(to_html(r"Energy $E = mc^2$ and \(a_i\) and:" "\n\n" r"$$\frac{a}{b}$$" "\n\n" r"\[ x^{2} \]"))
        for expected in ("E = mc²", "aᵢ", "a/b", "x²"):
            self.assertIn(expected, html)
        self.assertGreaterEqual(html.count('align="center"'), 2)              # both display forms are centered blocks
        self.assertNotIn("$", html); self.assertNotIn("\\frac", html)

    def test_multi_line_display_math_is_one_block(self):
        html = self.clean(to_html("Before\n\n$$\n\\sum_{i=1}^{n} i\n= \\frac{n(n+1)}{2}\n$$\n\nAfter"))
        self.assertIn("∑ᵢ₌₁ⁿ", html); self.assertIn("Before", html); self.assertIn("After", html)

    def test_dollar_signs_in_ordinary_text_are_not_math(self):
        for text in ("It costs $5 and $6 today.", "Between $5 and $10.", "A lone $ sign", "Price: $ 5 $ here"):
            html = self.clean(to_html(text))
            self.assertNotIn("DejaVu Serif", html, text)                      # no math span was produced
        self.assertIn("$5 and $6", to_html("It costs $5 and $6 today."))

    def test_math_inside_code_is_never_touched(self):
        inline = self.clean(to_html("Use `$x^2$` literally"))
        self.assertIn("$x^2$", inline); self.assertNotIn("DejaVu Serif", inline)
        fenced = self.clean(to_html("```\nprice = $5 * $6\n$$ not math $$\n```"))
        self.assertIn("$$ not math $$", fenced); self.assertNotIn("DejaVu Serif", fenced)

    def test_markdown_characters_inside_math_are_not_formatting(self):
        html = self.clean(to_html("Then $a_i * b_i$ and $x_1 + x_2$ are summed."))
        self.assertIn("aᵢ * bᵢ", html); self.assertIn("x₁ + x₂", html)

    def test_math_with_pipes_survives_a_table(self):
        html = self.clean(to_html("| expr | n |\n|---|---|\n| $|a|$ | 1 |"))
        self.assertIn("<table", html); self.assertIn("|a|", html)

    def test_math_html_is_escaped(self):
        html = self.clean(to_html("$a < b$ and $x<y>z$"))
        self.assertIn("a &lt; b", html); self.assertNotIn("<y", html)

    # ---- styles
    def test_thinking_style_is_quieter_and_smaller(self):
        reply, thought = to_html("## Title\n\nSome *words*"), to_html("## Title\n\nSome *words*", "thinking")
        self.assertIn("font-size:20px", reply); self.assertIn("font-size:15px", thought)          # headings shrink
        self.assertIn(C["text_muted"], thought); self.assertIn("margin-left:14px", thought); self.assertIn("font-style:italic", thought)
        self.assertNotIn("margin-left:14px", reply)
        self.assertIn(C["text"], reply)

    def test_thinking_gets_callouts_math_and_tables_too(self):
        html = self.clean(to_html("1. step $b^2 - 4ac$\n\n> [!NOTE]\n> careful\n\n| a | b |\n|---|---|\n| 1 | 2 |", "thinking"))
        for expected in ("b² − 4ac", "careful", "<table"):
            self.assertIn(expected, html)

    def test_output_follows_the_theme_even_for_cached_text(self):
        text = "[a link](https://example.com)"
        before = to_html(text)
        theme.apply_palette("amber")
        after = to_html(text)
        self.assertIn(theme.PALETTES["teal"]["accent"], before); self.assertIn(theme.PALETTES["amber"]["accent"], after)
        self.assertNotEqual(before, after)

    def test_empty_text_gives_nothing_and_html_is_still_escaped(self):
        self.assertEqual(to_html("   \n"), "")
        html = to_html("<b>not bold</b> <img src='http://evil.example/x.png'>")
        self.assertIn("&lt;b&gt;", html); self.assertNotIn("<img", html)

    def test_prepare_leaves_code_blocks_exactly_as_written(self):
        source = "text\n```\n<b> $x$ > quote\n```\nafter"
        self.assertIn("<b> $x$ > quote", markdown.prepare(source))


if __name__ == "__main__":
    unittest.main()
