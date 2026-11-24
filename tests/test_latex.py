"""LaTeX to readable Unicode, as used for math in model replies."""
from __future__ import annotations

import unittest

from gui.latex import to_text

CASES = {
    r"E = mc^2": "E = mc²",
    r"\frac{a}{b}": "a/b",
    r"\frac{1}{2}": "½",
    r"\frac{x+1}{2}": "(x+1)/2",
    r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}": "x = (−b ± √(b² − 4ac))/2a",
    r"\sum_{i=1}^{n} i^2": "∑ᵢ₌₁ⁿ i²",
    r"\int_0^1 x\,dx": "∫₀¹ x dx",
    r"\alpha + \beta = \gamma": "α + β = γ",
    r"\vec{v}": "v⃗",
    r"\hat{x}": "x̂",
    r"\text{if } x > 0": "if x > 0",
    r"x_{ij} + a_{n+1}": "xᵢⱼ + aₙ₊₁",
    r"a \leq b \neq c": "a ≤ b ≠ c",
    r"\mathbb{R}^n": "ℝⁿ",
    r"\sqrt[3]{27} = 3": "³√27 = 3",
    r"\left( \frac{a}{b} \right)^2": "(a/b)²",
    r"\binom{n}{k}": "C(n, k)",
    r"10^{-3}": "10⁻³",
    r"\sin^2 x + \cos^2 x = 1": "sin² x + cos² x = 1",
    r"\lim_{x \to \infty} \frac{1}{x} = 0": "lim_(x → ∞) 1/x = 0",
    r"\nabla \cdot \mathbf{E}": "∇ · E",
}


class LatexTests(unittest.TestCase):
    def test_common_expressions_become_readable_unicode(self):
        for source, expected in CASES.items():
            self.assertEqual(to_text(source), expected, source)

    def test_what_cannot_be_converted_is_left_as_written_not_guessed(self):
        self.assertEqual(to_text(r"\foo{x}"), r"\foo{x}")
        self.assertEqual(to_text(r"x^{\theta}"), "x^θ")                 # no superscript theta exists
        self.assertEqual(to_text(r"e^{i\pi} + 1 = 0"), "e^(iπ) + 1 = 0")
        self.assertIn(r"\begin{pmatrix}", to_text(r"\begin{pmatrix} a & b \end{pmatrix}"))   # matrices are not typeset
        self.assertEqual((to_text(""), to_text("   ")), ("", ""))

    def test_minus_signs_are_real_minus_signs_but_text_keeps_its_hyphens(self):
        self.assertEqual(to_text("a - b"), "a − b")
        self.assertEqual(to_text(r"\text{well-known} - 3"), "well-known − 3")

    def test_multi_line_environments_keep_their_lines(self):
        self.assertEqual(to_text(r"\begin{aligned} a &= b \\ c &= d \end{aligned}"), "a = b\nc = d")

    def test_unbalanced_braces_do_not_crash(self):
        for broken in (r"\frac{a", r"x^{", r"\sqrt{", "}{", r"\frac", "^", "_{"):
            self.assertIsInstance(to_text(broken), str, broken)


if __name__ == "__main__":
    unittest.main()
