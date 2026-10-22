"""Plain-text step math -> LaTeX, so the app can typeset it like a worked solution on paper.

Steps are written as readable text that mixes a few words with the math
("let u = x + 1, so du = 1 dx", "∫x^2 dx = x^3/3", "d/dx[sin(x)] = cos(x)").
This splits a step at its = / → / ; signs, then splits each piece into runs of
words (typeset as text) and runs of math (typeset through the same safe SymPy
parser the math tool uses, or lightly converted if SymPy can't read them).
It never raises: the worst case is a piece shown as plain text.
"""

from __future__ import annotations

import re

import sympy as sp

from .math_tool import _check_safe, _parse, normalize, unicode_math

_EXTRA = {"L": sp.Symbol("lambda"), "u": sp.Symbol("u"), "C": sp.Symbol("C"), "erfi": sp.erfi, "erf": sp.erf,
          "atan": sp.atan, "atan2": sp.atan2}
_FUNCTIONS = {"sqrt", "sin", "cos", "tan", "log", "ln", "exp", "abs", "det", "arctan", "atan", "atan2", "erf", "erfi",
              "lim", "max", "min", "gcd", "lcm", "mod"}
_SHORT_WORDS = {"so", "by", "is", "in", "to", "of", "at", "it", "be", "an", "as", "on", "we", "up", "if", "or"}

# Top-level separators (outside brackets), kept in the output. Longer first.
_SEPARATORS = [" → ", " ≈ ", " = ", "; ", ", ", " ± "]
_SEPARATOR_LATEX = {" → ": r" \;\longrightarrow\; ", " ≈ ": r" \approx ", " = ": " = ", "; ": r",\qquad ",
                    ", ": r",\quad ", " ± ": r" \pm "}


def _split_top_level(text: str) -> list[str]:
    parts, depth, i, current = [], 0, 0, ""
    while i < len(text):
        ch = text[i]
        depth += ch in "([{"
        depth -= ch in ")]}"
        if depth == 0:
            sep = next((s for s in _SEPARATORS if text.startswith(s, i)), None)
            if sep:
                parts += [current, sep]
                current, i = "", i + len(sep)
                continue
        current += ch
        i += 1
    parts.append(current)
    return parts


def _text(s: str) -> str:
    if not s.strip():
        return r"\ " if s else ""
    for old, new in (("\\", ""), ("{", "("), ("}", ")"), ("_", " "), ("^", "^"), ("#", ""), ("$", ""),
                     ("&", "and"), ("%", " percent"), ("~", "")):
        s = s.replace(old, new)
    return rf"\text{{ {s.strip()} }}"


def _light(s: str) -> str:
    """Typeset without re-parsing: keeps every bracket, as in 4(1)(5), and copes with notation SymPy can't read."""
    out = unicode_math(s).replace("−", "-")
    out = re.sub(r"\bsqrt\(([^()]*)\)", r"√(\1)", out)
    for old, new in (("λ", r"\lambda "), ("θ", r"\theta "), ("°", r"^\circ "), ("×", r" \times "), ("·", r" \cdot "), ("÷", r" \div "), ("±", r" \pm "),
                     ("∞", r"\infty "), ("∫", r"\int "), ("→", r"\to "), ("≈", r"\approx "), ("…", r"\dots "),
                     ("≤", r"\le "), ("≥", r"\ge "), ("|", r"\vert ")):
        out = out.replace(old, new)
    out = re.sub(r"√\(([^()]*)\)", r"\\sqrt{\1}", out)
    out = re.sub(r"√(\w+)", r"\\sqrt{\1}", out)
    out = re.sub(r"\^\(([^()]*)\)", r"^{\1}", out)
    out = re.sub(r"\^(-?\w+)", r"^{\1}", out)
    out = re.sub(r"(?<![\w)}])(-?)(\d+)/(\d+)(?![\w(])", r"\1\\frac{\2}{\3}", out)  # -7/10 -> -(7/10)
    out = re.sub(r"(?<!\\)\b(sin|cos|tan|ln|log|exp|det|lim)\b", r"\\\1", out)
    return re.sub(r"(?<!\\)\b([A-Za-z]{3,})\b", lambda m: m.group(0) if m.group(1) in _FUNCTIONS
                  or m.group(1) in ("frac", "sqrt", "theta", "lambda", "times", "cdot", "div", "infty", "int", "dots",
                                    "approx", "circ", "vert", "text") else rf"\text{{ {m.group(1)} }}", out)


def _sympy(s: str) -> str | None:
    """Typeset via SymPy; None if it can't be parsed safely."""
    s = s.strip()
    if not s or re.search(r"(?<![A-Za-z])\(-?\d+\)", s):  # a number in brackets, as in 4(1)(5): a substitution
        return None
    if re.fullmatch(r"-?\d+/\d+", s) or re.search(r"\bd[a-z]\b", s):  # keep 1/1 as a fraction, du/dx intact
        return None
    s = (unicode_math(s).replace("·", "*").replace("×", "*").replace("÷", "/").replace("λ", "L")
         .replace("−", "-").replace("∞", "oo"))
    s = re.sub(r"√\(", "sqrt(", s)
    s = re.sub(r"√([0-9a-z]+)", r"sqrt(\1)", s)
    try:
        s = normalize(s)
        _check_safe(s)
        expr = _parse(s, evaluate=False, extra=_EXTRA)
    except Exception:
        return None
    latex = sp.latex(expr, order="none", ln_notation=True, imaginary_unit="i")
    return re.sub(r"(?<![\d.])1 \\cdot ", "", latex)


def _math(s: str) -> str:
    stripped = s.strip()
    if not stripped:
        return ""
    return _sympy(stripped) or _light(stripped)


def _is_word(token: str) -> bool:
    w = token.strip(":,.")
    if not w.isalpha():
        return False
    return (len(w) >= 3 and w.lower() not in _FUNCTIONS) or w.lower() in _SHORT_WORDS


def _mixed(piece: str) -> str:
    """Alternate runs of words (as text) and runs of math (as math)."""
    out, run, run_is_words = [], [], None
    for token in piece.split(" "):
        is_word = _is_word(token)
        if run and is_word != run_is_words:
            joined = " ".join(run)
            out.append(_text(joined + " ") if run_is_words else _math(joined))
            run = []
        run.append(token)
        run_is_words = is_word
    if run:
        joined = " ".join(run)
        out.append(_text(joined) if run_is_words else _math(joined))
    return " ".join(x for x in out if x)


def _matrix(text: str) -> str | None:
    if not re.fullmatch(r"\s*\[\s*\[.*\]\s*\]\s*", text):
        return None
    rows = re.findall(r"\[([^\[\]]*)\]", text)
    cells = [[_math(c).replace(r"\frac", r"\dfrac") for c in row.split(",")] for row in rows]
    # Full-size fractions and roomier rows, so entries like 3/5 stay readable.
    gap = r" \\[10pt] " if any(r"\dfrac" in c for r in cells for c in r) else r" \\ "
    return r"\begin{bmatrix}" + gap.join(" & ".join(r) for r in cells) + r"\end{bmatrix}"


def _piece(text: str) -> str:
    s = text.strip()
    if not s:
        return ""
    if m := re.fullmatch(r"(.+?)\s*\(about ([^)]+)\)", s):  # "7/8 (about 0.875)"
        return rf"{_piece(m.group(1))}\;\left(\approx {_light(m.group(2))}\right)"
    if m := re.fullmatch(r"(.*?)\s*\+\s*(C|…)", s):  # "x^3/3 + C", "1 + x + … "
        return f"{_piece(m.group(1))} + {'C' if m.group(2) == 'C' else r'\dots'}"
    if (matrix := _matrix(s)) is not None:
        return matrix
    if m := re.fullmatch(r"d/d([a-z])\[(.+)\]", s):  # d/dx[x^2]
        return rf"\frac{{d}}{{d{m.group(1)}}}\left[{_math(m.group(2))}\right]"
    if m := re.fullmatch(r"(-?\d+ · )?∫\s*(.+?)\s*d([a-z])", s):  # ∫x^2 dx, 3 · ∫x^2 dx
        coeff = _light(m.group(1).replace(" · ", "")) + r" \cdot " if m.group(1) else ""
        return rf"{coeff}\int {_math(m.group(2))}\, d{m.group(3)}"
    if m := re.fullmatch(r"\|(.+)\|", s):  # |3 + 4i|
        return rf"\left|{_math(m.group(1))}\right|"
    if m := re.fullmatch(r"det\((.+)\)", s):
        return rf"\det\left({_light(m.group(1))}\right)"
    if re.fullmatch(r"[fy]'{1,3}\(\w+\)|[fy]\^\(\d+\)\(\w+\)|\w+!", s):  # f'(x), f^(4)(0), 5!
        return _light(s)
    if re.fullmatch(r"\d+(?:\s*×\s*\d+)+", s):  # 5 × 4 × 3 × 2 × 1: keep every factor
        return _light(s)
    return _mixed(s)


def split_description(math_text: str) -> tuple[str, str]:
    """"raise the power by 1 and divide by the new power: ∫x^2 dx = x^3/3" -> (words, math).

    On paper the explanation goes on its own line and the math sits cleanly underneath."""
    before, sep, after = math_text.rpartition(": ")
    if sep and after.strip() and "=" not in before and sum(_is_word(w) for w in before.split()) >= 2:
        return before.strip(), after.strip()
    return "", math_text


def is_prose(math_text: str) -> bool:
    """A step that's really a sentence ("Multiply top and bottom by the conjugate ...") stays as text."""
    words = [w for w in math_text.split() if _is_word(w)]
    return len(words) >= 4 and " = " not in math_text and "∫" not in math_text


def to_latex(math_text: str) -> str:
    """Best-effort LaTeX for one step's math. Never raises."""
    try:
        parts = []
        for segment in re.split(r"(?<=\w|\)|'): (?=\S)", math_text.strip()):  # "by 1: d/dx[x^2]" -> two segments
            pieces = _split_top_level(segment)
            parts.append("".join(_SEPARATOR_LATEX[p] if p in _SEPARATOR_LATEX else _piece(p) for p in pieces))
        return r"\colon\ ".join(parts)
    except Exception:
        return _text(math_text)
