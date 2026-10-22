"""Score one tutor run against a case's expectation, by mathematical meaning rather than exact text.

Outcomes:
  correct   solved, and the answer matches
  wrong     solved, but the answer is wrong (or answered when it should have declined) - the worst outcome
  declined  didn't give an answer (lesson or ask-for-context); fine only when allowed
  error     the run crashed
"""

from __future__ import annotations

import re

import sympy as sp

from tutor.math_tool import _parse, normalize

X = sp.Symbol("x")
_NAMES = {"I": sp.I, "pi": sp.pi, "E": sp.E}


def _expr(text: str) -> sp.Expr:
    """Parse an answer fragment the app produced, or an expected value from cases.json."""
    s = text.strip()
    s = re.sub(r"\s*\(about [^)]*\)", "", s)
    s = re.sub(r"\s*\((C\d(, C\d)*) (is a constant|are constants)[^)]*\)", "", s)
    s = re.sub(r"\s*\+\s*(C|…)\s*$", "", s)
    s = s.replace("·", "*").replace("−", "-")
    return _parse(normalize(s))  # app answers use the app's own notation (2x, e^(2x), ln); expected values use sympify


def _top_level_split(text: str, sep: str = ",") -> list[str]:
    parts, depth, current = [], 0, ""
    for ch in text:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == sep and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    return [p for p in parts + [current] if p.strip()]


def answer_values(answer: str) -> list[sp.Expr]:
    """'x = 2, x = 3' -> [2, 3]; "f'(x) = 2x" -> [2x]; 'λ = 3, λ = 1' -> [3, 1]; '7/8 (about 0.875)' -> [7/8]."""
    values = []
    for part in _top_level_split(answer.replace(";", ",")):
        rhs = part.split("=")[-1] if "=" in part else part
        rhs = re.sub(r"^\s*\w+\s*:\s*", "", rhs)  # "λ = 1: [-1, 1]" style labels
        values.append(_expr(rhs))
    return values


def _equal(a: sp.Expr, b: sp.Expr) -> bool:
    try:
        if sp.simplify(sp.expand(a - b)) == 0:
            return True
        return bool(abs(complex(sp.N(a - b))) < 1e-9)
    except Exception:
        return False


def _same_multiset(got: list[sp.Expr], expected: list[sp.Expr]) -> bool:
    remaining = list(expected)
    for g in got:
        match = next((e for e in remaining if _equal(g, e)), None)
        if match is None:
            return False
        remaining.remove(match)
    return not remaining


def check_answer(answer: str, expect: dict) -> bool:
    try:
        if "text" in expect:
            squash = lambda s: re.sub(r"\s+", "", s.lower().replace("**", "^"))
            return any(squash(t) in squash(answer) for t in expect["text"])
        if "matrix" in expect:
            rows = re.findall(r"\[([^\[\]]*)\]", answer)
            got = sp.Matrix([[_expr(c) for c in row.split(",")] for row in rows])
            return got.equals(sp.Matrix([[sp.sympify(c) for c in row] for row in expect["matrix"]]))
        if "antiderivative_of" in expect:
            got = answer_values(answer)[-1]
            integrand = sp.sympify(expect["antiderivative_of"], locals=_NAMES)
            return _equal(sp.diff(got, X), integrand)
        if "values" in expect:
            expected = [sp.sympify(v, locals=_NAMES) for v in expect["values"]]
            return _same_multiset(answer_values(answer), expected)
    except Exception:
        return False
    return False


def score(state: dict | None, expect: dict, error: str | None = None) -> str:
    if error:
        return "error"
    solved = bool(state) and state.get("status") == "solved" and state.get("math", {}).get("answer")
    if expect.get("decline"):
        return "wrong" if solved else "correct"
    if not solved:
        return "declined"
    return "correct" if check_answer(state["math"]["answer"], expect) else "wrong"
