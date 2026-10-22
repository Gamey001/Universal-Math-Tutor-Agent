"""Are two answers the same, mathematically? Used to verify cloud answers and to score the evaluation.

Answers from the math tool look like "x = 2, x = 3", "7/8 (about 0.875)", "x·e^(x) - e^(x) + C",
"[[3/5, -7/10], [-1/5, 2/5]]"; reference values are SymPy syntax ("2*p + 2*q", "exp(2)").
"""

from __future__ import annotations

import re

import sympy as sp

from .math_tool import _parse, normalize

_NAMES = {"I": sp.I, "pi": sp.pi, "E": sp.E, "e": sp.E, "oo": sp.oo}


def parse_app_value(text: str) -> sp.Expr:
    """One value in the app's own notation (2x, e^(2x), ln, i)."""
    s = text.strip()
    s = re.sub(r"\s*\(about [^)]*\)", "", s)
    s = re.sub(r"\s*\((C\d(, C\d)*) (is a constant|are constants)[^)]*\)", "", s)
    s = re.sub(r"\s*\+\s*(C|…)\s*$", "", s)
    return _parse(normalize(s.replace("·", "*").replace("−", "-")))


def parse_reference(text: str) -> sp.Expr:
    """A value in SymPy syntax, falling back to the app's notation."""
    try:
        return sp.sympify(text, locals=_NAMES)
    except Exception:
        return parse_app_value(text)


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
    """'x = 2, x = 3' -> [2, 3]; "f'(x) = 2x" -> [2x]; 'λ = 3, λ = 1' -> [3, 1]."""
    values = []
    for part in _top_level_split(answer.replace(";", ",")):
        rhs = part.split("=")[-1] if "=" in part else part
        rhs = re.sub(r"^\s*\w+\s*:\s*", "", rhs)
        values.append(parse_app_value(rhs))
    return values


def equal(a: sp.Expr, b: sp.Expr, up_to_constant: bool = False) -> bool:
    try:
        diff = sp.simplify(sp.expand(a - b))
        if diff == 0:
            return True
        if up_to_constant and not diff.free_symbols - {sp.Symbol("C")}:
            return True  # antiderivatives that differ by a constant are both right
        return bool(abs(complex(sp.N(diff))) < 1e-9)
    except Exception:
        return False


def same_multiset(got: list[sp.Expr], expected: list[sp.Expr], up_to_constant: bool = False) -> bool:
    remaining = list(expected)
    for g in got:
        match = next((e for e in remaining if equal(g, e, up_to_constant)), None)
        if match is None:
            return False
        remaining.remove(match)
    return not remaining


def answers_agree(app_answer: str, reference: str) -> bool:
    """Does an answer from the math tool match a reference value (or values, comma-separated)?"""
    try:
        got = answer_values(app_answer)
        expected = [parse_reference(r) for r in _top_level_split(reference.replace(";", ","))]
        indefinite = app_answer.rstrip().endswith("+ C")
        if len(got) == len(expected) and same_multiset(got, expected, up_to_constant=indefinite):
            return True
        # "x = 6, y = 4" also agrees with a reference that names only the asked-for value, e.g. "6".
        return len(expected) == 1 and any(equal(g, expected[0], indefinite) for g in got) and len(got) <= 3
    except Exception:
        return False
