"""Matrices: determinant, inverse, transpose, products, powers, eigenvalues, rank, row reduction, Ax = b.

Matrices are written row by row: [[1, 2], [3, 4]]. Each entry goes through the
same safe scalar parser as the rest of the math tool; the operation itself is
matched against a fixed list of patterns, so nothing here is ever eval'd.
One operation per problem keeps the steps short enough to follow.
"""

from __future__ import annotations

import re

import sympy as sp

from .math_tool import MathInputError, MathResult, _check_safe, _fmt, _parse, normalize

MAX_SIZE = 6
_LAMBDA = sp.Symbol("λ")

_MATRIX = re.compile(r"\[\s*\[[^\[\]]*\](?:\s*,\s*\[[^\[\]]*\])*\s*\]")
_VECTOR = re.compile(r"\[[^\[\]]*\]")

_OPERATION_WORDS = {
    "determinant": "det", "det": "det",
    "inverse": "inverse", "inv": "inverse",
    "transpose": "transpose",
    "eigenvalues": "eigenvalues", "eigenvalue": "eigenvalues",
    "eigenvectors": "eigenvectors", "eigenvector": "eigenvectors",
    "rank": "rank",
    "rref": "rref", "row reduce": "rref", "reduced row echelon form": "rref", "row echelon form": "rref",
    "trace": "trace",
}


def _entry(text: str) -> sp.Expr:
    s = normalize(text)
    _check_safe(s)
    value = sp.nsimplify(_parse(s)) if "." in s else _parse(s)
    if value.free_symbols:
        raise MathInputError("Matrix entries need to be numbers.")
    return value


def _parse_matrix(literal: str) -> sp.Matrix:
    rows = re.findall(r"\[([^\[\]]*)\]", literal)
    matrix = [[_entry(e) for e in row.split(",") if e.strip()] for row in rows]
    if not matrix or not matrix[0] or any(len(r) != len(matrix[0]) for r in matrix):
        raise MathInputError("Every row of a matrix needs the same number of entries.")
    if len(matrix) > MAX_SIZE or len(matrix[0]) > MAX_SIZE:
        raise MathInputError(f"I can check matrices up to {MAX_SIZE} × {MAX_SIZE}.")
    return sp.Matrix(matrix)


def _parse_vector(literal: str) -> sp.Matrix:
    entries = [_entry(e) for e in literal.strip("[] ").split(",") if e.strip()]
    if not entries or len(entries) > MAX_SIZE:
        raise MathInputError("I couldn't read that vector.")
    return sp.Matrix(entries)  # a column vector


def show(m: sp.Matrix | sp.Expr) -> str:
    """[[1, 2], [3, 4]], the same notation learners type."""
    if isinstance(m, sp.MatrixBase):
        return "[" + ", ".join("[" + ", ".join(sp.sstr(v) for v in m.row(i)) + "]" for i in range(m.rows)) + "]"
    return _fmt(m)


def _p(value: sp.Expr) -> str:
    """Negative numbers in brackets: "+ 1 × (-2)", never "+ 1 × -2"."""
    text = sp.sstr(value)
    return f"({text})" if text.startswith("-") else text


def _poly(expr: sp.Expr) -> str:
    return re.sub(r"(\d)\*λ", r"\1λ", sp.sstr(expr)).replace("**", "^")


def _shape(m: sp.Matrix) -> str:
    return f"{m.rows} × {m.cols}"


def _result(problem: str, problem_latex: str, answer, steps: list[str], verification: str) -> MathResult:
    answer_text = answer if isinstance(answer, str) else show(answer)
    answer_latex = answer_text if isinstance(answer, str) else sp.latex(answer)
    return MathResult(ok=True, kind="matrix", problem=problem, problem_latex=problem_latex,
                      answer=answer_text, answer_latex=answer_latex, steps=steps, verification=verification)


# ---- operations ------------------------------------------------------------

def _det_steps(m: sp.Matrix) -> list[str]:
    if m.shape == (2, 2):
        a, b, c, d = m
        return [f"Determinant of a 2 × 2 matrix: det = ad - bc = ({sp.sstr(a)})({sp.sstr(d)}) - ({sp.sstr(b)})({sp.sstr(c)})"
                f" = {sp.sstr(a * d)} - {sp.sstr(b * c)} = {sp.sstr(m.det())}"]
    if m.shape == (3, 3):
        steps = ["Expand along the first row: each entry times the determinant of what's left when you "
                 "cover its row and column, with signs + - +"]
        total_parts = []
        for j in range(3):
            minor = m.minor_submatrix(0, j)
            sign = "+" if j % 2 == 0 else "-"
            md = minor.det()
            steps.append(f"{sign} {_p(m[0, j])} × det({show(minor)}) = {sign} {_p(m[0, j])} × {_p(md)}"
                         f" = {sign} {_p(m[0, j] * md)}")
            total_parts.append((-1) ** j * m[0, j] * md)
        steps.append(f"Add them up: det = {sp.sstr(sum(total_parts))}")
        return steps
    return [f"Row-reduce to a triangular matrix and multiply the diagonal: det = {sp.sstr(m.det())}"]


def _det(m: sp.Matrix, label: str) -> MathResult:
    if not m.is_square:
        raise MathInputError("Only square matrices have a determinant.")
    det = m.det()
    return _result(f"det({label})", rf"\det {sp.latex(m)}", det, _det_steps(m),
                   f"SymPy computed det = {sp.sstr(det)} independently.")


def _inverse(m: sp.Matrix, label: str) -> MathResult:
    if not m.is_square:
        raise MathInputError("Only square matrices can have an inverse.")
    det = m.det()
    steps = _det_steps(m)
    if det == 0:
        steps.append("The determinant is 0, so this matrix squashes space flat and can't be undone")
        return _result(f"inverse({label})", rf"{sp.latex(m)}^{{-1}}", "no inverse (the determinant is 0)", steps,
                       "SymPy confirms the determinant is 0.")
    inv = m.inv()
    if m.shape == (2, 2):
        a, b, c, d = m
        swapped = sp.Matrix([[d, -b], [-c, a]])
        steps.append(f"Swap a and d, and change the signs of b and c: {show(swapped)}")
        steps.append(f"Divide every entry by the determinant, {sp.sstr(det)}: {show(inv)}")
    else:
        steps.append(f"Inverse = (adjugate matrix) ÷ determinant = {show(inv)}")
    identity_ok = (m * inv) == sp.eye(m.rows)
    return _result(f"inverse({label})", rf"{sp.latex(m)}^{{-1}}", inv, steps,
                   "Multiplying the matrix by this inverse gives the identity matrix"
                   + (" ✔" if identity_ok else " ✘"))


def _transpose(m: sp.Matrix, label: str) -> MathResult:
    t = m.T
    return _result(f"transpose({label})", rf"{sp.latex(m)}^{{T}}", t,
                   [f"Rows become columns: row 1 {show(m.row(0))[1:-1]} becomes column 1, and so on → {show(t)}"],
                   f"A {_shape(m)} matrix becomes {_shape(t)}.")


def _product(a: sp.Matrix, b: sp.Matrix, la: str, lb: str) -> MathResult:
    if a.cols != b.rows:
        raise MathInputError(f"A {_shape(a)} matrix can't multiply a {_shape(b)} one: "
                             "the first one's columns must match the second one's rows.")
    p = a * b
    steps = [f"A {_shape(a)} times a {_shape(b)} gives a {_shape(p)} result. "
             "Each entry is a row of the first times a column of the second"]
    if p.rows * p.cols <= 9:
        for i in range(p.rows):
            for j in range(p.cols):
                terms = " + ".join(f"({sp.sstr(a[i, k])})({sp.sstr(b[k, j])})" for k in range(a.cols))
                steps.append(f"Row {i + 1} · column {j + 1}: {terms} = {sp.sstr(p[i, j])}")
    return _result(f"{la} × {lb}", rf"{sp.latex(a)} {sp.latex(b)}", p, steps, "Every entry recomputed by SymPy.")


def _sum(a: sp.Matrix, b: sp.Matrix, la: str, lb: str, sign: int) -> MathResult:
    if a.shape != b.shape:
        raise MathInputError("Matrices must be the same size to add or subtract them.")
    r = a + sign * b
    word, op = ("Add", "+") if sign == 1 else ("Subtract", "-")
    return _result(f"{la} {op} {lb}", rf"{sp.latex(a)} {op} {sp.latex(b)}", r,
                   [f"{word} matching entries, position by position → {show(r)}"], "Every entry recomputed by SymPy.")


def _scalar(k: sp.Expr, m: sp.Matrix, label: str) -> MathResult:
    r = k * m
    return _result(f"{sp.sstr(k)} × {label}", rf"{sp.latex(k)} {sp.latex(m)}", r,
                   [f"Multiply every entry by {sp.sstr(k)} → {show(r)}"], "Every entry recomputed by SymPy.")


def _power(m: sp.Matrix, n: int, label: str) -> MathResult:
    if not m.is_square:
        raise MathInputError("Only square matrices can be raised to a power.")
    if not 1 <= n <= 10:
        raise MathInputError("I can do matrix powers from 1 to 10.")
    steps, acc = [f"Call the matrix A = {label}"], m
    for k in range(2, n + 1):
        acc = acc * m
        steps.append(f"A^{k} = A^{k - 1} × A = {show(acc)}" if k > 2 else f"A^2 = A × A = {show(acc)}")
    return _result(f"{label}^{n}", rf"{sp.latex(m)}^{{{n}}}", acc, steps, f"SymPy computed {label}^{n} directly too.")


def _char_poly_steps(m: sp.Matrix) -> tuple[sp.Expr, list[str]]:
    poly = sp.factor((m - _LAMBDA * sp.eye(m.rows)).det())
    return poly, [f"Subtract λ from every diagonal entry and take the determinant: det(A - λI) = "
                  f"{_poly(sp.expand(poly))}",
                  f"Factor it: {_poly(poly)} = 0"]


def _eigenvalues(m: sp.Matrix, label: str) -> MathResult:
    if not m.is_square:
        raise MathInputError("Only square matrices have eigenvalues.")
    poly, steps = _char_poly_steps(m)
    values = m.eigenvals()
    shown = ", ".join(f"λ = {_fmt(v)}" + (f" (twice)" if k == 2 else f" ({k} times)" if k > 2 else "")
                      for v, k in values.items())
    steps.append(f"The values of λ that make it 0 are the eigenvalues: {shown}")
    checks = all(sp.simplify((m - v * sp.eye(m.rows)).det()) == 0 for v in values)
    return _result(f"eigenvalues({label})", rf"\text{{eigenvalues of }} {sp.latex(m)}", shown, steps,
                   "For each eigenvalue, det(A - λI) is exactly 0" + (" ✔" if checks else " ✘"))


def _eigenvectors(m: sp.Matrix, label: str) -> MathResult:
    if not m.is_square:
        raise MathInputError("Only square matrices have eigenvectors.")
    poly, steps = _char_poly_steps(m)
    parts, checks = [], True
    for value, _, vectors in m.eigenvects():
        for v in vectors:
            v = v / sp.gcd(list(v)) if all(x.is_Integer for x in v) and any(v) else v
            parts.append(f"λ = {_fmt(value)}: {show(v.T)[1:-1]}")
            steps.append(f"For λ = {_fmt(value)}, solve (A - λI)v = 0 → v = {show(v.T)[1:-1]}")
            checks &= sp.simplify(m * v - value * v) == sp.zeros(m.rows, 1)
    return _result(f"eigenvectors({label})", rf"\text{{eigenvectors of }} {sp.latex(m)}", "; ".join(parts), steps,
                   "Each one satisfies A·v = λ·v" + (" ✔" if checks else " ✘"))


def _rref(m: sp.Matrix, label: str, rank_only: bool = False) -> MathResult:
    reduced, pivots = m.rref()
    steps = [f"Row-reduce (swap rows, scale rows, subtract multiples of rows) until each leading entry is 1 "
             f"with zeros above and below → {show(reduced)}",
             f"Leading 1s (pivots) sit in column{'s' if len(pivots) != 1 else ''} "
             f"{', '.join(str(p + 1) for p in pivots) or '(none)'}"]
    if rank_only:
        steps.append(f"The rank is the number of pivots: {len(pivots)}")
        return _result(f"rank({label})", rf"\operatorname{{rank}} {sp.latex(m)}", sp.Integer(len(pivots)), steps,
                       f"SymPy's rank() also gives {m.rank()}.")
    return _result(f"rref({label})", rf"\operatorname{{rref}} {sp.latex(m)}", reduced, steps,
                   "Row-reduced by SymPy.")


def _trace(m: sp.Matrix, label: str) -> MathResult:
    if not m.is_square:
        raise MathInputError("Only square matrices have a trace.")
    diag = [m[i, i] for i in range(m.rows)]
    t = sum(diag)
    return _result(f"trace({label})", rf"\operatorname{{tr}} {sp.latex(m)}", t,
                   [f"Add the diagonal entries: {' + '.join(f'({sp.sstr(d)})' for d in diag)} = {sp.sstr(t)}"],
                   "Recomputed by SymPy.")


def _solve_linear(a: sp.Matrix, b: sp.Matrix, la: str, lb: str) -> MathResult:
    if a.rows != b.rows:
        raise MathInputError("The vector needs one entry per row of the matrix.")
    names = sp.symbols(f"x1:{a.cols + 1}")
    steps = []
    if a.is_square and a.det() != 0:
        steps += _det_steps(a)
        steps.append("The determinant isn't 0, so there is exactly one solution: x = A⁻¹b")
        x = a.inv() * b
        steps.append(f"A⁻¹ = {show(a.inv())}, and A⁻¹b = {show(x.T)[1:-1]}")
    else:
        solutions = sp.linsolve((a, b), names)
        if not solutions:
            return _result(f"solve {la}·x = {lb}", rf"{sp.latex(a)}\,x = {sp.latex(b)}",
                           "no solution (the equations contradict each other)",
                           ["Row-reducing [A | b] gives a row of 0s equal to something non-zero"],
                           "SymPy's linsolve finds no solution.")
        x = sp.Matrix(list(solutions)[0])
        if x.free_symbols:
            return _result(f"solve {la}·x = {lb}", rf"{sp.latex(a)}\,x = {sp.latex(b)}",
                           "infinitely many solutions: " + ", ".join(f"{n} = {sp.sstr(v)}" for n, v in zip(names, x)),
                           ["Row-reducing [A | b] leaves free variables, so there are infinitely many solutions"],
                           "SymPy's linsolve gives the general solution.")
    answer = ", ".join(f"{n} = {_fmt(v)}" for n, v in zip(names, x))
    ok = a * x == b
    return _result(f"solve {la}·x = {lb}", rf"{sp.latex(a)}\,x = {sp.latex(b)}", answer, steps,
                   "Multiplying A by this x gives b" + (" ✔" if ok else " ✘"))


_UNARY = {"det": _det, "inverse": _inverse, "transpose": _transpose, "eigenvalues": _eigenvalues,
          "eigenvectors": _eigenvectors, "rref": _rref, "trace": _trace,
          "rank": lambda m, label: _rref(m, label, rank_only=True)}


def solve_matrix_problem(text: str) -> MathResult:
    """Called by math_tool.solve_problem whenever the input contains '['. Raises MathInputError."""
    s = text.strip()
    s = re.sub(r"^(what is|find|compute|calculate|work out)\b[:\s]*", "", s, flags=re.IGNORECASE).rstrip("?. ")
    if len(s) > 400:
        raise MathInputError("That problem is too long for me to check.")

    # Swap each matrix/vector literal for a letter, remembering what it was.
    values: dict[str, sp.Matrix] = {}
    shown: dict[str, str] = {}

    def take(match: re.Match, parse) -> str:
        name = "ABCDEFGH"[len(values)]
        values[name] = parse(match.group(0))
        shown[name] = show(values[name])
        return f" {name} "

    s = _MATRIX.sub(lambda m: take(m, _parse_matrix), s)
    s = _VECTOR.sub(lambda m: take(m, _parse_vector), s)
    if not values:
        raise MathInputError("I couldn't find a matrix written like [[1, 2], [3, 4]].")

    # Words and notation -> one canonical operation.
    s = re.sub(r"\s+", " ", s).strip()
    for word in sorted(_OPERATION_WORDS, key=len, reverse=True):
        s = re.sub(rf"\b(?:the\s+)?{word}\b\s*(?:of\s*)?", f"{_OPERATION_WORDS[word]} ", s, flags=re.IGNORECASE)
    s = re.sub(r"([A-H])\s*\^\s*\(?\s*-1\s*\)?", r"inverse \1", s)
    s = re.sub(r"([A-H])\s*\^\s*T\b", r"transpose \1", s)
    s = re.sub(r"[()]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    def label(name: str) -> str:
        return shown[name]

    if m := re.fullmatch(r"(det|inverse|transpose|eigenvalues|eigenvectors|rank|rref|trace) ([A-H])", s):
        op, name = m.groups()
        return _UNARY[op](values[name], label(name))
    if m := re.fullmatch(r"(?:solve )?([A-H]) ?\*? ?x ?= ?([A-H])", s):
        a, b = m.groups()
        return _solve_linear(values[a], values[b], label(a), label(b))
    if m := re.fullmatch(r"([A-H]) ?([*×+\-]|) ?([A-H])", s):
        a, op, b = m.groups()
        if op in ("", "*", "×"):
            return _product(values[a], values[b], label(a), label(b))
        return _sum(values[a], values[b], label(a), label(b), 1 if op == "+" else -1)
    if m := re.fullmatch(r"([A-H]) ?\^ ?(\d+)", s):
        name, n = m.groups()
        return _power(values[name], int(n), label(name))
    if m := re.fullmatch(r"(-?[\d/.]+) ?\*? ?([A-H])", s):
        k, name = m.groups()
        return _scalar(_entry(k), values[name], label(name))
    if re.fullmatch(r"[A-H]", s):
        raise MathInputError("What would you like to do with this matrix? For example: "
                             "det, inverse, transpose, eigenvalues, rank, or multiply it by another matrix.")
    raise MathInputError("I can do one matrix operation at a time, like det [[1, 2], [3, 4]] "
                         "or [[1, 2], [3, 4]] * [[5, 6], [7, 8]].")
