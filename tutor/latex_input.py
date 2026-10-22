"""LaTeX from the math keyboard -> the plain notation the solvers understand.

The keyboard (MathLive) produces LaTeX such as \\int_{0}^{1}\\frac{4x}{(1+x)^{3}}\\,dx. The solvers read
"integral from 0 to 1 of (4x)/((1+x)^(3)) dx". Converting (rather than handing the LaTeX to SymPy's own
parser, which evaluates immediately) keeps every input on the same path as typed questions, with the
same worked steps and checks.
"""

from __future__ import annotations

import re

GREEK = ["alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "iota",
         "kappa", "mu", "nu", "xi", "rho", "sigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega"]
_FUNCTIONS = ["arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "sin", "cos", "tan", "sec", "csc", "cot", "ln",
              "log", "exp", "det", "max", "min", "gcd", "lcm"]
_SIMPLE = [
    (r"\left", ""), (r"\right", ""), (r"\displaystyle", ""), (r"\limits", ""), (r"\,", " "), (r"\;", " "),
    (r"\:", " "), (r"\!", ""), (r"\quad", " "), (r"\qquad", " "), ("~", " "),
    (r"\cdot", "*"), (r"\times", "*"), (r"\div", "/"), (r"\leq", "<="), (r"\le", "<="), (r"\geq", ">="),
    (r"\ge", ">="), (r"\neq", "!="), (r"\ne", "!="), (r"\infty", "oo"), (r"\pi", "pi"), (r"\lambda", "lamda"),
    (r"\to", "->"), (r"\rightarrow", "->"), (r"\mathrm{d}", "d"), (r"\%", "%"), (r"\lvert", "|"), (r"\rvert", "|"),
    (r"\vert", "|"), (r"\mid", "|"), ("^\\circ", "*pi/180"), (r"\degree", "*pi/180"),
]


class LatexInputError(ValueError):
    pass


def _group(s: str, i: int) -> tuple[str, int]:
    """The {...} group (or single token) starting at s[i]; returns (content, index after it)."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        raise LatexInputError("Something is missing at the end.")
    if s[i] != "{":
        if s[i] == "\\":  # a command like \pi
            m = re.match(r"\\[A-Za-z]+", s[i:])
            return m.group(0), i + len(m.group(0))
        return s[i], i + 1
    depth, j = 0, i
    while j < len(s):
        depth += s[j] == "{"
        depth -= s[j] == "}"
        if depth == 0:
            return s[i + 1:j], j + 1
        j += 1
    raise LatexInputError("A bracket { isn't closed.")


def _convert(s: str) -> str:
    """Structural commands -> plain notation (recursive)."""
    out, i = [], 0
    while i < len(s):
        if s.startswith(r"\frac", i) or s.startswith(r"\dfrac", i) or s.startswith(r"\tfrac", i):
            i = s.index("frac", i) + 4
            num, i = _group(s, i)
            den, i = _group(s, i)
            out.append(f"(({_convert(num)})/({_convert(den)}))")
        elif s.startswith(r"\sqrt", i):
            i += 5
            index = None
            if i < len(s) and s[i] == "[":
                close = s.index("]", i)
                index, i = s[i + 1:close], close + 1
            body, i = _group(s, i)
            out.append(f"(({_convert(body)})^(1/({_convert(index)})))" if index else f"sqrt({_convert(body)})")
        elif s.startswith(r"\binom", i):
            i += 6
            n, i = _group(s, i)
            k, i = _group(s, i)
            out.append(f"binomial({_convert(n)}, {_convert(k)})")
        elif s.startswith(r"\begin{", i):
            m = re.match(r"\\begin\{([pbvB]?matrix)\}(.*?)\\end\{\1\}", s[i:], re.DOTALL)
            if not m:
                raise LatexInputError("I couldn't read that matrix.")
            rows = [r for r in m.group(2).split(r"\\") if r.strip()]
            cells = [[_convert(c).strip() for c in row.split("&")] for row in rows]
            matrix = "[" + ", ".join("[" + ", ".join(r) + "]" for r in cells) + "]"
            out.append(("det " if m.group(1) == "vmatrix" else "") + matrix)
            i += len(m.group(0))
        elif s.startswith(r"\logbase", i):
            base, i = _group(s, i + 8)
            arg_start = i
            depth = 0
            while i < len(s):  # the (...) argument that follows
                depth += s[i] == "("
                depth -= s[i] == ")"
                i += 1
                if depth == 0:
                    break
            out.append(f"log({_convert(s[arg_start + 1:i - 1])}, {_convert(base)})")
        elif s.startswith("^", i):
            body, i = _group(s, i + 1)
            out.append(f"^({_convert(body)})")
        elif s[i] == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            name = m.group(1) if m else ""
            if name in _FUNCTIONS:
                out.append(f" {name}")
            elif name in GREEK:
                out.append(f" {name.removeprefix('var')} ")
            elif name in ("operatorname", "mathrm", "text", "mathit"):
                body, j = _group(s, i + len(name) + 1)
                out.append(f" {body} ")
                i = j
                continue
            elif name:
                raise LatexInputError(f"I don't know the symbol \\{name} yet.")
            i += len(name) + 1
        elif s[i] in "{}":
            out.append("(" if s[i] == "{" else ")")
            i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _abs_bars(s: str) -> str:
    """|x - 1| -> abs(x - 1), pairing bars left to right."""
    parts = s.split("|")
    if len(parts) % 2 == 0:
        raise LatexInputError("An absolute value bar | isn't closed.")
    return "".join(p if k % 2 == 0 else f"abs({p})" for k, p in enumerate(parts))


def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    # cos^2 x -> cos(x)^2: the power belongs to the function's value, as in textbooks.
    s = re.sub(r"\b(sin|cos|tan|sec|csc|cot|sinh|cosh|tanh)\^\(?(\w+)\)?\s*(\([^()]*\)|[A-Za-z0-9]+)",
               r"\1(\3)^\2", s)
    s = re.sub(r"\(\(([A-Za-z0-9.]+)\)/\(([A-Za-z0-9.]+)\)\)", r"(\1/\2)", s)  # ((3)/(4)) -> (3/4)
    s = re.sub(r"\^\((-?[A-Za-z0-9.]+)\)", r"^\1", s)  # ^(2) -> ^2
    return s


def latex_to_problem(latex: str) -> str:
    """Convert keyboard LaTeX into a problem the solvers can read. Raises LatexInputError."""
    s = (latex or "").strip()
    if not s:
        raise LatexInputError("There's nothing to solve yet.")
    for old, new in _SIMPLE:
        s = s.replace(old, new)
    s = re.sub(r"\\log_\s*\{?([^{}\s]+)\}?\s*(\\left\(|\()", r"\\logbase{\1}(", s)  # log_2(x)
    s = re.sub(r"\\log_\s*\{?([^{}\s]+)\}?\s*([A-Za-z0-9])", r"\\logbase{\1}(\2)", s)  # log_2 x

    # Calculus operators that apply to everything after them.
    if m := re.match(r"\\int\s*_\s*(\{[^{}]*\}|\S)\s*\^\s*(\{[^{}]*\}|\S)(.*)$", s, re.DOTALL):
        lower, upper, rest = (g.strip("{}") for g in m.groups())
        body, var = _integrand(rest)
        return f"integral from {_tidy(_convert(lower))} to {_tidy(_convert(upper))} of {body} d{var}"
    if s.startswith(r"\int"):
        body, var = _integrand(s[4:])
        return f"integral of {body} d{var}"
    if m := re.match(r"\\frac\s*\{\s*d\s*(?:\^\s*\{?(\d)\}?)?\s*\}\s*\{\s*d\s*([a-z])\s*(?:\^\s*\{?\d\}?)?\s*\}(.*)$", s,
                     re.DOTALL):
        order, var, rest = m.groups()
        prefix = {None: "", "2": "second ", "3": "third "}.get(order, "")
        return f"{prefix}derivative of {_body(rest)} with respect to {var}"
    if m := re.match(r"\\frac\s*\{\s*\\partial\s*\}\s*\{\s*\\partial\s*([a-z])\s*\}(.*)$", s, re.DOTALL):
        var, rest = m.groups()
        return f"derivative of {_body(rest)} with respect to {var}"
    if m := re.match(r"\\lim\s*_\s*\{\s*([a-z])\s*->\s*([^{}]+?)\s*(?:\^\s*\{?([+-])\}?)?\s*\}(.*)$", s, re.DOTALL):
        var, point, side, rest = m.groups()
        point = _tidy(_convert(point.replace("^{+}", "+").replace("^{-}", "-")))
        side_text = {"+": " from the right", "-": " from the left"}.get(side or point[-1:] if point.endswith(("+", "-")) else side, "")
        point = point.rstrip("+-") if point.endswith(("+", "-")) and len(point) > 1 else point
        return f"limit of {_body(rest)} as {var} -> {point}{side_text}"
    if m := re.match(r"\\(sum|prod)\s*_\s*\{\s*([a-z])\s*=\s*([^{}]+)\}\s*\^\s*(\{[^{}]*\}|\S)(.*)$", s, re.DOTALL):
        kind, var, start, end, rest = m.groups()
        word = "sum" if kind == "sum" else "product"
        return (f"{word} of {_body(rest)} for {var} from {_tidy(_convert(start))} to "
                f"{_tidy(_convert(end.strip('{}')))}")
    if r"\int" in s or r"\lim" in s or r"\sum" in s or r"\prod" in s:
        raise LatexInputError("Please put the ∫, lim or Σ at the start, with what it applies to after it.")
    s = re.sub(r"\\frac\s*\{\s*d\s*y\s*\}\s*\{\s*d\s*([a-z])\s*\}", r"dy/d\1", s)  # differential equations
    if r"\partial" in s:
        raise LatexInputError("Please write a partial derivative as ∂/∂x followed by the expression.")
    return _tidy(_abs_bars(_convert(s)))


def _body(rest: str) -> str:
    return _tidy(_abs_bars(_convert(rest.strip())))


def _integrand(rest: str) -> tuple[str, str]:
    m = re.match(r"(.*?)\s*d\s*([a-z])\s*$", rest.strip(), re.DOTALL)
    if not m or not m.group(1).strip():
        raise LatexInputError("An integral needs what to integrate and a dx (or dt, du, ...) at the end.")
    return _body(m.group(1)), m.group(2)
