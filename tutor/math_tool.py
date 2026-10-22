"""Deterministic math engine.

Every number the tutor shows the learner comes from here, never from the LLM.
Input is a plain-text expression, equation, or inequality (as extracted from a
photo or typed by the user); output is a verified ``MathResult``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

MAX_INPUT_LENGTH = 200
MAX_EXPONENT = 1000

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# Names the parser may see. Anything else made of letters must be a short
# variable name (e.g. "x", "xy"), which keeps parse_expr's eval() safe.
_ALLOWED_NAMES = {
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "abs": sp.Abs,
    "pi": sp.pi,
    "factorial": sp.factorial,
    "binomial": sp.binomial,
    "gcd": sp.gcd,
    "lcm": sp.lcm,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "ceil": sp.ceiling,
    "mod": sp.Mod,
}

_UNICODE_REPLACEMENTS = {
    "×": "*",
    "·": "*",
    "⋅": "*",
    "÷": "/",
    "−": "-",
    "–": "-",
    "—": "-",
    "²": "^2",
    "³": "^3",
    "≤": "<=",
    "≥": ">=",
    "π": "pi",
    "√": "sqrt",
}

_RELATIONS = ("<=", ">=", "<", ">", "=")


@dataclass
class MathResult:
    ok: bool
    kind: str = ""  # "equation" | "inequality" | "evaluate" | "simplify" | "check"
    problem: str = ""  # cleaned-up problem as SymPy understood it
    problem_latex: str = ""
    answer: str = ""
    answer_latex: str = ""
    steps: list[str] = field(default_factory=list)
    verification: str = ""
    error: str = ""
    values: dict[str, str] = field(default_factory=dict)  # system solutions, e.g. {"x": "50"}

    def to_dict(self) -> dict:
        return asdict(self)


class MathInputError(ValueError):
    pass


def normalize(text: str) -> str:
    """Turn handwriting-ish / unicode math into something SymPy can parse."""
    s = text.strip()
    # "Solve: 2x + 3 = 11" / "Simplify ..." / "What is ..."
    s = re.sub(r"^(solve|simplify|evaluate|calculate|compute|find|what is|work out)\b[:\s]*",
               "", s, flags=re.IGNORECASE)
    for old, new in _UNICODE_REPLACEMENTS.items():
        s = s.replace(old, new)
    s = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", s)  # "3 x 4" meaning times
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)  # "15%" -> "(15/100)"
    s = re.sub(r"\bof\b", "*", s, flags=re.IGNORECASE)  # "15% of 40"
    s = re.sub(r"(\d+)\s*(?:C|\s+choose\s+)\s*(\d+)", r"binomial(\1, \2)", s)  # "5C2", "5 choose 2"
    s = re.sub(r"(\d+)\s+mod\s+(\d+)", r"mod(\1, \2)", s, flags=re.IGNORECASE)  # "17 mod 5"
    s = s.replace("==", "=")
    s = re.sub(r"\bsqrt\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", s)  # "sqrt16" -> "sqrt(16)"
    # Drop a trailing "= ?" / "=" / "?" that just means "what is this?"
    s = re.sub(r"\s*=\s*\?*\s*$", "", s)
    s = s.rstrip("?. ").strip()
    s = re.sub(r"\s+for\s+[a-zA-Z]\s*$", "", s)  # "... for x"
    return s.strip()


def _check_safe(s: str) -> None:
    if not s:
        raise MathInputError("There is no math to work with.")
    if len(s) > MAX_INPUT_LENGTH:
        raise MathInputError("That problem is too long for me to check.")
    if not re.fullmatch(r"[0-9a-zA-Z+\-*/^().,=<>! ]+", s):
        raise MathInputError("I found symbols I don't know how to read.")
    if "!=" in s.replace(" ", "") and not re.search(r"\d!\s+=", s):
        raise MathInputError("I can check equals, but not 'not equal to' (!=) yet.")
    for word in re.findall(r"[a-zA-Z]+", s):
        if word.lower() not in _ALLOWED_NAMES and len(word) > 2:
            raise MathInputError(f"I'm not sure what '{word}' means here.")
    for exp in re.findall(r"(?:\^|\*\*)\s*\(?\s*(\d+)", s):
        if int(exp) > MAX_EXPONENT:
            raise MathInputError("That power is too big for me to check.")


# evaluate=False doesn't reach function calls, so 5! would still become 120 at once.
# These wrappers keep them unevaluated, so steps like "5! = 5 × 4 × 3 × 2 × 1" can be shown.
_UNEVALUATED_NAMES = {
    **_ALLOWED_NAMES,
    **{name: (lambda f: lambda *args: f(*args, evaluate=False))(f)
       for name, f in _ALLOWED_NAMES.items() if f in (sp.factorial, sp.binomial, sp.Mod, sp.floor, sp.ceiling)},
}


def _parse(side: str, evaluate: bool = True) -> sp.Expr:
    side = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", side)  # thousands separators: 1,000 -> 1000
    if not side.strip():
        raise MathInputError("One side of the problem is empty.")
    try:
        names = _ALLOWED_NAMES if evaluate else _UNEVALUATED_NAMES
        expr = parse_expr(side, local_dict=dict(names), transformations=_TRANSFORMS, evaluate=evaluate)
    except Exception as exc:  # SyntaxError, TokenError, TypeError, ...
        raise MathInputError("I couldn't make sense of how this problem is written.") from exc
    # (A plain Symbol is also a Boolean in SymPy, so it's allowed explicitly.)
    if not isinstance(expr, sp.Basic) or (isinstance(expr, sp.logic.boolalg.Boolean) and not expr.is_Symbol):
        raise MathInputError("That doesn't look like a calculation.")
    return expr


def _split_relation(s: str) -> tuple[str, str | None, str | None]:
    for rel in _RELATIONS:
        if rel in s:
            lhs, _, rhs = s.partition(rel)
            if any(r in rhs for r in _RELATIONS):
                raise MathInputError("I can only check one equals sign at a time.")
            return lhs, rel, rhs
    return s, None, None


def _fmt(value: sp.Basic) -> str:
    """Human-friendly form: fractions stay fractions, with a decimal hint."""
    if isinstance(value, sp.Float):
        return f"{float(value):.6g}"
    text = sp.sstr(value)
    if value.is_number and value.is_real and not value.is_Integer:
        return f"{text} (about {float(value):.6g})"
    return text


def _pick_variable(symbols: set[sp.Symbol]) -> sp.Symbol:
    for preferred in ("x", "y", "n", "a"):
        for sym in symbols:
            if sym.name == preferred:
                return sym
    return sorted(symbols, key=lambda s: s.name)[0]


def _fraction_sum_steps(unevaluated: sp.Expr) -> list[str]:
    """Common-denominator steps for sums/differences of plain fractions.

    Takes the expression parsed with evaluate=False so the original terms
    (e.g. 3/4 and 1/8) are still visible.
    """
    if not isinstance(unevaluated, sp.Add):
        return []
    terms = [sp.simplify(t) for t in unevaluated.args]
    if not all(t.is_Rational for t in terms) or all(t.q == 1 for t in terms):
        return []
    common = sp.ilcm(*[t.q for t in terms])
    scaled = [t.p * (common // t.q) for t in terms]
    rewritten = f"{scaled[0]}/{common}" + "".join(
        f" {'-' if n < 0 else '+'} {abs(n)}/{common}" for n in scaled[1:]
    )
    numerator = sum(scaled)
    steps = [
        f"Find a common denominator (same-size pieces): {common}",
        f"Rewrite each fraction with denominator {common}: {rewritten}",
        f"Add the numerators: {numerator}/{common}",
    ]
    total = sp.Rational(numerator, common)
    if total.q != common:
        steps.append(f"Simplify: {numerator}/{common} = {total}")
    return steps


def _number(value: sp.Expr) -> str:
    return f"{float(value):g}" if isinstance(value, sp.Float) else sp.sstr(value)


def _operation_steps(unevaluated: sp.Expr, max_steps: int = 8) -> list[str]:
    """One line per operation, innermost first, e.g. ["3 × 4 = 12", "12 + 2 = 14"].

    Works on the expression parsed with evaluate=False. Returns [] for anything
    beyond plain arithmetic (functions, symbols), or if it would be too long.
    """
    steps: list[str] = []

    def walk(node: sp.Expr) -> sp.Expr:
        if node.is_Number:
            return node
        if isinstance(node, sp.factorial):
            n = walk(node.args[0])
            value = sp.factorial(n)
            if n.is_Integer and 2 <= n <= 12:
                expansion = " × ".join(str(k) for k in range(int(n), 0, -1))
                steps.append(f"{_number(n)}! = {expansion} = {_number(value)}")
            else:
                steps.append(f"{_number(n)}! = {_number(value)}")
            return value
        if isinstance(node, sp.binomial):
            n, k = walk(node.args[0]), walk(node.args[1])
            value = sp.binomial(n, k)
            steps.append(f"Ways to choose {_number(k)} from {_number(n)}: "
                         f"{_number(n)}! ÷ ({_number(k)}! × {_number(n - k)}!) = {_number(value)}")
            return value
        if isinstance(node, (sp.Mod, sp.floor, sp.ceiling)):
            args = [walk(a) for a in node.args]
            value = node.func(*args)
            shown = {sp.Mod: "Remainder of {} ÷ {}", sp.floor: "Round {} down", sp.ceiling: "Round {} up"}[node.func]
            steps.append(f"{shown.format(*map(_number, args))} = {_number(value)}")
            return value
        if isinstance(node, sp.Pow):
            base, exp = walk(node.args[0]), walk(node.args[1])
            if exp == -1:  # a bare reciprocal, as in "1/4": keep it as a fraction
                return sp.Rational(1, 1) / base
            value = base ** exp
            shown = f"√{_number(base)}" if exp == sp.Rational(1, 2) else f"{_number(base)}^{_number(exp)}"
            steps.append(f"{shown} = {_number(value)}")
            return value
        if isinstance(node, sp.Mul):
            factors = list(node.args)
            if factors[0] == -1 and len(factors) == 2:  # "-x"
                return -walk(factors[1])
            acc = walk(factors[0])
            for f in factors[1:]:
                if isinstance(f, sp.Pow) and f.args[1] == -1:
                    divisor = walk(f.args[0])
                    value = acc / divisor
                    steps.append(f"{_number(acc)} ÷ {_number(divisor)} = {_number(value)}")
                else:
                    other = walk(f)
                    value = acc * other
                    steps.append(f"{_number(acc)} × {_number(other)} = {_number(value)}")
                acc = value
            return acc
        if isinstance(node, sp.Add):
            terms = [walk(t) for t in node.args]
            acc = terms[0]
            for t in terms[1:]:
                value = acc + t
                op = f"- {_number(-t)}" if t < 0 else f"+ {_number(t)}"
                steps.append(f"{_number(acc)} {op} = {_number(value)}")
                acc = value
            return acc
        raise ValueError("not plain arithmetic")

    try:
        walk(unevaluated)
    except (ValueError, TypeError):
        return []
    return steps if len(steps) <= max_steps else []


def _quadratic_steps(lhs: sp.Expr, rhs: sp.Expr, var: sp.Symbol) -> list[str]:
    """Factor-and-split steps when a quadratic factors over the rationals."""
    expr = sp.expand(lhs - rhs)
    if expr.free_symbols != {var} or sp.Poly(expr, var).degree() != 2:
        return []
    factored = sp.factor(expr)
    factors = [f for f in sp.Mul.make_args(factored) if f.has(var)]
    linear = []
    for f in factors:
        base, power = (f.args if isinstance(f, sp.Pow) else (f, 1))
        if sp.Poly(base, var).degree() != 1:
            return []
        linear.append(base)
    steps = []
    if rhs != 0:
        steps.append(f"Move everything to one side: {sp.sstr(expr)} = 0")
    steps.append(f"Factor (split into two brackets that multiply together): {sp.sstr(factored)} = 0")
    if len(linear) == 2:
        steps.append(f"If two things multiply to make 0, one of them must be 0: "
                     f"{sp.sstr(linear[0])} = 0 or {sp.sstr(linear[1])} = 0")
    return steps


def _linear_parts(expr: sp.Expr, var: sp.Symbol) -> tuple[sp.Expr, sp.Expr] | None:
    """(coefficient, constant) if expr is linear in var, else None."""
    expr = sp.expand(expr)
    if not expr.free_symbols <= {var}:
        return None
    poly = sp.Poly(expr, var)
    if poly.degree() > 1:
        return None
    return poly.coeff_monomial(var), poly.coeff_monomial(1)


_FLIPPED = {"=": "=", "<": ">", ">": "<", "<=": ">=", ">=": "<="}


def _show(expr: sp.Basic) -> str:
    """Print an expression in the order it was written."""
    # evaluate=False parses "1/8" as 1*8**-1 and "- 2" as -1*2; hide those 1s.
    return re.sub(r"(?<![\d.])1\*", "", sp.sstr(expr, order="none"))


def _show_latex(expr: sp.Basic) -> str:
    return re.sub(r"(?<![\d.])1 \\cdot ", "", sp.latex(expr, order="none"))


def _linear_steps(lhs: sp.Expr, rhs: sp.Expr, var: sp.Symbol, rel: str = "=") -> list[str]:
    """Balance-the-scale steps for a*var + b (rel) c*var + d."""
    left, right = _linear_parts(lhs, var), _linear_parts(rhs, var)
    if left is None or right is None or left[0] == right[0]:
        return []
    steps = []
    (a, b), (c, d) = left, right
    if a == 0:  # "12 = 2y": put the unknown on the left first
        (a, b), (c, d) = (c, d), (a, b)
        rel = _FLIPPED[rel]
        steps.append(f"Swap the two sides so {var} is on the left: {sp.sstr(a * var + b)} {rel} {sp.sstr(d)}")
    if c != 0:
        verb = f"Take {sp.sstr(c * var)} away from" if c > 0 else f"Add {sp.sstr(-c * var)} to"
        a, c = a - c, 0
        steps.append(f"{verb} both sides: {sp.sstr(a * var + b)} {rel} {sp.sstr(d)}")
    if b != 0:
        verb = f"Take {sp.sstr(b)} away from" if b > 0 else f"Add {sp.sstr(-b)} to"
        b, d = 0, d - b
        steps.append(f"{verb} both sides: {sp.sstr(a * var)} {rel} {sp.sstr(d)}")
    if a != 1:
        note = ""
        if a < 0 and rel != "=":
            rel = _FLIPPED[rel]
            note = " (dividing by a negative number flips the inequality sign)"
        steps.append(f"Divide both sides by {sp.sstr(a)}{note}: {var} {rel} {sp.sstr(d / a)}")
    return steps


def _fmt_interval(solution: sp.Set, var: sp.Symbol) -> str:
    """Interval(3, oo, left_open=True) -> 'x > 3' (no infinity symbols)."""
    if isinstance(solution, sp.Interval):
        lo, hi = solution.start, solution.end
        lo_op = "<" if solution.left_open else "<="
        hi_op = "<" if solution.right_open else "<="
        if lo == -sp.oo and hi == sp.oo:
            return f"every value of {var} works"
        if lo == -sp.oo:
            return f"{var} {hi_op} {_fmt(hi)}"
        if hi == sp.oo:
            return f"{var} {lo_op.replace('<', '>')} {_fmt(lo)}"
        return f"{_fmt(lo)} {lo_op} {var} {hi_op} {_fmt(hi)}"
    if solution == sp.S.EmptySet:
        return f"no value of {var} works"
    return sp.sstr(solution)


_LATEX_RELATIONS = {"=": "=", "<": "<", ">": ">", "<=": r"\le", ">=": r"\ge"}


def _solve_relation(lhs_text: str, rel: str, rhs_text: str) -> MathResult:
    lhs, rhs = _parse(lhs_text), _parse(rhs_text)
    shown_lhs, shown_rhs = _parse(lhs_text, evaluate=False), _parse(rhs_text, evaluate=False)
    symbols = lhs.free_symbols | rhs.free_symbols
    pretty = f"{_show(shown_lhs)} {rel} {_show(shown_rhs)}"
    problem_latex = (f"{_show_latex(shown_lhs)} {_LATEX_RELATIONS[rel]} {_show_latex(shown_rhs)}")

    if not symbols:  # e.g. "2 + 2 = 4": checking a statement
        compare = {"=": lambda a, b: a == b, "<": sp.Lt, ">": sp.Gt, "<=": sp.Le, ">=": sp.Ge}[rel]
        truth = bool(compare(sp.simplify(lhs), sp.simplify(rhs)))
        left = sp.simplify(lhs)
        return MathResult(
            ok=True, kind="check", problem=pretty, problem_latex=problem_latex,
            answer="true" if truth else "false",
            answer_latex=r"\text{true}" if truth else r"\text{false}",
            steps=[f"Work out the left side: {_fmt(left)}", f"Work out the right side: {_fmt(sp.simplify(rhs))}"],
            verification=f"The statement is {'true' if truth else 'false'}.",
        )

    if len(symbols) > 1:
        # One equation can't pin down two unknowns ("2x + 3y = 23" alone); word problems
        # like that go to solve_system with the rest of their equations.
        raise MathInputError("This has more than one unknown, so I need more information to pin them down.")
    var = _pick_variable(symbols)

    if rel != "=":
        relation = {"<": sp.Lt, ">": sp.Gt, "<=": sp.Le, ">=": sp.Ge}[rel](lhs, rhs)
        try:
            solution = sp.solve_univariate_inequality(relation, var, relational=False)
        except Exception as exc:
            raise MathInputError("I can't solve that kind of inequality yet.") from exc
        answer = _fmt_interval(solution, var)
        answer_latex = answer
        for op in (">=", "<="):
            answer_latex = answer_latex.replace(op, _LATEX_RELATIONS[op])
        return MathResult(
            ok=True, kind="inequality", problem=pretty, problem_latex=problem_latex,
            answer=answer, answer_latex=answer_latex,
            steps=_linear_steps(lhs, rhs, var, rel),
            verification=f"SymPy solved for {var}: {answer}.",
        )

    try:
        solutions = sp.solve(sp.Eq(lhs, rhs), var)
    except Exception as exc:
        raise MathInputError("I can't solve that kind of equation yet.") from exc
    real = [s for s in solutions if s.is_real is not False]
    solutions = real or solutions
    if not solutions:
        return MathResult(
            ok=True, kind="equation", problem=pretty, problem_latex=problem_latex,
            answer="no solution", answer_latex=r"\text{no solution}",
            verification=f"No value of {var} makes both sides equal.",
        )
    checks = []
    for s in solutions:
        left = sp.simplify(lhs.subs(var, s))
        right = sp.simplify(rhs.subs(var, s))
        checks.append(f"{var} = {_fmt(s)}: left side = {_fmt(left)}, right side = {_fmt(right)}")
    answer = ", ".join(f"{var} = {_fmt(s)}" for s in solutions)
    answer_latex = r",\quad ".join(f"{sp.latex(var)} = {sp.latex(s)}" for s in solutions)
    return MathResult(
        ok=True, kind="equation", problem=pretty, problem_latex=problem_latex,
        answer=answer, answer_latex=answer_latex,
        steps=_linear_steps(lhs, rhs, var) or _quadratic_steps(lhs, rhs, var),
        verification="Check by putting it back in: " + "; ".join(checks),
    )


def _tidy(text: str) -> str:
    """SymPy's "2*x**2" -> the "2x^2" learners see in books."""
    text = text.replace("**", "^")
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\*(?=[a-z(])", r"\1", text)  # 2*x -> 2x, 3*(x+1) -> 3(x+1)
    text = text.replace(")*(", ")(")
    text = re.sub(r"factorial\((\w+)\)", r"\1!", text)  # factorial(5) -> 5!
    return text


def _tidied(result: MathResult) -> MathResult:
    result.problem = _tidy(result.problem)
    result.answer = _tidy(result.answer)
    result.steps = [_tidy(step) for step in result.steps]
    result.verification = _tidy(result.verification)
    return result


def solve_problem(text: str) -> MathResult:
    """Solve/evaluate ``text``. Never raises; failures come back as ok=False."""
    try:
        return _tidied(_solve(text))
    except MathInputError as exc:
        return MathResult(ok=False, error=str(exc))
    except Exception:  # anything unexpected from SymPy internals
        return MathResult(ok=False, error="I couldn't check this problem with my calculator.")


def _solve(text: str) -> MathResult:
    s = normalize(text or "")
    _check_safe(s)
    lhs_text, rel, rhs_text = _split_relation(s)
    if rel is not None:
        return _solve_relation(lhs_text, rel, rhs_text)
    lhs = _parse(lhs_text)

    if not lhs.free_symbols:
        value = sp.simplify(lhs)
        original = _parse(lhs_text, evaluate=False)
        return MathResult(
            ok=True, kind="evaluate", problem=_show(original),
            problem_latex=_show_latex(original),
            answer=_fmt(value), answer_latex=sp.latex(value),
            steps=_fraction_sum_steps(original) or _operation_steps(original),
            verification=f"SymPy evaluates {_show(original)} to {_fmt(value)}.",
        )

    simplified = sp.simplify(lhs)
    factored = sp.factor(lhs)
    expanded = sp.expand(lhs)
    steps = []
    if expanded != lhs:
        steps.append(f"Expanded: {sp.sstr(expanded)}")
    if factored != expanded:
        steps.append(f"Factored: {sp.sstr(factored)}")
    return MathResult(
        ok=True, kind="simplify", problem=sp.sstr(lhs), problem_latex=sp.latex(lhs),
        answer=sp.sstr(simplified), answer_latex=sp.latex(simplified),
        steps=steps,
        verification="SymPy confirms both forms are equal for every value of the variable.",
    )


MAX_SYSTEM_EQUATIONS = 10


def solve_system(equations: list[tuple[str, str]], target: str | None = None) -> MathResult:
    """Solve several equations together. ``equations`` is [(equation, reason), ...].

    Steps follow how a person would do it: solve whichever equation has a single
    unknown left, substitute that value everywhere, repeat. Each step carries the
    reason it's allowed (e.g. "angles in a triangle add up to 180").
    Only a single, fully determined solution counts as a success.
    """
    try:
        return _tidied(_solve_system(equations, target))
    except MathInputError as exc:
        return MathResult(ok=False, error=str(exc))
    except Exception:
        return MathResult(ok=False, error="I couldn't check these equations with my calculator.")


def _system_steps(parsed: list, solution: dict) -> list[str]:
    """Human-style steps: solve single-unknown equations first; when every equation
    still has two or more unknowns, rearrange one and substitute it into the rest."""
    steps: list[str] = []
    known: dict = {}  # unknown -> number
    isolated: list = []  # (unknown, expression in other unknowns), for back-substitution
    remaining = [(shown, reason, lhs, rhs) for shown, reason, lhs, rhs in parsed]

    while remaining:
        for i, (shown, reason, lhs, rhs) in enumerate(remaining):
            free = (lhs - rhs).subs(known).free_symbols
            if len(free) == 1:
                var = free.pop()
                used = [k for k in known if (lhs - rhs).has(k)]
                note = f" (using {', '.join(f'{k} = {_fmt(known[k])}' for k in used)})" if used else ""
                known[var] = solution[var]
                steps.append(f"{reason or 'From the problem'}: {shown}{note} → {var} = {_fmt(solution[var])}")
                del remaining[i]
                break
            if not free:
                del remaining[i]
                break
        else:
            # Every equation still has 2+ unknowns: isolate the easiest unknown in one of them.
            candidates = []
            for i, (shown, reason, lhs, rhs) in enumerate(remaining):
                diff = sp.expand((lhs - rhs).subs(known))
                for var in sorted(diff.free_symbols, key=str):
                    coeff = diff.coeff(var)
                    if coeff.is_number and coeff != 0 and not (diff - coeff * var).has(var):
                        candidates.append((abs(coeff) != 1, len(str(diff)), i, var, diff))
            if not candidates:
                rest = "; ".join(r[0] for r in remaining)
                found = ", ".join(f"{k} = {_fmt(v)}" for k, v in solution.items() if k not in known)
                steps.append(f"Solve these together: {rest} → {found}")
                return steps
            _, _, i, var, diff = min(candidates, key=lambda c: c[:2])
            expr = sp.solve(diff, var)[0]
            shown, reason = remaining[i][0], remaining[i][1]
            lhs_i, rhs_i = remaining[i][2], remaining[i][3]
            if lhs_i == var and not rhs_i.has(var):  # already "m = 3s": nothing to rearrange
                steps.append(f"{reason or 'From the problem'}: {shown}")
            else:
                steps.append(f"{reason or 'From the problem'}: rearrange {shown} → {var} = {sp.sstr(expr)}")
            isolated.append((var, expr))
            del remaining[i]
            swap = f"{var} = {sp.sstr(expr)}"
            remaining = [(f"{sh}, {swap}" if " using " in sh else f"{sh} using {swap}",
                          rs, l.subs(var, expr), r.subs(var, expr))
                         for sh, rs, l, r in remaining]

    for var, expr in reversed(isolated):  # back-substitute what we rearranged
        value = expr.subs(known)
        steps.append(f"Put the value{'s' if len(expr.free_symbols) > 1 else ''} back in: "
                     f"{var} = {sp.sstr(expr)} → {var} = {_fmt(value)}")
        known[var] = value
    return steps


def _solve_system(equations: list[tuple[str, str]], target: str | None) -> MathResult:
    if not equations:
        raise MathInputError("There are no equations to solve.")
    if len(equations) > MAX_SYSTEM_EQUATIONS:
        raise MathInputError("That's more equations than I can check at once.")

    parsed = []  # (shown text, reason, lhs, rhs)
    for text, reason in equations:
        s = normalize(text)
        _check_safe(s)
        lhs_text, rel, rhs_text = _split_relation(s)
        if rel != "=":
            raise MathInputError("I can only combine equations that use = for now.")
        parsed.append((s, reason.strip(), _parse(lhs_text), _parse(rhs_text)))

    symbols = set().union(*(lhs.free_symbols | rhs.free_symbols for _, _, lhs, rhs in parsed))
    if not symbols:
        raise MathInputError("These equations have no unknowns to find.")
    solutions = sp.solve([sp.Eq(lhs, rhs) for _, _, lhs, rhs in parsed], sorted(symbols, key=str), dict=True)
    if not solutions:
        raise MathInputError("These equations contradict each other, so something was set up wrong.")
    if len(solutions) > 1:
        raise MathInputError("These equations allow more than one answer, so something is missing.")
    solution = solutions[0]
    if set(solution) != symbols or any(v.free_symbols for v in solution.values()):
        raise MathInputError("There isn't enough information to pin down every unknown.")

    steps = _system_steps(parsed, solution)

    checks = all(sp.simplify(lhs.subs(solution) - rhs.subs(solution)) == 0 for _, _, lhs, rhs in parsed)
    if not checks:
        raise MathInputError("The solution didn't check out when I put it back in.")
    values = {str(k): _fmt(v) for k, v in solution.items()}
    target_sym = next((k for k in solution if str(k) == target), None)
    shown_vars = [target_sym] if target_sym is not None else sorted(solution, key=str)
    answer = ", ".join(f"{k} = {_fmt(solution[k])}" for k in shown_vars)
    return MathResult(
        ok=True, kind="system",
        problem="; ".join(s for s, _, _, _ in parsed),
        problem_latex=r"\\ ".join(f"{sp.latex(lhs)} = {sp.latex(rhs)}" for _, _, lhs, rhs in parsed),
        answer=answer,
        answer_latex=r",\quad ".join(f"{sp.latex(k)} = {sp.latex(solution[k])}" for k in shown_vars),
        steps=steps,
        verification=f"Putting {', '.join(f'{k} = {v}' for k, v in values.items())} back in, "
                     f"all {len(parsed)} equations balance.",
        values=values,
    )
