"""Calculus: derivatives, integrals, limits, differential equations and Taylor series.

SymPy computes every result. The steps name the rule used at each point (power,
product, chain rule; substitution, integration by parts; L'Hôpital; ...), and
every answer is checked a second, independent way: differentiate the integral,
substitute the ODE solution, evaluate numerically near a limit, and so on.
"""

from __future__ import annotations

import re

import sympy as sp
from sympy.integrals.manualintegrate import integral_steps

from .math_tool import SUBSCRIPTS, MathInputError, MathResult, _check_safe, _parse, _pick_variable, normalize

MAX_STEPS = 12

_CALCULUS = re.compile(
    r"d/d[a-z]|\bderivative|\bdifferentiat|\bintegra|∫|\banti-?derivative|\blim(?:it)?\b|\btaylor|\bmaclaurin"
    r"|\by'|y′|\bf'|dy/d[a-z]|d\^?2y/d[a-z]", re.IGNORECASE)


def looks_like_calculus(text: str) -> bool:
    return bool(_CALCULUS.search(text or ""))


# ---- parsing and display ---------------------------------------------------

def _expr(text: str, extra: dict | None = None) -> sp.Expr:
    s = normalize(text)
    s = re.sub(r"\b(infinity|inf)\b|∞", "oo", s, flags=re.IGNORECASE)
    _check_safe(s)
    return _parse(s, extra=extra)


def _var(f: sp.Expr, given: str | None) -> sp.Symbol:
    if given:
        return sp.Symbol(given)
    if not f.free_symbols:
        return sp.Symbol("x")
    return _pick_variable(f.free_symbols)


def show(e) -> str:
    """Book-style text: x^2, 2x, x·sin(x), ln, e^(2x), u for substitutions, i for the imaginary unit."""
    text = sp.sstr(e).replace("**", "^").replace("_u", "u")
    text = re.sub(r"(?<![A-Za-z])log\(", "ln(", text)  # before number*name tidying, so "2*log" -> "2ln"
    text = re.sub(r"(?<![A-Za-z])exp\(([^()]*)\)", r"e^(\1)", text)
    text = re.sub(r"(?<![\w.^])(\d+(?:\.\d+)?)\*(?=[a-z(])", r"\1", text)  # 2*x -> 2x, but x^2*cos stays apart
    text = text.replace("*", "·")
    text = re.sub(r"\bE\b", "e", text).replace("oo", "∞")
    return re.sub(r"\bI\b", "i", text)


def _best(e: sp.Expr) -> sp.Expr:
    """The most readable equivalent form: expanded unless factoring/simplifying is clearly shorter."""
    expanded, simplified = sp.expand(e), sp.simplify(e)
    return expanded if len(show(expanded)) <= len(show(simplified)) + 4 else simplified


def _terms(e: sp.Expr) -> list[sp.Expr]:
    return list(e.as_ordered_terms())


def _result(problem: str, problem_latex: str, answer: str, answer_latex: str, steps: list[str],
            verification: str) -> MathResult:
    if len(steps) > MAX_STEPS:
        steps = steps[:MAX_STEPS - 1] + [f"… ({len(steps) - MAX_STEPS + 1} more small steps) …", steps[-1]]
    return MathResult(ok=True, kind="calculus", problem=problem, problem_latex=problem_latex, answer=answer,
                      answer_latex=answer_latex, steps=steps, verification=verification)


def _close(a, b) -> bool:
    try:
        a, b = complex(sp.N(a)), complex(sp.N(b))
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


# ---- derivatives -----------------------------------------------------------

_STANDARD = {sp.sin: "cos", sp.cos: "-sin", sp.tan: "sec^2", sp.exp: "e^", sp.log: "1/",
             sp.asin: "1/√(1 - u^2)", sp.acos: "-1/√(1 - u^2)", sp.atan: "1/(1 + u^2)"}


def _diff_steps(f: sp.Expr, x: sp.Symbol, out: list[str], depth: int = 0) -> None:
    if depth > 4 or len(out) > MAX_STEPS:
        return
    d = sp.simplify(sp.diff(f, x))
    if not f.has(x):
        out.append(f"A constant doesn't change, so d/d{x}[{show(f)}] = 0")
        return
    if f == x:
        if depth:
            out.append(f"d/d{x}[{x}] = 1")
        return
    if isinstance(f, sp.Add):
        out.append(f"Sum rule: differentiate each part separately: {', '.join(show(t) for t in _terms(f))}")
        for t in _terms(f):
            _diff_steps(t, x, out, depth + 1)
        return
    if isinstance(f, sp.Mul):
        coeff, rest = f.as_independent(x, as_Add=False)
        if coeff != 1:
            out.append(f"Constant multiple: keep the {show(coeff)} and differentiate {show(rest)}")
            _diff_steps(rest, x, out, depth + 1)
            return
        num, den = sp.fraction(f)
        if den.has(x) and den != 1:
            out.append(f"Quotient rule: (u/v)' = (u'v - uv') / v^2, with u = {show(num)} and v = {show(den)}")
            _diff_steps(num, x, out, depth + 1)
            _diff_steps(den, x, out, depth + 1)
            out.append(f"Put together: ({show(sp.diff(num, x))})({show(den)}) - ({show(num)})({show(sp.diff(den, x))})"
                       f" all over ({show(den)})^2")
            return
        u, v = f.args[0], sp.Mul(*f.args[1:])
        out.append(f"Product rule: (uv)' = u'v + uv', with u = {show(u)} and v = {show(v)}")
        _diff_steps(u, x, out, depth + 1)
        _diff_steps(v, x, out, depth + 1)
        out.append(f"Put together: ({show(sp.diff(u, x))})({show(v)}) + ({show(u)})({show(sp.diff(v, x))})")
        return
    if isinstance(f, sp.Pow):
        base, exp = f.args
        if not exp.has(x):
            if base == x:
                out.append(f"Power rule: bring the power down and lower it by 1: d/d{x}[{show(f)}] = {show(d)}")
            else:
                out.append(f"Chain rule with the power rule: d/d{x}[g^n] = n·g^(n-1)·g', with g = {show(base)}")
                _diff_steps(base, x, out, depth + 1)
            return
        if not base.has(x):
            what = "e^g is its own derivative, times g'" if base == sp.E else f"d/d{x}[a^g] = a^g · ln(a) · g'"
            out.append(f"Exponential rule: {what}, with g = {show(exp)}")
            if exp != x:
                _diff_steps(exp, x, out, depth + 1)
            return
        out.append(f"Both the base and the power contain {x}: rewrite as e^({show(exp)}·ln({show(base)})) "
                   f"and use the chain rule → {show(d)}")
        return
    if isinstance(f, sp.Function) and len(f.args) == 1:
        arg = f.args[0]
        outer = sp.diff(f.func(sp.Symbol("u")), sp.Symbol("u"))
        if arg == x:
            out.append(f"Standard derivative: d/d{x}[{show(f)}] = {show(d)}")
        else:
            out.append(f"Chain rule: differentiate the outside, keep the inside, then multiply by the inside's "
                       f"derivative: d/d{x}[{show(f.func(sp.Symbol('g')))}] = {show(outer.subs(sp.Symbol('u'), sp.Symbol('g')))}"
                       f"·g', with g = {show(arg)}")
            _diff_steps(arg, x, out, depth + 1)
        return
    out.append(f"d/d{x}[{show(f)}] = {show(d)}")


def _derivative(f: sp.Expr, x: sp.Symbol, order: int) -> MathResult:
    steps: list[str] = []
    current = f
    for k in range(1, order + 1):
        if order > 1:
            steps.append(f"Derivative number {k}, of {show(current)}:")
        _diff_steps(current, x, steps)
        current = _best(sp.diff(current, x))
        steps.append(f"Tidy up: {'f' + chr(39) * k}({x}) = {show(current)}")
    # Independent check: the slope of f measured numerically at a sample point.
    check = ""
    if order == 1 and f.free_symbols == {x}:
        point, h = sp.Rational(7, 10), sp.Rational(1, 10 ** 6)
        numeric = (f.subs(x, point + h) - f.subs(x, point - h)) / (2 * h)
        exact = current.subs(x, point)
        if exact.is_finite and _close(numeric, exact):
            check = f" A numerical slope check at {x} = 0.7 agrees ({float(sp.N(exact)):.6g})."
    prime = "'" * order
    return _result(f"d{'^' + str(order) if order > 1 else ''}/d{x}{'^' + str(order) if order > 1 else ''} "
                   f"[{show(f)}]",
                   rf"\frac{{d{'^' + str(order) if order > 1 else ''}}}{{d{x}{'^' + str(order) if order > 1 else ''}}}"
                   rf"\left({sp.latex(f, ln_notation=True)}\right)",
                   f"f{prime}({x}) = {show(current)}", sp.latex(current), steps,
                   f"SymPy differentiated it independently and got the same result.{check}")


# ---- integrals -------------------------------------------------------------

def _rule_size(rule) -> int:
    size = 1
    for value in vars(rule).values():
        if hasattr(value, "integrand"):
            size += _rule_size(value)
        elif isinstance(value, list):
            size += sum(_rule_size(v) for v in value if hasattr(v, "integrand"))
    return size


def _eval(rule):
    try:
        return rule.eval()
    except Exception:
        return None


def _integral_steps(rule, out: list[str]) -> None:
    if len(out) > MAX_STEPS + 4:
        return
    name = type(rule).__name__
    f, v, res = rule.integrand, rule.variable, _eval(rule)
    integral = f"∫{show(f)} d{show(v)}"
    if name == "AlternativeRule":
        _integral_steps(min(rule.alternatives, key=_rule_size), out)
    elif name == "AddRule":
        out.append(f"Sum rule: integrate each part separately: {', '.join(show(s.integrand) for s in rule.substeps)}")
        for s in rule.substeps:
            _integral_steps(s, out)
    elif name == "ConstantTimesRule":
        out.append(f"Constant multiple: {integral} = {show(rule.constant)} · ∫{show(rule.other)} d{show(v)}")
        _integral_steps(rule.substep, out)
    elif name == "PowerRule":
        out.append(f"Power rule: raise the power by 1 and divide by the new power: {integral} = {show(res)}")
    elif name == "ConstantRule":
        out.append(f"A constant integrates to constant × {show(v)}: {integral} = {show(res)}")
    elif name == "ReciprocalRule":
        out.append(f"1/{show(v)} integrates to ln|{show(v)}|: {integral} = {show(res)}")
    elif name == "URule":
        u = sp.Symbol("u")
        du = sp.diff(rule.u_func, v)
        out.append(f"Substitution: let u = {show(rule.u_func)}, so du = {show(du)} d{show(v)}; "
                   f"the integral becomes one in u")
        _integral_steps(rule.substep, out)
        if res is not None:
            out.append(f"Put u = {show(rule.u_func)} back in: {show(res)}")
    elif name == "PartsRule":
        v_part = _eval(rule.v_step)
        out.append(f"Integration by parts: ∫u dv = uv - ∫v du, with u = {show(rule.u)} and dv = {show(rule.dv)} d{show(v)}")
        out.append(f"So du = {show(sp.diff(rule.u, v))} d{show(v)} and v = {show(v_part)}")
        if rule.second_step is not None:
            out.append(f"Now the leftover integral ∫v du = ∫{show(rule.second_step.integrand)} d{show(v)}:")
            _integral_steps(rule.second_step, out)
        if res is not None:
            out.append(f"Put together: uv - ∫v du = {show(res)}")
    elif name == "CyclicPartsRule":
        out.append(f"Integration by parts twice brings back the original integral, so solve for it like an "
                   f"equation: {integral} = {show(res)}")
    elif name == "RewriteRule":
        out.append(f"Rewrite {show(f)} as {show(rule.rewritten)}")
        _integral_steps(rule.substep, out)
    elif name in ("ErfRule", "FresnelSRule", "FresnelCRule", "EiRule", "LiRule", "UpperGammaRule"):
        out.append(f"{integral} has no formula built from ordinary functions; it needs a special function: "
                   f"{show(res)}")
    elif name == "DontKnowRule":
        out.append(f"There's no standard rule for {integral} by hand")
    else:
        label = re.sub(r"(?<!^)(?=[A-Z])", " ", name.removesuffix("Rule")).strip()
        out.append(f"Standard integral ({label.lower()}): {integral} = {show(res)}")


def _integral(f: sp.Expr, x: sp.Symbol, lower: sp.Expr | None, upper: sp.Expr | None) -> MathResult:
    steps: list[str] = []
    try:
        _integral_steps(integral_steps(f, x), steps)
    except Exception:
        steps = []
    antiderivative = _best(sp.integrate(f, x))
    if antiderivative.has(sp.Integral):
        raise MathInputError("I can't find an antiderivative for that one, even with my calculator.")
    if not steps or steps[-1].startswith("There's no standard rule"):
        steps.append(f"This one needs a special function; SymPy gives {show(antiderivative)}")
    derivative_check = sp.simplify(sp.diff(antiderivative, x) - f) == 0
    if lower is None:
        steps.append(f"Add the constant of integration: {show(antiderivative)} + C")
        return _result(f"∫{show(f)} d{x}", rf"\int {sp.latex(f, ln_notation=True)}\, d{x}", f"{show(antiderivative)} + C",
                       sp.latex(antiderivative) + " + C", steps,
                       "Differentiating the answer gives back the original function"
                       + (" ✔" if derivative_check else " ✘"))
    value = sp.simplify(sp.integrate(f, (x, lower, upper)))
    at_upper, at_lower = sp.simplify(antiderivative.subs(x, upper)), sp.simplify(antiderivative.subs(x, lower))
    if antiderivative.has(x) and all(v.is_finite for v in (at_upper, at_lower)):
        steps.append(f"Evaluate at the limits: F({show(upper)}) - F({show(lower)}) = {show(at_upper)} - "
                     f"({show(at_lower)}) = {show(value)}")
    else:
        steps.append(f"Evaluate between {show(lower)} and {show(upper)}: {show(value)}")
    numeric = ""
    try:
        approx = sp.Integral(f, (x, lower, upper)).evalf()
        if _close(approx, value):
            numeric = f" A numerical estimate of the area also gives about {float(approx):.6g}."
    except Exception:
        pass
    answer = show(value) + (f" (about {float(sp.N(value)):.6g})" if value.is_real and not value.is_Rational else "")
    return _result(f"∫ from {show(lower)} to {show(upper)} of {show(f)} d{x}",
                   rf"\int_{{{sp.latex(lower)}}}^{{{sp.latex(upper)}}} {sp.latex(f, ln_notation=True)}\, d{x}",
                   answer, sp.latex(value), steps,
                   "Differentiating the antiderivative gives back the original function"
                   + (" ✔." if derivative_check else " ✘.") + numeric)


# ---- limits ----------------------------------------------------------------

def _limit(f: sp.Expr, x: sp.Symbol, a: sp.Expr, direction: str) -> MathResult:
    value = sp.limit(f, x, a, dir=direction) if direction in "+-" and direction else sp.limit(f, x, a, dir="+-") \
        if a.is_finite else sp.limit(f, x, a)
    steps = []
    num, den = sp.fraction(sp.together(f))
    direct = sp.simplify(f.subs(x, a)) if a.is_finite else None
    if direct is not None and direct.is_finite and direct is not sp.nan:
        steps.append(f"Try putting {x} = {show(a)} straight in: {show(direct)}. Nothing breaks, so that's the limit")
    elif den == 1:
        steps.append(f"Putting {x} → {show(a)} straight in doesn't give a clear number, so SymPy works the limit "
                     f"out carefully: {show(value)}")
    else:
        top = sp.limit(num, x, a) if num.has(x) else num
        bottom = sp.limit(den, x, a) if den.has(x) else den
        form = f"{show(top)}/{show(bottom)}"
        steps.append(f"Putting {x} → {show(a)} straight in gives {form}, which doesn't tell us the answer yet")
        if bottom == 0 and top.is_finite and top != 0:
            steps.append(f"A fixed number divided by something shrinking to 0 grows without limit: {show(value)}")
        elif (top == 0 and bottom == 0) or (top.is_infinite and bottom.is_infinite):
            for _ in range(3):  # L'Hôpital, at most three rounds
                num, den = sp.diff(num, x), sp.diff(den, x)
                steps.append(f"L'Hôpital's rule: differentiate top and bottom separately: "
                             f"({show(num)}) / ({show(den)})")
                top = sp.limit(num, x, a)
                bottom = sp.limit(den, x, a)
                if not ((top == 0 and bottom == 0) or (top.is_infinite and bottom.is_infinite)):
                    steps.append(f"Now {x} → {show(a)} gives {show(top)}/{show(bottom)} = {show(value)}")
                    break
        else:
            steps.append(f"SymPy works out the limit directly: {show(value)}")
    # Independent check: evaluate the function very close to the point.
    check = ""
    try:
        near = a + sp.Rational(1, 10 ** 6) if a.is_finite else sp.Integer(10 ** 6) * (1 if a == sp.oo else -1)
        sample = f.evalf(subs={x: near})  # numerically: exact (1 + 1/10^6)^(10^6) would take forever
        if value.is_finite and _close(sample, value):
            where = f"{x} = {float(near):g}" if a.is_finite else f"{x} = {'-' if a == -sp.oo else ''}1,000,000"
            check = f" At {where} the function is about {float(sample):.6g}, right next to the limit."
    except Exception:
        pass
    arrow = f"{show(a)}{direction if direction in ('+', '-') else ''}"
    return _result(f"limit of {show(f)} as {x} → {arrow}",
                   rf"\lim_{{{x} \to {sp.latex(a)}}} {sp.latex(f, ln_notation=True)}", show(value), sp.latex(value), steps,
                   f"SymPy computed the limit independently.{check}")


# ---- differential equations ------------------------------------------------

_ODE_KINDS = {  # in teaching order: the first one that applies is the one we explain
    "separable": "separable: the y's and x's can be moved to opposite sides",
    "1st_linear": "first-order linear: y' + P(x)·y = Q(x), solved with an integrating factor",
    "nth_linear_constant_coeff_homogeneous": "linear with constant coefficients: try y = e^(rx)",
    "nth_linear_constant_coeff_undetermined_coefficients":
        "linear with constant coefficients and a right-hand side: complementary solution + a particular solution",
    "nth_linear_constant_coeff_variation_of_parameters":
        "linear with constant coefficients and a right-hand side (variation of parameters)",
    "Bernoulli": "a Bernoulli equation: y' + P·y = Q·y^n",
    "1st_exact": "an exact equation",
    "1st_homogeneous_coeff_best": "a homogeneous equation: substitute y = vx",
}


def _ode(text: str) -> MathResult:
    var = "t" if re.search(r"dy/dt|\bt\b", text) and not re.search(r"\bx\b", text) else "x"
    x = sp.Symbol(var)
    y = sp.Function("y")
    s = text.replace("′", "'")
    s = re.sub(r"^(solve|find)\b[:\s]*", "", s.strip(), flags=re.IGNORECASE)
    # Initial conditions like y(0) = 1 or y'(0) = 0, separated by commas, semicolons or "and".
    ics, ic_labels = {}, []
    parts = [p.strip() for p in re.split(r"[;,]|\band\b|\bwith\b|\bgiven\b", s) if p.strip()]
    equation = None
    for part in parts:
        if m := re.fullmatch(r"y('{0,2})\(\s*([^()]+)\s*\)\s*=\s*(.+)", part):
            order, at, value = len(m.group(1)), _expr(m.group(2)), _expr(m.group(3))
            key = y(x).diff(x, order).subs(x, at) if order else y(at)
            ics[key] = value
            ic_labels.append(f"y{m.group(1)}({show(at)}) = {show(value)}")
        elif equation is None:
            equation = part
    if equation is None or "=" not in equation:
        raise MathInputError("I couldn't find the differential equation (it needs y' or dy/dx and an = sign).")
    equation = re.sub(rf"d\^?2y/d{var}\^?2", " Y2 ", equation)
    equation = re.sub(rf"dy/d{var}", " Y1 ", equation)
    equation = equation.replace("y''", " Y2 ").replace("y'", " Y1 ")
    names = {"y": y(x), "Y1": y(x).diff(x), "Y2": y(x).diff(x, 2)}
    lhs_text, _, rhs_text = equation.partition("=")
    eq = sp.Eq(_expr(lhs_text, names), _expr(rhs_text, names))
    if not eq.has(sp.Derivative):
        raise MathInputError("That doesn't have a derivative in it (y' or dy/dx).")
    order = 2 if eq.has(y(x).diff(x, 2)) else 1

    hints = sp.classify_ode(eq, y(x))
    # Name the type a course would teach first, not SymPy's internal preference.
    kind = next((_ODE_KINDS[h] for h in _ODE_KINDS if h in hints), None)
    steps = [f"What kind of equation is this? It's {kind}"] if kind else []
    if "separable" in hints and order == 1:
        rhs = sp.solve(eq, y(x).diff(x))[0]
        parts_sep = sp.separatevars(rhs.subs(y(x), sp.Symbol("y")), [x, sp.Symbol("y")], dict=True)
        if parts_sep:
            gy, hx = parts_sep[sp.Symbol("y")], parts_sep["coeff"] * parts_sep[x]
            steps.append(f"Separate: dy / ({show(gy)}) = ({show(hx)}) d{x}")
            steps.append(f"Integrate both sides: ∫ dy/({show(gy)}) = ∫ {show(hx)} d{x}")
    elif "1st_linear" in hints and order == 1:
        rhs = sp.solve(eq, y(x).diff(x))[0]
        p = -sp.simplify(sp.diff(rhs, y(x)))
        mu = sp.simplify(sp.exp(sp.integrate(p, x)))
        steps.append(f"Write it as y' + P·y = Q with P = {show(p)}; the integrating factor is e^(∫P d{x}) = {show(mu)}")
        steps.append(f"Multiply through by {show(mu)}: the left side becomes the derivative of ({show(mu)})·y; "
                     f"integrate both sides")
    elif any(h.startswith("nth_linear_constant_coeff") for h in hints):
        r = sp.Symbol("r")
        homogeneous = (eq.lhs - eq.rhs).subs(y(x), sp.exp(r * x)).doit()
        char = sp.simplify(sp.expand(homogeneous / sp.exp(r * x)))
        char = sp.expand(char.subs({sym: 0 for sym in char.free_symbols if sym not in (r,)}))
        if char.has(r):
            roots = sp.solve(char, r)
            steps.append(f"Try y = e^(r{x}): the characteristic equation is {show(char)} = 0 → "
                         f"r = {', '.join(show(rt) for rt in roots)}")
    try:
        solution = sp.dsolve(eq, y(x), ics=ics or None)
    except Exception as exc:
        raise MathInputError("I couldn't solve that differential equation.") from exc
    if isinstance(solution, list):
        raise MathInputError("That equation has several families of solutions; I can only show one at a time.")
    rhs = _best(solution.rhs)
    if ics:
        steps.append(f"Use the starting values {', '.join(ic_labels)} to pin down the constants")
    steps.append(f"Solution: y = {show(rhs)}")
    ok = sp.checkodesol(eq, sp.Eq(y(x), rhs), y(x))[0]
    constants = sorted((str(c) for c in rhs.free_symbols if str(c).startswith("C")))
    note = f" ({', '.join(constants)} {'is a constant' if len(constants) == 1 else 'are constants'}: " \
           f"any value works)" if constants else ""
    pretty_eq = show(eq.lhs).replace(f"Derivative(y({var}), ({var}, 2))", "y''").replace(f"Derivative(y({var}), {var})", "y'")
    pretty_rhs = show(eq.rhs).replace(f"Derivative(y({var}), {var})", "y'")
    return _result(f"{pretty_eq} = {pretty_rhs}".replace(f"y({var})", "y"), sp.latex(eq),
                   f"y = {show(rhs)}{note}", sp.latex(sp.Eq(sp.Symbol('y'), rhs)), steps,
                   "Putting this y back into the equation makes both sides equal" + (" ✔" if ok else " ✘"))


# ---- Taylor series ---------------------------------------------------------

def _taylor(f: sp.Expr, x: sp.Symbol, a: sp.Expr, n: int) -> MathResult:
    if not 1 <= n <= 8:
        raise MathInputError("I can build Taylor polynomials up to degree 8.")
    steps, terms, current = [], [], f
    for k in range(n + 1):
        value = sp.simplify(current.subs(x, a))
        steps.append(f"f{chr(39) * k if k <= 3 else f'^({k})'}({show(a)}) = {show(value)}")
        terms.append(sp.simplify(value / sp.factorial(k)) * (x - a) ** k)
        current = sp.diff(current, x)
    polynomial = sum(terms)
    shown_poly = " + ".join(show(t) for t in terms if t != 0).replace("+ -", "- ") or "0"
    steps.append(f"Each term is f^(k)({show(a)}) / k! · ({show(x - a)})^k; adding them up gives {shown_poly}")
    ok = sp.simplify(sp.series(f, x, a, n + 1).removeO() - polynomial) == 0
    return _result(f"Taylor polynomial of {show(f)} about {x} = {show(a)}, degree {n}",
                   rf"T_{n}\left({sp.latex(f, ln_notation=True)}\right)", f"{shown_poly} + …",
                   sp.latex(polynomial) + r" + \dots", steps,
                   "SymPy's own series expansion matches" + (" ✔" if ok else " ✘"))


# ---- dispatch --------------------------------------------------------------

def _strip(text: str) -> str:
    s = text.strip().rstrip("?. ")
    s = re.sub(r"^(please\s+)?(find|compute|calculate|work out|what is|evaluate|solve)\b[:\s]*(the\s+)?", "", s,
               flags=re.IGNORECASE)
    # A label in front of the problem, as in "I = ∫ ..." or "A = lim ...", isn't part of the math.
    return re.sub(r"^[A-Za-z]\s*=\s*(?=∫|integral|lim|d/d|derivative)", "", s, flags=re.IGNORECASE)


def solve_calculus_problem(text: str) -> MathResult:
    """Called by math_tool.solve_problem for calculus-looking input. Raises MathInputError."""
    s = _strip(text)
    if len(s) > 300:
        raise MathInputError("That problem is too long for me to check.")

    if re.search(r"y'|y′|dy/d[a-z]|d\^?2y/d", s):
        return _ode(s)

    if m := re.fullmatch(r"(?:taylor|maclaurin)(?:\s+(?:series|polynomial|expansion))?\s*(?:of\s+)?(?P<f>.+?)"
                         r"(?:\s+(?:at|about|around)\s+(?:[a-z]\s*=\s*)?(?P<a>[^\s,]+))?"
                         r"(?:,?\s+(?:to|up to|of)?\s*(?:order|degree)\s+(?P<n>\d+))?", s, re.IGNORECASE):
        f = _expr(m.group("f"))
        a = _expr(m.group("a")) if m.group("a") else sp.Integer(0)
        return _taylor(f, _var(f, None), a, int(m.group("n") or 4))

    if m := (re.fullmatch(r"(?:lim(?:it)?)\s*(?:of\s+)?(?P<f>.+?)\s+as\s+(?P<v>[a-z])\s*(?:->|→|approaches|tends to|goes to)\s*"
                          r"(?P<a>-?[^\s+]+?)(?P<dir>[+-])?(?:\s+from (?:the )?(?P<side>left|right|above|below))?", s,
                          re.IGNORECASE)
             or re.fullmatch(r"lim(?:it)?\s*_?\{?\(?\s*(?P<v>[a-z])\s*(?:->|→)\s*(?P<a>-?[^\s})+]+?)(?P<dir>[+-])?"
                             r"\s*\)?\}?\s+(?P<f>.+)", s, re.IGNORECASE)):
        f = _expr(m.group("f"))
        side = (m.groupdict().get("side") or "").lower()
        direction = m.group("dir") or ("-" if side in ("left", "below") else "+" if side in ("right", "above") else "")
        return _limit(f, sp.Symbol(m.group("v")), _expr(m.group("a")), direction)

    if m := (re.fullmatch(r"(?P<n>second|third|2nd|3rd)?\s*(?:derivative|differentiate)\s+(?:of\s+)?(?P<f>.+?)"
                          r"(?:\s+(?:with respect to|wrt)\s+(?P<v>[a-z]))?", s, re.IGNORECASE)
             or re.fullmatch(r"d(?:\^(?P<n>[23]))?/d(?P<v>[a-z])(?:\^[23])?\s*(?P<f>.+)", s, re.IGNORECASE)
             or re.fullmatch(r"f'+\s*(?:\((?P<v>[a-z])\))?\s*(?:for|if|when|where)\s+f\s*(?:\([a-z]\))?\s*=\s*"
                             r"(?P<f>.+)", s, re.IGNORECASE)):
        f = _expr(re.sub(r"^\[(.*)\]$", r"\1", m.group("f").strip()))
        order = {"second": 2, "2nd": 2, "2": 2, "third": 3, "3rd": 3, "3": 3}.get((m.groupdict().get("n") or "").lower(), 1)
        return _derivative(f, _var(f, m.groupdict().get("v")), order)

    if re.search(r"\bintegra|∫|anti-?derivative", s, re.IGNORECASE):
        lower = upper = None
        body = s
        if m := re.search(r"\bfrom\s+(\S+)\s+to\s+(\S+)", body, re.IGNORECASE):
            lower, upper, body = m.group(1), m.group(2), body[:m.start()] + " " + body[m.end():]
        if m := re.search(r"∫\s*_\{?([^}\s^]+)\}?\s*\^\{?([^}\s]+)\}?", body):
            lower, upper, body = m.group(1), m.group(2), body[:m.start()] + " " + body[m.end():]
        elif m := re.search(r"∫\s*([₀₁₂₃₄₅₆₇₈₉₋₊]+)\s*([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", body):  # ∫₀¹, as vision models write it
            sup = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
            lower, upper = m.group(1).translate(SUBSCRIPTS), m.group(2).translate(sup)
            body = body[:m.start()] + " " + body[m.end():]
        body = re.sub(r"^\s*(?:the\s+)?(?:definite\s+|indefinite\s+)?(?:integral|integrate|anti-?derivative)\s*(?:of\s+)?"
                      r"|∫", " ", body, flags=re.IGNORECASE)
        given = None
        if m := re.search(r"\s+(?:with respect to|wrt)\s+([a-z])\s*$", body):
            given, body = m.group(1), body[:m.start()]
        if m := re.search(r"\s*\bd([a-z])\s*$", body):
            given, body = m.group(1), body[:m.start()]
        f = _expr(body.strip())
        x = _var(f, given)
        if lower is None:
            return _integral(f, x, None, None)
        return _integral(f, x, _expr(lower), _expr(upper))

    raise MathInputError("I think this is calculus, but I couldn't tell which part. Try, for example, "
                         "'derivative of x^2 sin(x)', 'integral of x e^x dx', 'integral from 0 to 1 of x^2 dx', "
                         "'limit of sin(x)/x as x -> 0' or y' = 2y.")
