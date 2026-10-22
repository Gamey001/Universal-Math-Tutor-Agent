import pytest

from tutor.math_tool import solve_problem


@pytest.mark.parametrize("text, answer", [
    ("derivative of x^2 sin(x)", "f'(x) = x^2·cos(x) + 2x·sin(x)"),
    ("d/dx (3x^4 - 2x + 7)", "f'(x) = 12x^3 - 2"),
    ("derivative of sin(x^2)", "f'(x) = 2x·cos(x^2)"),
    ("second derivative of x^3 + 2x", "f''(x) = 6x"),
    ("f'(x) for f(x) = sqrt(x^2 + 1)", "f'(x) = x/sqrt(x^2 + 1)"),
    ("integral of x^2 dx", "x^3/3 + C"),
    ("integrate 3x^2 + 2x", "x^3 + x^2 + C"),
    ("integral of x e^x dx", "x·e^(x) - e^(x) + C"),
    ("integral of 2x cos(x^2) dx", "sin(x^2) + C"),
    ("integral from 0 to 1 of x^2 dx", "1/3"),
    ("integral from 0 to pi of sin(x) dx", "2"),
    ("limit of sin(x)/x as x -> 0", "1"),
    ("lim x->0 (1 - cos(x))/x^2", "1/2"),
    ("limit of (1 + 1/n)^n as n -> infinity", "e"),
    ("limit of 1/x as x -> 0 from the right", "∞"),
    ("y' = 2y, y(0) = 3", "y = 3e^(2x)"),
    ("y'' + y = 0, y(0) = 0, y'(0) = 1", "y = sin(x)"),
    ("y'' - 3y' + 2y = 0", "y = C1·e^(x) + C2·e^(2x) (C1, C2 are constants: any value works)"),
    ("taylor series of e^x", "1 + x + x^2/2 + x^3/6 + x^4/24 + …"),
])
def test_calculus_answers(text, answer):
    result = solve_problem(text)
    assert result.ok, result.error
    assert result.answer == answer


@pytest.mark.parametrize("text, rule", [
    ("derivative of x^2 sin(x)", "Product rule"),
    ("differentiate e^(2x) / x", "Quotient rule"),
    ("derivative of sin(x^2)", "Chain rule"),
    ("integral of x e^x dx", "Integration by parts"),
    ("integral of 2x cos(x^2) dx", "Substitution: let u = x^2"),
    ("limit of sin(x)/x as x -> 0", "L'Hôpital's rule"),
    ("y' + y = x", "integrating factor"),
    ("y'' + y = 0", "characteristic equation is r^2 + 1 = 0"),
])
def test_steps_name_the_rule(text, rule):
    assert any(rule in step for step in solve_problem(text).steps)


@pytest.mark.parametrize("text, check", [
    ("derivative of x^2 sin(x)", "numerical slope check"),
    ("integral of x e^x dx", "gives back the original function ✔"),
    ("integral from 0 to 1 of x^2 dx", "numerical estimate"),
    ("limit of sin(x)/x as x -> 0", "right next to the limit"),
    ("y' = 2y", "both sides equal ✔"),
    ("taylor series of e^x", "matches ✔"),
])
def test_every_answer_is_checked_independently(text, check):
    assert check in solve_problem(text).verification


def test_limit_check_is_fast_for_huge_powers():
    import time
    start = time.time()
    solve_problem("limit of (1 + 1/n)^n as n -> infinity")
    assert time.time() - start < 5


@pytest.mark.parametrize("text", ['derivative of __import__("os")', "limit of x as"])
def test_bad_calculus_input_is_rejected_kindly(text):
    result = solve_problem(text)
    assert not result.ok and result.error
