import pytest

from tutor.math_tool import solve_problem


@pytest.mark.parametrize("text, answer", [
    ("(2 + 3i) + (1 - i)", "3 + 2i"),
    ("(2 + 3i) - (5 - 2i)", "-3 + 5i"),
    ("(2 + 3i)(1 - i)", "5 + i"),
    ("(1 + 2i) / (3 - i)", "1/10 + (7/10)i"),
    ("(3 + 4i)^2", "-7 + 24i"),
    ("abs(3 + 4i)", "5"),
    ("conjugate(2 - 5i)", "2 + 5i"),
    ("i^7", "-i"),
    ("x^2 + 2x + 5 = 0", "x = -1 - 2i, x = -1 + 2i"),
])
def test_complex_answers(text, answer):
    result = solve_problem(text)
    assert result.ok, result.error
    assert result.answer == answer


def test_complex_multiplication_uses_i_squared():
    steps = solve_problem("(2 + 3i)(1 - i)").steps
    assert "Use i^2 = -1" in steps[1]


def test_negative_squares_are_bracketed():
    assert "3^2 + (-1)^2" in solve_problem("(1 + 2i) / (3 - i)").steps[1]


def test_quadratic_formula_when_it_does_not_factor():
    steps = solve_problem("x^2 + 2x + 5 = 0").steps
    assert steps[1] == "Discriminant b^2 - 4ac = (2)^2 - 4(1)(5) = -16"
    assert "complex" in steps[2]


@pytest.mark.parametrize("text", ["x + y = 10; x - y = 2", "x + y = 10, x - y = 2", "x + y = 10 and x - y = 2"])
def test_typed_simultaneous_equations(text):
    result = solve_problem(text)
    assert result.ok, result.error
    assert result.answer == "x = 6, y = 4"
    assert result.steps[0].startswith("Equation 2: rearrange")


def test_commas_inside_functions_do_not_split_equations():
    assert solve_problem("gcd(12, 18)").answer == "6"
