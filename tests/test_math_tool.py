import pytest

from tutor.math_tool import normalize, solve_problem


@pytest.mark.parametrize("text, answer", [
    ("2x + 3 = 11", "x = 4"),
    ("Solve: 2x+3=11", "x = 4"),
    ("12 = 2y", "y = 6"),
    ("5x - 2 = 3x + 8", "x = 5"),
    ("3/4 + 1/8 = ?", "7/8 (about 0.875)"),
    ("3 × 4 ÷ 6", "2"),
    ("√16 + 3²", "13"),
    ("what is 15% of 40", "6"),
    ("0.5*4.2", "2.1"),
    ("x^2 - 5x + 6 = 0", "x = 2, x = 3"),
    ("2x + 1 > 7", "x > 3"),
    ("-3x <= 9", "x >= -3"),
    ("2 + 2 = 5", "false"),
    ("x + 1 = x + 2", "no solution"),
])
def test_answers(text, answer):
    result = solve_problem(text)
    assert result.ok, result.error
    assert result.answer == answer


def test_linear_steps_balance_the_scale():
    steps = solve_problem("2x + 3 = 11").steps
    assert steps == ["Take 3 away from both sides: 2x = 8", "Divide both sides by 2: x = 4"]


def test_inequality_flips_when_dividing_by_negative():
    steps = solve_problem("-3x <= 9").steps
    assert "flips the inequality sign" in steps[-1]
    assert steps[-1].endswith("x >= -3")


def test_fraction_steps_use_common_denominator():
    result = solve_problem("1/3 + 1/6")
    assert result.problem == "1/3 + 1/6"
    assert result.steps[0].endswith(": 6")
    assert result.steps[-1] == "Simplify: 3/6 = 1/2"


def test_check_shows_original_problem():
    assert solve_problem("2 + 2 = 5").problem == "2 + 2 = 5"


def test_solution_is_verified_by_substitution():
    assert "left side = 11, right side = 11" in solve_problem("2x + 3 = 11").verification


@pytest.mark.parametrize("text", [
    "", "hello world", '__import__("os").system("ls")', "2**99999", "x = = 3",
    "lambda: 1", "a.__class__", "[1, 2]", "1 = 2 = 3",
])
def test_rejects_unsafe_or_meaningless_input(text):
    result = solve_problem(text)
    assert not result.ok
    assert result.error


def test_normalize_strips_question_framing():
    assert normalize("What is 3 x 4 = ?") == "3*4"
