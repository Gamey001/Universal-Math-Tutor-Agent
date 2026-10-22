import pytest

from tutor.latex import is_prose, split_description, to_latex
from tutor.math_tool import solve_problem
from tutor.topics import TOPICS

PROBLEMS = [p.expression for t in TOPICS for p in t.proxies] + [
    "x^2 + 2x + 5 = 0", "I = ∫₀¹ 4x / (1+x)³ dx", "integral of x e^x dx", "limit of sin(x)/x as x -> 0",
    "eigenvectors of [[2, 1], [1, 2]]", "polar(1 + i)", "taylor polynomial of ln(x) at x = 1 degree 3",
    "solve [[2, 1], [1, 3]] x = [3, 5]", "-3x <= 9", "5!", "10 choose 3",
]


def _balanced(latex: str) -> bool:
    depth = 0
    for i, ch in enumerate(latex):
        if ch == "{" and (i == 0 or latex[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or latex[i - 1] != "\\"):
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


@pytest.mark.parametrize("problem", PROBLEMS)
def test_every_step_converts_to_balanced_latex(problem):
    result = solve_problem(problem)
    assert result.ok, result.error
    for text in [result.answer, *result.steps]:
        latex = to_latex(text)
        assert latex and _balanced(latex), (text, latex)
        assert "$" not in latex


@pytest.mark.parametrize("text, expected", [
    ("7/8 (about 0.875)", r"\frac{7}{8}"),
    ("[[3/5, -7/10], [-1/5, 2/5]]", r"-\dfrac{7}{10}"),
    ("d/dx[x^2] = 2x", r"\frac{d}{dx}"),
    ("∫x^2 dx = x^3/3", r"\int x^{2}\, dx"),
    ("(2)^2 - 4(1)(5) = -16", "4(1)(5)"),  # substitutions keep every bracket
    ("x = 2, x = 3", r"x = 2,\quad x = 3"),
    ("λ = 3, λ = 1", r"\lambda"),
])
def test_specific_conversions(text, expected):
    assert expected in to_latex(text)


def test_words_and_math_are_separated():
    assert split_description("raise the power by 1 and divide by the new power: ∫x^2 dx = x^3/3") == \
        ("raise the power by 1 and divide by the new power", "∫x^2 dx = x^3/3")
    assert split_description("2 + 3 = 5; imaginary part: -2 + 3 = 1")[0] == ""  # math on both sides: keep together
    assert is_prose("Multiply top and bottom by the conjugate of the bottom, 3 + i, so the bottom becomes real")
    assert not is_prose("x + y = 10 using x = y + 2")


def test_never_raises_on_garbage():
    for junk in ["", "}}}{{{", "$$ \\evil", "∫∫∫", "[[", "a: b: c:"]:
        assert isinstance(to_latex(junk), str)
