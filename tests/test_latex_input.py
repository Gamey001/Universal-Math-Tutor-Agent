"""Math-keyboard LaTeX -> solver notation -> verified answer, for every kind of input the keyboard offers."""
import pytest

from tutor.latex_input import LatexInputError, latex_to_problem
from tutor.math_tool import solve_problem


@pytest.mark.parametrize("latex, answer", [
    (r"\int_{0}^{1}\frac{4x}{(1+x)^{3}}\,dx", "1/2"),
    (r"\int x e^{x}\,dx", "x·e^(x) - e^(x) + C"),
    (r"\frac{d}{dx}\left(x^{2}\sin x\right)", "f'(x) = x^2·cos(x) + 2x·sin(x)"),
    (r"\frac{d^{2}}{dx^{2}}x^{3}", "f''(x) = 6x"),
    (r"\frac{\partial}{\partial x}x^{2}y", "f'(x) = 2x·y"),
    (r"\lim_{x\to0}\frac{\sin x}{x}", "1"),
    (r"\lim_{x\to0^{+}}\frac{1}{x}", "∞"),
    (r"\lim_{n\to\infty}\left(1+\frac{2}{n}\right)^{n}", "e^(2)"),
    (r"\sum_{k=1}^{100}k", "5050"),
    (r"\prod_{k=1}^{5}k", "120"),
    (r"\frac{dy}{dx}=3y", "y = C1·e^(3x) (C1 is a constant: any value works)"),
    (r"x^{2}-5x+6=0", "x = 2, x = 3"),
    (r"\frac{3}{4}+\frac{1}{8}", "7/8 (about 0.875)"),
    (r"\sqrt[3]{27}", "3"),
    (r"\binom{10}{3}", "120"),
    (r"|-5|+2", "7"),
    (r"\begin{pmatrix}4&7\\2&6\end{pmatrix}^{-1}", "[[3/5, -7/10], [-1/5, 2/5]]"),
    (r"\begin{vmatrix}1&2\\3&4\end{vmatrix}", "-2"),
    (r"2x+3\le11", "x <= 4"),
    (r"\log_{2}x=5", "x = 32"),
    (r"(2+3i)(1-i)", "5 + i"),
    (r"\cos^{2}x+\sin^{2}x", "1"),
    (r"\int_{0}^{\pi}\frac{8x}{4\cos^{2}x+\sin^{2}x}\,dx", "2pi^2 (about 19.7392)"),
])
def test_keyboard_input_is_solved(latex, answer):
    result = solve_problem(latex_to_problem(latex))
    assert result.ok, (latex_to_problem(latex), result.error)
    assert result.answer == answer


def test_greek_letters_are_variables():
    assert solve_problem(latex_to_problem(r"\sin\theta=\frac{1}{2}")).answer.startswith("theta = pi/6")


@pytest.mark.parametrize("latex, message", [
    ("", "nothing to solve"),
    (r"\frac{1}{2", "isn't closed"),
    (r"2\int x\,dx", "at the start"),
    (r"\int x^{2}", "dx"),
    (r"\weirdsymbol x", "don't know the symbol"),
])
def test_unusable_input_gets_a_kind_message(latex, message):
    with pytest.raises(LatexInputError, match=message):
        latex_to_problem(latex)


def test_sum_steps_show_the_substitution():
    steps = solve_problem("sum of k for k from 1 to 100").steps
    assert steps[0].endswith("n·(n + 1)/2") and "100(1 + 100)/2 = 5050" in steps[1]


def test_infinite_sum_is_checked_with_partial_sums():
    result = solve_problem("sum of 1/n^2 for n from 1 to oo")
    assert result.answer.startswith("pi^2/6") and "20,000 terms" in result.verification


def test_definite_integral_without_antiderivative_is_found_numerically_and_says_so():
    result = solve_problem("integral from 0 to pi of 8x/(4cos(x)^2 + sin(x)^2) dx")
    assert "measure the area numerically" in result.steps[0] and "30 digits" in result.verification
