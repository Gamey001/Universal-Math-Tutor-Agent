import pytest

from tutor.math_tool import solve_problem


@pytest.mark.parametrize("text, answer", [
    ("det [[1, 2], [3, 4]]", "-2"),
    ("determinant of [[2, 0, 1], [1, 3, 2], [1, 1, 1]]", "0"),
    ("inverse of [[4, 7], [2, 6]]", "[[3/5, -7/10], [-1/5, 2/5]]"),
    ("[[1, 2], [3, 4]]^-1", "[[-2, 1], [3/2, -1/2]]"),
    ("inverse [[1, 2], [2, 4]]", "no inverse (the determinant is 0)"),
    ("[[1, 2], [3, 4]] * [[5, 6], [7, 8]]", "[[19, 22], [43, 50]]"),
    ("[[1, 2], [3, 4]] + [[1, 1], [1, 1]]", "[[2, 3], [4, 5]]"),
    ("3 [[1, 2], [3, 4]]", "[[3, 6], [9, 12]]"),
    ("[[1, 1], [0, 1]]^3", "[[1, 3], [0, 1]]"),
    ("transpose of [[1, 2, 3], [4, 5, 6]]", "[[1, 4], [2, 5], [3, 6]]"),
    ("eigenvalues of [[2, 1], [1, 2]]", "λ = 3, λ = 1"),
    ("eigenvectors of [[2, 1], [1, 2]]", "λ = 1: [-1, 1]; λ = 3: [1, 1]"),
    ("rank of [[1, 2], [2, 4]]", "1"),
    ("rref [[1, 2, 3], [4, 5, 6]]", "[[1, 0, -1], [0, 1, 2]]"),
    ("trace [[1, 2], [3, 4]]", "5"),
    ("solve [[2, 1], [1, 3]] x = [3, 5]", "x1 = 4/5 (about 0.8), x2 = 7/5 (about 1.4)"),
])
def test_matrix_answers(text, answer):
    result = solve_problem(text)
    assert result.ok, result.error
    assert result.answer == answer


def test_2x2_inverse_steps_swap_and_divide():
    steps = solve_problem("inverse of [[4, 7], [2, 6]]").steps
    assert steps[1] == "Swap a and d, and change the signs of b and c: [[6, -7], [-2, 4]]"
    assert "identity matrix ✔" in solve_problem("inverse of [[4, 7], [2, 6]]").verification


def test_product_shows_each_row_times_column():
    steps = solve_problem("[[1, 2], [3, 4]] * [[5, 6], [7, 8]]").steps
    assert "Row 1 · column 1: (1)(5) + (2)(7) = 19" in steps


@pytest.mark.parametrize("text, error", [
    ("[[1, 2], [3]]", "same number of entries"),
    ("[[1, 2], [3, 4]] * [[1, 2, 3]]", "columns must match"),
    ("[[1, 2], [3, 4]]", "What would you like to do"),
    ("det [[__import__, 2], [3, 4]]", "symbols I don't know"),
    ("det [[x, 2], [3, 4]]", "need to be numbers"),
    ("det [[1, 2, 3], [4, 5, 6]]", "square"),
])
def test_matrix_errors_are_kind_and_specific(text, error):
    result = solve_problem(text)
    assert not result.ok
    assert error in result.error
