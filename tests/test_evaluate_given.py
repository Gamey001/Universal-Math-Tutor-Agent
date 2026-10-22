import pytest

from tests.conftest import FakeServices
from tutor.graph import run
from tutor.math_tool import evaluate_given

GIVEN = [("p(x+y)^2 = 5", "given"), ("q(x-y)^2 = 3", "given")]
TARGET = "p^2(x+y)^2 + 4*p*q*x*y - q^2(x-y)^2"


@pytest.mark.parametrize("facts, target, answer", [
    (GIVEN, TARGET, "2(p + q)"),
    ([("x + y = 5", "sum"), ("x*y = 6", "product")], "x^2 + y^2", "13"),
    ([("a + b = 7", "sum"), ("a*b = 10", "product")], "a^3 + b^3", "133"),
    ([("x + 1/x = 3", "given")], "x^2 + 1/x^2", "7"),
    ([("a = 2", "a"), ("b = 3", "b")], "a^2 + b", "7"),
])
def test_values_from_given_facts(facts, target, answer):
    result = evaluate_given(facts, target)
    assert result.ok, result.error
    assert result.answer == answer
    assert "✔" in result.verification or "same for every solution" in result.verification


def test_textbook_route_names_the_known_quantities():
    steps = evaluate_given([("x + y = 5", "sum"), ("x*y = 6", "product")], "x^2 + y^2").steps
    assert steps[1].endswith("x^2 + y^2 = A^2 - 2B")


def test_facts_that_do_not_help_are_reported():
    assert not evaluate_given([("x + y = 5", "sum")], "x*z").ok


def test_letter_products_are_split_in_setups():
    from tutor.services import normalize_setup
    setup = normalize_setup({"unknowns": [{"name": "p"}, {"name": "q"}, {"name": "x"}, {"name": "y"}],
                             "equations": [{"equation": "p(x+y)^2 = 5", "reason": "given"}],
                             "target": "", "target_expression": "4pqxy", "choices": []})
    assert setup["target_expression"] == "4p*q*x*y"


def test_photo_asking_for_an_expression_value(text_photo):
    read = {"legibility": "clear", "expression": "p*(x+y)^2 = 5; q*(x-y)^2 = 3", "topic": "algebra",
            "confidence": 1.0, "problem_text": "If p(x+y)^2 = 5, and q(x-y)^2 = 3, then simplified value of "
                                              "the expression p^2(x+y)^2 + 4pqxy - q^2(x-y)^2 is"}
    setup = {"unknowns": [], "equations": [{"equation": e, "reason": r} for e, r in GIVEN],
             "target": "", "target_expression": TARGET, "choices": []}
    services = FakeServices(reads={"primary": read}, setups=[setup, setup],
                            transcriptions={"primary": read["problem_text"]})
    state = run({"image": text_photo}, services)
    assert state["status"] == "solved"
    assert state["math"]["answer"] == "2(p + q)"
