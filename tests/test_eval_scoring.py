"""The evaluation scorer must be right, or every score it reports is meaningless."""
import json
from pathlib import Path

import pytest

from eval.scoring import check_answer, score


@pytest.mark.parametrize("answer, expect, ok", [
    ("x = 3, x = 2", {"values": ["2", "3"]}, True),
    ("x = 2", {"values": ["2", "3"]}, False),
    ("7/8 (about 0.875)", {"values": ["7/8"]}, True),
    ("x = -1 - 2i, x = -1 + 2i", {"values": ["-1+2*I", "-1-2*I"]}, True),
    ("[[3/5, -7/10], [-1/5, 2/5]]", {"matrix": [["3/5", "-7/10"], ["-1/5", "2/5"]]}, True),
    ("λ = 3, λ = 1", {"values": ["1", "3"]}, True),
    ("f'(x) = x^2·cos(x) + 2x·sin(x)", {"values": ["x**2*cos(x) + 2*x*sin(x)"]}, True),
    ("x·e^(x) - e^(x) + C", {"antiderivative_of": "x*exp(x)"}, True),
    ("x·e^(x) + C", {"antiderivative_of": "x*exp(x)"}, False),
    ("y = 3e^(2x)", {"values": ["3*exp(2*x)"]}, True),
    ("2(p + q)", {"values": ["2*p + 2*q"]}, True),
    ("x >= -3", {"text": ["x >= -3"]}, True),
    # the whiteboard's own (wrong) final answer must be scored wrong
    ("x^4·ln(x^3 + 1)/4 - x^4/16 + 3x/4 - ln(x + 1)/4 + ln(x^2 - x + 1)/8 - sqrt(3)·atan(sqrt(3)·(2x - 1)/3)/4 + C",
     {"antiderivative_of": "x**3*log(x**3 + 1)"}, False),
])
def test_answers_are_compared_by_meaning(answer, expect, ok):
    assert check_answer(answer, expect) is ok


def test_outcomes():
    assert score({"status": "solved", "math": {"answer": "x = 4"}}, {"decline": True}) == "wrong"
    assert score({"status": "needs_context"}, {"decline": True}) == "correct"
    assert score({"status": "fallback"}, {"values": ["4"]}) == "declined"
    assert score(None, {"values": ["4"]}, error="boom") == "error"


def test_every_case_is_well_formed():
    cases = json.loads(Path("eval/cases.json").read_text())["cases"]
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert set(c["input"]) <= {"text", "image"} and c["expect"], c["id"]
        if "image" in c["input"] and not c["input"]["image"].endswith((".jpeg", "_screenshot.png")) \
                and c["input"]["image"] not in ("images/whiteboard_x3_ln.png", "images/pq_expression.png"):
            # third-party photos are gitignored (kept locally only); our own test images must exist
            assert (Path("eval") / c["input"]["image"]).exists(), c["id"]
