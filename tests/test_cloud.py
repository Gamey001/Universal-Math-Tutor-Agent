"""The cloud path, with OpenAI faked: verification labels, self-correction and the local fallback."""
import pytest

from tests.conftest import FakeServices
from tutor import cloud, config
from tutor.graph import run


def reply(**fields):
    base = {"found_math": True, "problem": "Solve 2x + 3 = 11", "problem_latex": "2x + 3 = 11", "topic": "algebra",
            "picture": "A balance scale.", "steps": [{"title": "Take 3 from both sides", "latex": "2x = 8",
                                                     "explanation": "Keep the scale level."},
                                                    {"title": "Halve both sides", "latex": "x = 4",
                                                     "explanation": "Share it into two equal groups."}],
            "final_answer": "x = 4", "final_answer_latex": "x = 4", "answer_for_check": "4",
            "check_problem": "2x + 3 = 11", "check_facts": [], "check_expression": "",
            "encouragement": "You did it."}
    return {**base, **fields}


@pytest.fixture
def fake_openai(monkeypatch):
    """Script the cloud model's replies; record how many calls were made."""
    calls = []

    def install(*replies, fail=False):
        queue = list(replies)

        def ask(messages):
            calls.append(messages)
            if fail:
                raise cloud.CloudUnavailable("APIConnectionError: network down")
            return queue.pop(0) if len(queue) > 1 else queue[0]

        monkeypatch.setattr(cloud, "_ask", ask)
        monkeypatch.setattr(config, "cloud_enabled", lambda: True)
        return calls
    return install


def test_checked_answer(fake_openai):
    fake_openai(reply())
    state = run({"text": "Solve 2x + 3 = 11"}, FakeServices())
    assert state["status"] == "solved"
    assert state["cloud"]["verified"] == "checked"
    assert "Checked with SymPy" in state["explanation"]
    assert "**Step 1: Take 3 from both sides**" in state["explanation"]
    assert "\\boxed{x = 4}" in state["explanation"]


def test_wrong_answer_is_caught_and_corrected(fake_openai):
    calls = fake_openai(reply(final_answer="x = 5", final_answer_latex="x = 5", answer_for_check="5"), reply())
    state = run({"text": "Solve 2x + 3 = 11"}, FakeServices())
    assert len(calls) == 2  # the calculator disagreed, so the model was asked to re-check once
    assert "got: x = 4" in calls[1][-1]["content"]
    assert state["cloud"]["verified"] == "corrected"
    assert state["math"]["answer"] == "x = 4"


def test_answer_that_stays_wrong_is_labelled_not_confirmed(fake_openai):
    wrong = reply(final_answer="x = 5", final_answer_latex="x = 5", answer_for_check="5")
    fake_openai(wrong, wrong)
    state = run({"text": "Solve 2x + 3 = 11"}, FakeServices())
    assert state["cloud"]["verified"] == "unchecked"
    assert "Not confirmed" in state["explanation"] and "**x = 4**" in state["explanation"]


def test_uncheckable_answer_is_labelled_honestly(fake_openai):
    fake_openai(reply(check_problem="", answer_for_check="", final_answer="the sequence converges"))
    state = run({"text": "Prove the sequence converges"}, FakeServices())
    assert state["cloud"]["verified"] == "unchecked"
    assert "Solved by the AI" in state["explanation"]


def test_value_of_expression_given_facts_is_checked(fake_openai):
    fake_openai(reply(problem="If x + y = 5 and xy = 6, find x^2 + y^2", final_answer="13",
                      final_answer_latex="13", answer_for_check="13", check_problem="",
                      check_facts=["x + y = 5", "x*y = 6"], check_expression="x^2 + y^2"))
    state = run({"text": "If x + y = 5 and xy = 6, find x^2 + y^2"}, FakeServices())
    assert state["cloud"]["verified"] == "checked"


def test_no_math_asks_for_context(fake_openai):
    fake_openai(reply(found_math=False))
    state = run({"text": "help"}, FakeServices())
    assert state["status"] == "needs_context"


def test_cloud_failure_falls_back_to_local_models(fake_openai):
    fake_openai(reply(), fail=True)
    state = run({"text": "2x + 3 = 11"}, FakeServices())
    assert state["status"] == "solved"
    assert state["math"]["answer"] == "x = 4"  # solved by the local pipeline
    assert "network down" in state["cloud_error"]


def test_blank_photo_is_never_sent_to_the_cloud(fake_openai, blank_photo):
    calls = fake_openai(reply())
    state = run({"image": blank_photo}, FakeServices())
    assert state["status"] == "needs_context"
    assert calls == []


def test_no_key_means_local_only(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "CLOUD", "auto")
    assert config.cloud_enabled() is False
