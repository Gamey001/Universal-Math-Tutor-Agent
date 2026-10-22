import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tutor.explain import allowed_numbers, explain_solution, explain_topic, unverified_numbers
from tutor.math_tool import solve_problem

MATH = solve_problem("34 + 18").to_dict()


def reply(**fields):
    base = {"asking": "Add two piles.", "picture": "Two piles of cookies.", "step_notes": ["Push the piles together."],
            "answer_note": "That's all the cookies.", "encouragement": "Nice work. Why does order not matter?"}
    return json.dumps({**base, **fields})


def test_unverified_numbers_are_detected():
    allowed = allowed_numbers("34 + 18 = 52")
    assert unverified_numbers("We have 34 and 18, making 52.", allowed) == []
    assert unverified_numbers("That's 6 groups of 9, so 54.", allowed) == ["6", "9", "54"]


def test_clean_reply_is_rendered_with_math_from_sympy():
    llm = FakeListChatModel(responses=[reply()])
    lesson = explain_solution(llm, MATH)
    assert "$$\n34 + 18 = 52\n$$" in lesson.markdown  # typeset step
    assert "\\boxed{52}" in lesson.markdown
    assert not lesson.retried and not lesson.removed_numbers


def test_invented_numbers_trigger_a_retry():
    bad = reply(picture="Make 6 groups of 9 cookies.")
    llm = FakeListChatModel(responses=[bad, reply()])
    lesson = explain_solution(llm, MATH)
    assert lesson.retried
    assert "6 groups" not in lesson.markdown


def test_numbers_still_invented_after_retry_are_removed_by_sentence():
    bad = reply(picture="Two piles of cookies. Make 6 groups of 9 cookies.")
    llm = FakeListChatModel(responses=[bad, bad])
    lesson = explain_solution(llm, MATH)
    assert lesson.removed_numbers == ["6", "9"]
    assert "Two piles of cookies." in lesson.markdown
    assert "6 groups" not in lesson.markdown


def test_garbage_reply_still_gives_a_kind_verified_lesson():
    llm = FakeListChatModel(responses=["not json at all"])
    lesson = explain_solution(llm, MATH)
    assert "\\boxed{52}" in lesson.markdown
    assert "You've got this" in lesson.markdown


def test_topic_lesson_without_proxy_allows_no_digits():
    llm = FakeListChatModel(responses=[json.dumps({"big_idea": "Think of 3 pizzas. A fraction is a slice of a whole.",
                                                    "encouragement": "Keep going."})] * 2)
    lesson = explain_topic(llm, "Fractions", None, "blurry", "Try another photo.")
    assert "3 pizzas" not in lesson.markdown
    assert "A fraction is a slice of a whole." in lesson.markdown
    assert "too blurry" in lesson.markdown and "Try another photo." in lesson.markdown
