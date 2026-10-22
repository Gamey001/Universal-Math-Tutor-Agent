import pytest

from tutor.graph import ServiceUnavailable, run
from tests.conftest import FakeServices

CLEAR = {"legibility": "clear", "problem_text": "2x + 3 = 11", "expression": "2*x + 3 = 11",
         "topic": "linear equations", "confidence": 0.95}
BLURRY = {"legibility": "partial", "problem_text": "3/? + ?/8", "expression": "",
          "topic": "fractions", "confidence": 0.2}


# --- Condition A: clear photo -------------------------------------------------

def test_clear_photo_is_solved_without_backup(text_photo):
    services = FakeServices(reads={"primary": CLEAR})
    state = run({"image": text_photo}, services)
    assert state["status"] == "solved"
    assert state["math"]["answer"] == "x = 4"
    assert state["explanation"]
    assert ("read_image", "backup") not in services.calls


def test_backup_cannot_overrule_primary_judging_photo_unclear(text_photo):
    from tutor.graph import unconfirmed_readings
    services = FakeServices(reads={"primary": BLURRY, "backup": CLEAR})
    state = run({"image": text_photo}, services)
    assert state["status"] == "fallback"
    assert unconfirmed_readings(state) == ["2*x + 3 = 11"]  # offered to the learner to confirm
    assert [a["model"] for a in state["attempts"]] == ["primary", "backup"]


def test_backup_solves_when_primary_could_not_run(text_photo):
    services = FakeServices(reads={"primary": RuntimeError("model not found"), "backup": CLEAR})
    assert run({"image": text_photo}, services)["status"] == "solved"


def test_low_confidence_numbers_are_not_solved(text_photo):
    guess = {**CLEAR, "confidence": 0.3}
    state = run({"image": text_photo}, FakeServices(reads={"primary": guess, "backup": guess}))
    assert state["status"] == "fallback"  # topic known, numbers not trusted


def test_clear_word_problem_gets_expression_from_chat_model(text_photo):
    word = {"legibility": "clear", "problem_text": "Sam has 3 bags of 4 apples. How many apples?",
            "expression": "", "topic": "", "confidence": 0.9}
    services = FakeServices(reads={"primary": word},
                            text_extractions=[{"expression": "3*4", "topic": "multiplication", "problem_text": ""}])
    state = run({"image": text_photo}, services)
    assert state["status"] == "solved"
    assert state["math"]["answer"] == "12"


# --- Condition B: blurry photo, topic visible ---------------------------------

def test_blurry_photo_falls_back_to_verified_proxy(blurry_photo):
    services = FakeServices(reads={"primary": BLURRY, "backup": BLURRY})
    state = run({"image": blurry_photo}, services)
    assert state["status"] == "fallback"
    assert state["topic"] == "Fractions"
    assert state["proxy"]["math"]["ok"]
    assert ("make_proxy", "Fractions") not in services.calls  # known topic: template proxy


def test_unknown_topic_uses_llm_proxy_only_if_it_checks_out(blurry_photo):
    calculus = {**BLURRY, "topic": "topology"}
    services = FakeServices(reads={"primary": calculus, "backup": calculus},
                            proxies=[{"problem_text": "Bad", "expression": "d/dx of what?"}])
    state = run({"image": blurry_photo}, services)
    assert state["status"] == "fallback"
    assert state["proxy"] is None  # the unverifiable proxy was dropped


def test_topic_is_ignored_when_model_could_not_read_anything(blurry_photo):
    guess = {"legibility": "unreadable", "topic": "geometry"}
    state = run({"image": blurry_photo}, FakeServices(reads={"primary": guess, "backup": guess}))
    assert state["status"] == "needs_context"


def test_topic_from_either_model_is_kept(blurry_photo):
    services = FakeServices(reads={"primary": {**BLURRY, "topic": ""}, "backup": BLURRY})
    assert run({"image": blurry_photo}, services)["topic"] == "Fractions"


# --- Condition C: nothing usable -> ask the learner ---------------------------

def test_blank_photo_asks_for_context_without_calling_models(blank_photo):
    services = FakeServices()
    state = run({"image": blank_photo}, services)
    assert state["status"] == "needs_context"
    assert "blank" in state["message"]
    assert services.calls == []


def test_unreadable_photo_asks_for_context_after_both_models(blurry_photo):
    services = FakeServices(reads={"primary": {}, "backup": {}})
    state = run({"image": blurry_photo}, services)
    assert state["status"] == "needs_context"
    assert [c[1] for c in services.calls if c[0] == "read_image"] == ["primary", "backup"]
    assert "which area of math" in state["message"]
    assert "explanation" not in state


def test_non_image_upload_asks_for_context():
    state = run({"image": b"not an image"}, FakeServices())
    assert state["status"] == "needs_context"


def test_vague_text_asks_for_context():
    state = run({"text": "help"}, FakeServices())
    assert state["status"] == "needs_context"


# --- Typed questions and replies to "ask for context" -------------------------

def test_typed_equation_is_solved_without_llm_parsing():
    services = FakeServices()
    state = run({"text": "3/4 + 1/8"}, services)
    assert state["status"] == "solved"
    assert state["math"]["answer"].startswith("7/8")
    assert not any(c[0] == "extract_from_text" for c in services.calls)


def test_typed_word_problem_is_translated_then_verified():
    services = FakeServices(text_extractions=[{"expression": "24 / 6", "topic": "division", "problem_text": ""}])
    state = run({"text": "24 cookies shared by 6 friends?"}, services)
    assert state["math"]["answer"] == "4"


def test_typed_topic_gives_lesson():
    services = FakeServices(text_extractions=[{"expression": "", "topic": "fractions", "problem_text": ""}])
    state = run({"text": "I don't get fractions"}, services)
    assert state["status"] == "fallback"
    assert state["proxy"]["math"]["ok"]


def test_topic_button_goes_straight_to_lesson():
    services = FakeServices()
    state = run({"requested_topic": "Percentages"}, services)
    assert state["status"] == "fallback"
    assert state["topic"] == "Percentages"
    assert services.calls == []


# --- Infrastructure failures are not blamed on the photo ----------------------

def test_ollama_down_raises_instead_of_blaming_photo(text_photo):
    down = ConnectionError("connection refused")
    with pytest.raises(ServiceUnavailable):
        run({"image": text_photo}, FakeServices(reads={"primary": down, "backup": down}))


def test_missing_backup_model_still_uses_primary(blurry_photo):
    services = FakeServices(reads={"primary": BLURRY, "backup": RuntimeError("model not found")})
    assert run({"image": blurry_photo}, services)["status"] == "fallback"


# --- Hallucination guards -----------------------------------------------------

def test_reading_contradicted_by_transcription_keeps_topic_but_is_not_solved(blurry_photo):
    misread = {**CLEAR, "problem_text": "34 + 18", "expression": "34 + 18", "topic": "fractions"}
    services = FakeServices(reads={"primary": misread, "backup": misread},
                            transcriptions={"primary": "3/? + ?/8", "backup": "3/? + ?/8"})
    state = run({"image": blurry_photo}, services)
    assert state["status"] == "fallback"
    assert state["situation"] == "blurry"
    assert "math" not in state


def test_confident_read_of_unreadable_photo_is_discarded_entirely(blurry_photo):
    invented = {**CLEAR, "problem_text": "200 ÷ 20 × 100", "expression": "200 / 20 * 100", "topic": "arithmetic"}
    services = FakeServices(reads={"primary": invented, "backup": invented},
                            transcriptions={"primary": "UNREADABLE", "backup": "UNREADABLE"})
    state = run({"image": blurry_photo}, services)
    assert state["status"] == "needs_context"  # neither the numbers nor the topic are trusted


def test_unconfirmed_reading_is_offered_back_to_learner(blurry_photo):
    from tutor.graph import unconfirmed_readings
    invented = {**CLEAR, "expression": "36 + 48 / 6", "topic": ""}
    services = FakeServices(reads={"primary": invented, "backup": RuntimeError("can't load")},
                            transcriptions={"primary": "200"})
    state = run({"image": blurry_photo}, services)
    assert state["status"] == "needs_context"
    assert unconfirmed_readings(state) == ["36 + 48 / 6"]


def test_transcription_with_answer_appended_still_agrees(text_photo):
    services = FakeServices(reads={"primary": {**CLEAR, "expression": "34 + 18"}},
                            transcriptions={"primary": "34 + 18 = 52"})
    assert run({"image": text_photo}, services)["status"] == "solved"


def test_explanation_contains_verified_answer_and_steps():
    state = run({"text": "2x + 3 = 11"}, FakeServices())
    md = state["explanation"]
    assert "**x = 4**" in md
    assert "`2x = 8`" in md and "Keep the scale level." in md
    assert "Checked with SymPy" in md


# --- Photos of calculus -------------------------------------------------------

INTEGRAL_READ = {"legibility": "clear", "problem_text": "I = ∫₀¹ 4x / (1+x)³ dx\nLet u = 1+x ...",
                 "expression": "I = ∫₀¹ 4x / (1+x)³ dx", "topic": "Calculus", "confidence": 1.0}


def test_photo_of_definite_integral_is_solved(text_photo):
    services = FakeServices(reads={"primary": INTEGRAL_READ},
                            transcriptions={"primary": "I = ∫₀¹ 4x/(1+x)³ dx\nLet u = 1+x ..."})
    state = run({"image": text_photo}, services)
    assert state["status"] == "solved"
    assert state["math"]["answer"] == "1/2"


def test_restating_never_drops_the_integral(text_photo):
    unparsable = {**INTEGRAL_READ, "expression": "I = ∫ weird ⌠ notation dx"}
    services = FakeServices(reads={"primary": unparsable, "backup": unparsable},
                            text_extractions=[{"expression": "4*x/(x+1)**3", "topic": "calculus", "problem_text": ""}] * 2)
    state = run({"image": text_photo}, services)
    assert state.get("math", {}).get("answer") != "4x/(x + 1)^3"
    assert state["status"] == "fallback"  # a calculus lesson, not a fake "simplified" answer
    assert not any(c[0] == "formulate" for c in services.calls)


def test_bare_expression_from_photo_is_not_an_answer(text_photo):
    bare = {"legibility": "clear", "problem_text": "4x/(x+1)^3", "expression": "4x/(x+1)^3",
            "topic": "algebra", "confidence": 1.0}
    state = run({"image": text_photo}, FakeServices(reads={"primary": bare, "backup": bare}))
    assert state["status"] != "solved"


def test_photo_asking_to_expand_is_still_solved(text_photo):
    ask = {"legibility": "clear", "problem_text": "Expand (x+1)^2", "expression": "(x+1)^2",
           "topic": "algebra", "confidence": 1.0}
    state = run({"image": text_photo}, FakeServices(reads={"primary": ask}, transcriptions={"primary": "Expand (x+1)^2"}))
    assert state["status"] == "solved"
