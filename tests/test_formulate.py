from tests.conftest import FakeServices
from tutor.graph import agreed_setup, run

SAT_READ = {"legibility": "clear", "problem_text": "Find the value of x in the diagram.", "expression": "",
            "topic": "geometry", "confidence": 1.0}


def setup(*equations, target="x", choices=(25, 35, 45, 50), unknowns=None):
    return {"unknowns": unknowns or [{"name": "x", "meaning": "the marked angle at S and T"}],
            "equations": [{"equation": e, "reason": r} for e, r in equations],
            "target": target, "choices": list(choices)}


GOOD = setup(("35 + 65 + a = 180", "Angles in a triangle add up to 180"),
             ("b = a", "Vertically opposite angles are equal"),
             ("x + x + b = 180", "Angles in a triangle add up to 180"))
# Same answer, written differently (different helper names) — still agrees.
GOOD_ALT = setup(("p = 180 - 35 - 65", "Angles in a triangle add up to 180"),
                 ("2x + p = 180", "Angles in a triangle; the angle at M is shared by vertical angles"))
WRONG = setup(("35 + 65 + a = 180", "Angles in a triangle add up to 180"),
              ("x + x + 35 = 180", "Angles in a triangle add up to 180"))  # x = 72.5


def test_diagram_problem_is_solved_when_setups_agree(text_photo):
    services = FakeServices(reads={"primary": SAT_READ}, setups=[GOOD, GOOD_ALT])
    state = run({"image": text_photo}, services)
    assert state["status"] == "solved"
    assert state["math"]["answer"] == "x = 50"
    assert ("describe_image", "primary") in services.calls
    assert ("read_image", "backup") not in services.calls  # clear read: no need for a second model
    assert "Vertically opposite angles are equal" in state["explanation"]
    assert "Let x = the marked angle at S and T" in state["explanation"]


def test_disagreeing_setups_are_not_trusted(text_photo):
    services = FakeServices(reads={"primary": SAT_READ}, setups=[GOOD, WRONG])
    state = run({"image": text_photo}, services)
    assert state["status"] == "fallback"
    assert state["situation"] == "uncheckable"
    assert "math" not in state


def test_answer_must_match_a_choice():
    assert agreed_setup([{**WRONG, "result": {"ok": True, "values": {"x": "145/2 (about 72.5)"}}}] * 2) is None


def test_failed_setup_is_not_trusted():
    broken = setup(("x + y = 10", "only one equation"))
    services = FakeServices(text_extractions=[{"expression": "", "topic": "algebra",
                                               "problem_text": "Two numbers add to 10. What are they?"}],
                            setups=[broken, broken])
    state = run({"text": "Two numbers add to 10. What are they?"}, services)
    assert state["status"] == "fallback"


def test_typed_word_problem_with_two_unknowns():
    fruit = setup(("2a + 3c = 23", "Apples cost 2 and cherries 3, total 23"),
                  ("a + c = 9", "Nine pieces of fruit in total"), target="a", choices=(),
                  unknowns=[{"name": "a", "meaning": "number of apples"}, {"name": "c", "meaning": "number of cherries"}])
    question = "Apples cost $2 and cherries $3. I bought 9 fruits for $23. How many apples?"
    services = FakeServices(text_extractions=[{"expression": "", "topic": "simultaneous equations",
                                               "problem_text": question}],
                            setups=[fruit, fruit])
    state = run({"text": question}, services)
    assert state["status"] == "solved"
    assert state["math"]["answer"] == "a = 4"


def test_topic_request_does_not_trigger_setup():
    services = FakeServices(text_extractions=[{"expression": "", "topic": "fractions", "problem_text": ""}])
    state = run({"text": "I don't get fractions"}, services)
    assert state["status"] == "fallback"
    assert not any(c[0] == "formulate" for c in services.calls)


def test_learner_corrected_diagram_goes_straight_to_setup():
    services = FakeServices(setups=[GOOD, GOOD_ALT])
    state = run({"text": "Find the value of x in the diagram.", "diagram": "PT and QS cross at M. ..."}, services)
    assert state["status"] == "solved"
    assert state["math"]["answer"] == "x = 50"
    assert not any(c[0] in ("extract_from_text", "describe_image") for c in services.calls)


def test_descriptive_names_are_mapped_to_single_letters():
    from tutor.services import normalize_setup
    setup = normalize_setup({
        "unknowns": [{"name": "x", "meaning": "angle at S"}, {"name": "angle_PMQ", "meaning": "angle at M"}],
        "equations": [{"equation": "35 + 65 + angle PMQ = 180", "reason": "triangle"},
                      {"equation": "angle_PMQ = angle_SMT", "reason": "vertical angles"},
                      {"equation": "x + x + angle_SMT = 180", "reason": "triangle"}],
        "target": "x", "choices": ["25°", 50]})
    eqs = [e["equation"] for e in setup["equations"]]
    assert eqs == ["35 + 65 + y = 180", "y = z", "x + x + z = 180"]
    assert {"name": "y", "meaning": "angle at M"} in setup["unknowns"]
    assert {"name": "z", "meaning": "angle SMT"} in setup["unknowns"]
    assert setup["target"] == "x" and setup["choices"] == [25.0, 50.0]


def test_slow_pass_runs_only_when_fast_setups_disagree(text_photo):
    services = FakeServices(reads={"primary": SAT_READ}, setups=[GOOD, WRONG, GOOD, GOOD_ALT])
    state = run({"image": text_photo}, services)
    assert state["status"] == "solved"
    assert [s["think"] for s in state["setups"]] == [False, False, True, True]

    services = FakeServices(reads={"primary": SAT_READ}, setups=[GOOD, GOOD_ALT])
    state = run({"image": text_photo}, services)
    assert [s["think"] for s in state["setups"]] == [False, False]


def test_diagram_with_no_usable_setup_skips_slow_pass(text_photo):
    junk = setup(("x + y = 180", "made up"))  # underdetermined: not solvable
    services = FakeServices(reads={"primary": SAT_READ}, setups=[junk, junk, GOOD, GOOD])
    state = run({"image": text_photo}, services)
    assert state["status"] == "fallback"
    assert [s["think"] for s in state["setups"]] == [False, False]
