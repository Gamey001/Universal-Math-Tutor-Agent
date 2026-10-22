"""The LangGraph pipeline.

    image ─► precheck ─┬─ blank/unopenable ─────────────────────────► ask_context
                       └─► read_primary ─┬─ clear & confirmed ──────► solve ──► tutor
                                         ├─ clear word/diagram ─────► formulate
                                         └─► read_backup ─┐
    text ──► read_text ───────────────────────────────────┤
                                                          ├─ clear & confirmed ► solve ──► tutor
                                                          ├─ word/diagram ─────► formulate ─┬─ setups agree ► tutor
                                                          │                                 └─► fallback / ask_context
                                                          ├─ topic known ──────► fallback ► tutor
                                                          └─ nothing usable ───► ask_context
    topic ─────────────────────────────────────────────────────────► fallback ► tutor

A photo only counts as "clear" when two independent reads agree on its
numbers (a structured read and a plain transcription), because small
vision models will confidently invent numbers for blurry photos.

Word and diagram problems have no single expression to solve. ``formulate``
asks the reasoning model for equations (each with its reason) several times
independently; SymPy solves each setup, and the answer is only trusted when
every setup reaches the same value (and it matches an answer choice, if any).

``ask_context`` ends the run with a question for the learner. Their answer
comes back as a new run with ``text`` (a typed problem or topic) or
``requested_topic`` (a topic button), so the conversation picks up where it
left off.
"""

from __future__ import annotations

import operator
import re
from collections import Counter
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import config, prompts
from .explain import explain_solution, explain_topic
from .images import detect_image_issue, prepare_for_model
import sympy as sp

from .calculus_tool import looks_like_calculus
from .math_tool import normalize, solve_problem, solve_system
from .services import OllamaServices, TutorServices, normalize_extraction
from .topics import match_topic, pick_proxy


class TutorState(TypedDict, total=False):
    # Inputs: exactly one of these starts a run.
    image: bytes
    text: str  # optionally with ``diagram``: a learner-corrected diagram description
    requested_topic: str

    source: str  # "image" | "text" | "topic"
    image_issue: str | None
    attempts: Annotated[list[dict], operator.add]  # every vision read, for the "what I saw" panel
    extraction: dict
    math: dict
    topic: str
    proxy: dict | None
    situation: str  # for fallback lessons: "blurry" | "uncheckable" | "topic"
    status: str  # "solved" | "fallback" | "needs_context"
    message: str  # the question shown when status == "needs_context"
    explanation: str
    guard: dict  # what the number guard did to the tutor's prose
    diagram: str  # the vision model's description of a diagram, for word/diagram problems
    setups: list[dict]  # every independent setup attempt and its SymPy result
    setup: dict  # the agreed setup that produced ``math``


class ServiceUnavailable(RuntimeError):
    """Raised when no model could be reached at all (e.g. Ollama isn't running)."""


def _numbers(text: str) -> Counter:
    return Counter(re.findall(r"\d+(?:\.\d+)?", normalize(text)))


def readings_agree(expression: str, transcription: str) -> bool:
    """True if every number in ``expression`` also appears in the independent transcription.

    The transcription may contain extra numbers (e.g. the model wrote the answer
    "= 52" after the problem), but nothing in the expression may be missing.
    """
    if not transcription or "UNREADABLE" in transcription.upper():
        return False
    needed = _numbers(expression)
    return bool(needed) and not (needed - _numbers(transcription))


def can_solve(extraction: dict | None) -> bool:
    """Only solve when the numbers were read clearly and SymPy can check them."""
    if not extraction or not extraction.get("expression"):
        return False
    if extraction.get("legibility") != "clear":
        return False
    if extraction.get("confidence", 0) < config.MIN_CONFIDENCE:
        return False
    return solve_problem(extraction["expression"]).ok


_LINE_LABEL = re.compile(r"^\s*(?:solution|question|problem|task|exercise|q\d*|\d+[.)])\s*[:.)\-]?\s*", re.IGNORECASE)


def problem_lines(text: str, limit: int = 4) -> list[str]:
    """The first few written lines, without labels like "Solution:" or "1)".

    A photo of a worked solution has no single "expression" for a model to fill in,
    but the problem is almost always written first: "Solution: ∫x³·ln(x³+1) dx"."""
    lines = []
    for raw in (text or "").splitlines()[:8]:
        line = _LINE_LABEL.sub("", raw).strip()
        if line and not re.match(r"step\b", line, re.IGNORECASE):
            lines.append(line)
    return lines[:limit]


def _is_bare_expression(expression: str, problem_text: str) -> bool:
    """An expression with a variable but no question (no =, not calculus, no 'simplify/expand/factor')."""
    if not expression:
        return False
    result = solve_problem(expression)
    asked = re.search(r"simplif|expand|factori[sz]e|factor\b", problem_text or "", re.IGNORECASE)
    return result.ok and result.kind == "simplify" and not asked


def needs_setup(extraction: dict | None) -> bool:
    """A clearly read, specific problem with no directly solvable expression (word or diagram problem)."""
    if not extraction or extraction.get("legibility") != "clear":
        return False
    if not extraction.get("problem_text") or not extraction.get("specific", True):
        return False
    if looks_like_calculus(extraction["problem_text"]):
        return False  # setups write equations; calculus that didn't parse goes to a lesson instead
    return not can_solve(extraction)


def _as_number(value: str) -> float | None:
    try:  # values come from our own math tool, e.g. "7/8 (about 0.875)"
        return float(sp.N(sp.sympify(value.split(" (")[0])))
    except Exception:
        return None


def agreed_setup(setups: list[dict]) -> dict | None:
    """The setup to trust, or None. Every attempt must succeed and agree on the target's value,
    and that value must match an answer choice when the problem has choices."""
    if not setups or any(not s.get("result", {}).get("ok") for s in setups):
        return None
    targets = [_as_number(s["result"]["values"].get(s["target"], "")) for s in setups]
    if any(t is None for t in targets) or any(abs(t - targets[0]) > 1e-9 for t in targets):
        return None
    choices = [c for s in setups for c in s.get("choices", [])]
    if choices and not any(abs(c - targets[0]) < 1e-6 for c in choices):
        return None
    return setups[0]


def _rank(extraction: dict) -> tuple:
    return (can_solve(extraction), bool(extraction.get("topic")), extraction.get("confidence", 0))


def build_graph(services: TutorServices | None = None):
    services = services or OllamaServices()

    def _read_with(model: str, image: bytes) -> dict:
        attempt = {"model": model, "error": None, "transcription": None, "agreed": None}
        try:
            extraction = services.read_image(model, image)
        except Exception as exc:  # model missing, can't load, server down...
            return {**attempt, "error": str(exc), "extraction": normalize_extraction({})}
        if extraction["legibility"] == "unreadable":
            extraction["topic"] = ""  # a topic for a photo it couldn't read is a guess

        # A clear word problem may come back without a usable expression; let the
        # chat model translate the words. (Never for blurry reads: numbers would be guesses.)
        if (extraction["legibility"] == "clear" and extraction["problem_text"]
                and not solve_problem(extraction["expression"]).ok):
            # First, deterministically: is one of the first written lines the problem itself?
            for line in problem_lines(extraction["problem_text"]):
                if solve_problem(line).ok and not _is_bare_expression(line, extraction["problem_text"]):
                    extraction["expression"] = line
                    break
        if (extraction["legibility"] == "clear" and extraction["problem_text"]
                and not solve_problem(extraction["expression"]).ok):
            try:
                from_text = services.extract_from_text(extraction["problem_text"])
                restated = from_text["expression"]
                # Restating can silently drop the operation (an integral becoming just its
                # integrand), so it may only fill a gap, never replace calculus we did read.
                if restated and not looks_like_calculus(extraction["expression"]) and not _is_bare_expression(
                        restated, extraction["problem_text"]):
                    extraction["expression"] = restated
                extraction["topic"] = extraction["topic"] or from_text["topic"]
            except Exception:
                pass
        if _is_bare_expression(extraction["expression"], extraction["problem_text"]):
            # "4x/(x+1)^3" alone isn't a question: re-printing it as "the answer" would be nonsense.
            extraction = {**extraction, "expression": ""}

        # Second, independent read before trusting any numbers.
        if can_solve(extraction):
            try:
                attempt["transcription"] = services.transcribe_image(model, image)
            except Exception:
                attempt["transcription"] = ""
            attempt["agreed"] = readings_agree(extraction["expression"], attempt["transcription"])
            if not attempt["agreed"]:
                extraction = {**extraction, "legibility": "partial", "unconfirmed": extraction["expression"]}
                if not attempt["transcription"] or "UNREADABLE" in attempt["transcription"].upper():
                    # The model claimed a clear read of a photo it can't actually read:
                    # its guessed topic is no more trustworthy than its guessed numbers.
                    extraction.update(legibility="unreadable", topic="")
        return {**attempt, "extraction": extraction}

    # ---- nodes -------------------------------------------------------------

    def precheck(state: TutorState) -> dict:
        return {"source": "image", "image_issue": detect_image_issue(state["image"])}

    def read_primary(state: TutorState) -> dict:
        attempt = _read_with(services.vision_model, prepare_for_model(state["image"]))
        return {"attempts": [attempt], "extraction": attempt["extraction"]}

    def read_backup(state: TutorState) -> dict:
        attempts = list(state.get("attempts", []))
        new = []
        if services.backup_vision_model:
            new.append(_read_with(services.backup_vision_model, prepare_for_model(state["image"])))
            primary_ran = any(not a["error"] for a in attempts)
            backup = new[-1]["extraction"]
            if primary_ran and can_solve(backup):
                # The primary model already judged this photo unclear; a second model's
                # confident read can't overrule that. Offer it to the learner to confirm instead.
                new[-1]["extraction"] = {**backup, "legibility": "partial", "unconfirmed": backup["expression"]}
        attempts += new
        if all(a["error"] for a in attempts):
            raise ServiceUnavailable(attempts[0]["error"])
        candidates = [a["extraction"] for a in attempts if not a["error"]]
        best = dict(max(candidates, key=_rank))
        if not best["topic"]:  # keep a topic either model spotted
            best["topic"] = next((c["topic"] for c in candidates if c["topic"]), "")
        return {"attempts": new, "extraction": best}

    def read_text(state: TutorState) -> dict:
        text = state["text"].strip()
        if state.get("diagram"):  # the learner corrected a diagram description: go straight to setup
            return {"source": "text", "extraction": {"legibility": "clear", "problem_text": text, "expression": "",
                                                     "topic": "", "confidence": 1.0, "specific": True}}
        direct = solve_problem(text)
        if direct.ok:  # typed something SymPy already understands, e.g. "2x + 3 = 11"
            extraction = {"legibility": "clear", "problem_text": text, "expression": text,
                          "topic": "", "confidence": 1.0}
        else:
            try:
                parsed = services.extract_from_text(text)
            except Exception as exc:
                raise ServiceUnavailable(str(exc)) from exc
            extraction = {"legibility": "clear", "problem_text": parsed["problem_text"] or text,
                          "expression": parsed["expression"], "topic": parsed["topic"], "confidence": 1.0,
                          # a topic request ("I don't get fractions") has neither of these
                          "specific": bool(parsed["problem_text"] or parsed["expression"])}
        return {"source": "text", "extraction": extraction}

    def solve(state: TutorState) -> dict:
        extraction = state["extraction"]
        result = solve_problem(extraction["expression"])
        return {"status": "solved", "math": result.to_dict(), "topic": extraction.get("topic", "")}

    def formulate(state: TutorState) -> dict:
        extraction = state["extraction"]
        diagram = state.get("diagram") or ""  # set when the learner corrected it
        if not diagram and state.get("source") == "image":
            try:
                diagram = services.describe_image(services.vision_model, prepare_for_model(state["image"]))
            except Exception:
                pass
        setups, chosen = [], None
        for think in (False, True):  # fast pass first; think hard only if those disagree
            batch = []
            for _ in range(config.SETUP_ATTEMPTS):
                try:
                    setup = services.formulate(extraction["problem_text"], diagram, think=think)
                except Exception as exc:
                    batch.append({"error": str(exc), "result": {"ok": False}, "think": think})
                    continue
                result = solve_system([(e["equation"], e["reason"]) for e in setup["equations"]],
                                      setup["target"] or None)
                batch.append({**setup, "result": result.to_dict(), "think": think})
            setups += batch
            chosen = agreed_setup(batch)
            if chosen:
                break
            if diagram and not any(s["result"].get("ok") for s in batch):
                # No setup was even solvable: the diagram description is probably wrong, and
                # thinking harder about a wrong picture only burns minutes. Ask the learner instead.
                break
        update = {"diagram": diagram, "setups": setups}
        if chosen:
            update.update(status="solved", math=chosen["result"], setup=chosen,
                          topic=extraction.get("topic", ""))
        return update

    def fallback(state: TutorState) -> dict:
        extraction = state.get("extraction") or {}
        source = state.get("source") or "topic"
        raw_topic = state.get("requested_topic") or extraction.get("topic", "")
        known = match_topic(raw_topic)
        topic_name = known.name if known else raw_topic

        if source == "image" and extraction.get("legibility") != "clear":
            situation = "blurry"
        elif source in ("image", "text") and (extraction.get("expression") or state.get("setups")):
            situation = "uncheckable"
        else:
            situation = "topic"

        candidate = None
        if known:
            p = pick_proxy(known)
            candidate = {"problem_text": p.problem_text, "expression": p.expression}
        else:
            try:
                candidate = services.make_proxy(topic_name)
            except Exception:
                pass
        proxy = None
        if candidate and candidate.get("problem_text"):
            checked = solve_problem(candidate.get("expression", ""))
            if checked.ok:  # a practice problem we can't verify is worse than none
                proxy = {**candidate, "math": checked.to_dict()}

        return {"source": source, "status": "fallback", "topic": topic_name,
                "proxy": proxy, "situation": situation}

    def ask_context(state: TutorState) -> dict:
        issue = state.get("image_issue")
        if issue in ("blank", "unopenable"):
            reason = issue
        elif state.get("source") == "image":
            reason = "unreadable"
        else:
            reason = "text"
        message = f"{prompts.ASK_CONTEXT_MESSAGES[reason]}\n\n{prompts.ASK_CONTEXT_FOLLOW_UP}"
        return {"status": "needs_context", "message": message}

    def tutor(state: TutorState) -> dict:
        if state["status"] == "solved":
            original = state.get("extraction", {}).get("problem_text", "")
            if state.get("diagram"):
                original += "\n" + state["diagram"]
            legend = [f"{u['name']} = {u['meaning']}" for u in (state.get("setup") or {}).get("unknowns", [])
                      if u.get("meaning")]
            lesson = explain_solution(services.tutor_llm, state["math"], original, legend)
        else:
            invitation = {"blurry": prompts.INVITE_RETAKE, "uncheckable": prompts.INVITE_REWRITE,
                          "topic": prompts.INVITE_SPECIFIC}[state["situation"]]
            lesson = explain_topic(services.tutor_llm, state["topic"], state.get("proxy"),
                                   state["situation"], invitation)
        return {"explanation": lesson.markdown,
                "guard": {"retried": lesson.retried, "removed_numbers": lesson.removed_numbers}}

    # ---- routing -----------------------------------------------------------

    def route_start(state: TutorState) -> str:
        if state.get("image"):
            return "precheck"
        if state.get("requested_topic"):
            return "fallback"
        if (state.get("text") or "").strip():
            return "read_text"
        return "ask_context"

    def after_precheck(state: TutorState) -> str:
        return "ask_context" if state.get("image_issue") else "read_primary"

    def after_primary(state: TutorState) -> str:
        if can_solve(state["extraction"]):
            return "solve"
        return "formulate" if needs_setup(state["extraction"]) else "read_backup"

    def after_read(state: TutorState) -> str:
        extraction = state["extraction"]
        if can_solve(extraction):
            return "solve"
        if needs_setup(extraction):
            return "formulate"
        if extraction.get("topic"):
            return "fallback"
        return "ask_context"

    def after_formulate(state: TutorState) -> str:
        if state.get("status") == "solved":
            return "tutor"
        return "fallback" if state["extraction"].get("topic") else "ask_context"

    graph = StateGraph(TutorState)
    for name, node in [("precheck", precheck), ("read_primary", read_primary), ("read_backup", read_backup),
                       ("read_text", read_text), ("solve", solve), ("formulate", formulate), ("fallback", fallback),
                       ("ask_context", ask_context), ("tutor", tutor)]:
        graph.add_node(name, node)

    graph.add_conditional_edges(START, route_start, ["precheck", "fallback", "read_text", "ask_context"])
    graph.add_conditional_edges("precheck", after_precheck, ["ask_context", "read_primary"])
    graph.add_conditional_edges("read_primary", after_primary, ["solve", "formulate", "read_backup"])
    graph.add_conditional_edges("read_backup", after_read, ["solve", "formulate", "fallback", "ask_context"])
    graph.add_conditional_edges("read_text", after_read, ["solve", "formulate", "fallback", "ask_context"])
    graph.add_conditional_edges("formulate", after_formulate, ["tutor", "fallback", "ask_context"])
    graph.add_edge("solve", "tutor")
    graph.add_edge("fallback", "tutor")
    graph.add_edge("tutor", END)
    graph.add_edge("ask_context", END)
    return graph.compile()


def unconfirmed_readings(state: TutorState) -> list[str]:
    """Readings a model produced but couldn't confirm, offered to the learner to tap and check."""
    seen = []
    for attempt in state.get("attempts", []):
        reading = attempt["extraction"].get("unconfirmed")
        if reading and reading not in seen:
            seen.append(reading)
    return seen


def run(inputs: dict[str, Any], services: TutorServices | None = None) -> TutorState:
    """Convenience one-shot call (used by tests and scripts)."""
    return build_graph(services).invoke(inputs)
