"""Cloud solving (OpenAI) with independent SymPy verification.

A strong cloud model reads the photo or text, picks out the actual question (ignoring
anything else in the image), and solves it step by step. It also restates the problem in
calculator form; SymPy solves that independently. Agreement -> "checked". Disagreement ->
the model gets one chance to re-examine; if it still can't be confirmed, the answer is shown
clearly labelled as unchecked rather than presented as fact.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field

from . import config, prompts
from .compare import answers_agree
from .images import prepare_for_model
from .math_tool import evaluate_given, solve_problem

_STEP = {
    "type": "object", "additionalProperties": False,
    "required": ["title", "latex", "explanation"],
    "properties": {
        "title": {"type": "string"},
        "latex": {"type": "string"},
        "explanation": {"type": "string"},
    },
}

SOLUTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["found_math", "problem", "problem_latex", "topic", "picture", "steps", "final_answer",
                 "final_answer_latex", "answer_for_check", "check_problem", "check_facts", "check_expression",
                 "encouragement"],
    "properties": {
        "found_math": {"type": "boolean"},
        "problem": {"type": "string"},
        "problem_latex": {"type": "string"},
        "topic": {"type": "string"},
        "picture": {"type": "string"},
        "steps": {"type": "array", "items": _STEP},
        "final_answer": {"type": "string"},
        "final_answer_latex": {"type": "string"},
        "answer_for_check": {"type": "string"},
        "check_problem": {"type": "string"},
        "check_facts": {"type": "array", "items": {"type": "string"}},
        "check_expression": {"type": "string"},
        "encouragement": {"type": "string"},
    },
}


class CloudUnavailable(RuntimeError):
    """No key, no network, quota exhausted... the caller falls back to the local models."""


@dataclass
class CloudSolution:
    found_math: bool
    data: dict
    verified: str  # "checked" | "unchecked" | "corrected"
    check_answer: str = ""  # what SymPy computed, when it could
    attempts: list[dict] = field(default_factory=list)


def _client():
    if not config.OPENAI_API_KEY:
        raise CloudUnavailable("No OpenAI API key configured.")
    from openai import OpenAI
    return OpenAI(api_key=config.OPENAI_API_KEY, timeout=config.CLOUD_TIMEOUT, max_retries=2)


def _ask(messages: list[dict]) -> dict:
    kwargs = {"model": config.OPENAI_MODEL, "messages": messages,
              "response_format": {"type": "json_schema",
                                  "json_schema": {"name": "solution", "strict": True, "schema": SOLUTION_SCHEMA}}}
    if config.OPENAI_REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.OPENAI_REASONING_EFFORT
    try:
        reply = _client().chat.completions.create(**kwargs)
    except CloudUnavailable:
        raise
    except Exception as exc:  # network, auth, quota, model name...: fall back to local
        raise CloudUnavailable(f"{type(exc).__name__}: {exc}") from exc
    content = reply.choices[0].message.content or "{}"
    return json.loads(content)


def _user_content(image: bytes | None, text: str | None) -> list[dict]:
    parts: list[dict] = []
    if text:
        parts.append({"type": "text", "text": f"The learner's question:\n{text}"})
    if image:
        b64 = base64.b64encode(prepare_for_model(image)).decode()
        parts.append({"type": "text", "text": "The learner's photo:"})
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return parts


def independent_check(data: dict) -> tuple[bool | None, str]:
    """(agrees?, what SymPy got). None when SymPy couldn't check this problem at all."""
    reference = (data.get("answer_for_check") or "").strip()
    if not reference:
        return None, ""
    facts = [f for f in data.get("check_facts") or [] if f.strip()]
    if facts and data.get("check_expression", "").strip():
        result = evaluate_given([(f, "given") for f in facts], data["check_expression"])
    elif data.get("check_problem", "").strip():
        result = solve_problem(data["check_problem"])
    else:
        return None, ""
    if not result.ok:
        return None, ""
    return answers_agree(result.answer, reference), result.answer


def solve(image: bytes | None = None, text: str | None = None) -> CloudSolution:
    messages = [{"role": "system", "content": prompts.CLOUD_SYSTEM_PROMPT},
                {"role": "user", "content": _user_content(image, text)}]
    data = _ask(messages)
    attempts = [{"answer": data.get("final_answer"), "check": data.get("check_problem")}]
    if not data.get("found_math"):
        return CloudSolution(False, data, "unchecked", attempts=attempts)

    agrees, computed = independent_check(data)
    if agrees:
        return CloudSolution(True, data, "checked", computed, attempts)
    if agrees is False:
        # Self-correction: show the model what the calculator got and let it re-examine once.
        messages += [{"role": "assistant", "content": json.dumps(data)},
                     {"role": "user", "content": prompts.CLOUD_RECHECK.format(
                         check=data.get("check_problem") or data.get("check_expression"), computed=computed,
                         answer=data.get("final_answer"))}]
        second = _ask(messages)
        attempts.append({"answer": second.get("final_answer"), "check": second.get("check_problem")})
        agrees2, computed2 = independent_check(second)
        if agrees2:
            return CloudSolution(True, second, "corrected", computed2, attempts)
        return CloudSolution(True, second, "unchecked", computed2 or computed, attempts)
    return CloudSolution(True, data, "unchecked", "", attempts)
