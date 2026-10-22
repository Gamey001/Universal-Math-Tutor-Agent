"""Turns verified math into a calm lesson, without letting the LLM touch the numbers.

The tutor model only writes prose (meaning, analogy, the "why" of each step)
as JSON. This module:
  1. builds the request from SymPy's verified result,
  2. checks the prose for any number the calculator didn't produce (retrying
     once, then dropping offending sentences),
  3. renders the final Markdown, inserting every calculation from SymPy itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import prompts
from .latex import is_prose, split_description, to_latex
from .services import parse_json

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_ALWAYS_ALLOWED = {"0", "1"}


@dataclass
class Lesson:
    """What the learner sees, plus guard diagnostics for the 'how this was checked' panel."""

    markdown: str
    removed_numbers: list[str] = field(default_factory=list)
    retried: bool = False


def allowed_numbers(*texts: str) -> set[str]:
    numbers = set(_ALWAYS_ALLOWED)
    for text in texts:
        for n in _NUMBER.findall(text or ""):
            numbers.add(n)
            numbers.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return numbers


def unverified_numbers(text: str, allowed: set[str]) -> list[str]:
    found = []
    for n in _NUMBER.findall(text):
        canonical = n.rstrip("0").rstrip(".") if "." in n else n
        if n not in allowed and canonical not in allowed and n not in found:
            found.append(n)
    return found


def _drop_unverified_sentences(text: str, allowed: set[str]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(s for s in sentences if not unverified_numbers(s, allowed)).strip()


def _fields_text(data: dict) -> str:
    parts = []
    for value in data.values():
        parts.extend(value if isinstance(value, list) else [value])
    return " ".join(str(p) for p in parts)


def _clean(data: dict, keys: list[str], n_steps: int) -> dict:
    """Keep only expected keys, with the right types and the right number of step notes."""
    out = {}
    for key in keys:
        if key == "step_notes":
            notes = data.get("step_notes")
            notes = [str(x).strip() for x in notes] if isinstance(notes, list) else []
            out[key] = (notes + [""] * n_steps)[:n_steps]
        else:
            value = data.get(key)
            out[key] = value.strip() if isinstance(value, str) else ""
    return out


def _ask(llm: BaseChatModel, request: str, keys: list[str], n_steps: int, allowed: set[str]) -> tuple[dict, list, bool]:
    messages = [SystemMessage(prompts.TUTOR_SYSTEM_PROMPT), HumanMessage(request)]
    reply = llm.invoke(messages)
    data = _clean(parse_json(reply.content), keys, n_steps)
    bad = unverified_numbers(_fields_text(data), allowed)
    retried = False
    if bad:
        retried = True
        messages += [AIMessage(reply.content),
                     HumanMessage(prompts.RETRY_NUMBERS.format(numbers=", ".join(bad)))]
        retry = _clean(parse_json(llm.invoke(messages).content), keys, n_steps)
        if any(retry.values()):
            data = retry
        bad = unverified_numbers(_fields_text(data), allowed)
    if bad:  # still there: remove just the sentences that contain them
        data = {k: ([_drop_unverified_sentences(x, allowed) for x in v] if isinstance(v, list)
                    else _drop_unverified_sentences(v, allowed))
                for k, v in data.items()}
    return data, bad, retried


def _numbered(steps: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def _display(latex: str) -> str:
    """A centred, typeset line of math (KaTeX in Streamlit's Markdown)."""
    return f"$$\n{latex.replace('$', '')}\n$$"


def _render_steps(steps: list[str], notes: list[str], problem_latex: str = "") -> str:
    """Laid out like a worked solution on paper: the problem, then "Step n: rule" with typeset math."""
    blocks = [_display(problem_latex)] if problem_latex else []
    for i, (step, note) in enumerate(zip(steps, notes), 1):
        # Split at the first ": " (later ones belong to the math, e.g. "Real part: ...; imaginary part: ...").
        label, sep, math = step.partition(": ")
        if "=" in label or not sep:  # no real label, e.g. "x = y + 2 → x = 6"
            label, math = "", step
        label = label.rstrip(".")  # reasons are often full sentences: "son." + ":" reads badly
        heading = f"**Step {i}: {label}**" if label else f"**Step {i}**"
        description, math = split_description(math)
        body = math if is_prose(math) else _display(to_latex(math))
        description = description[:1].upper() + description[1:] if description else ""
        blocks.append("\n\n".join(part for part in (heading, description, body, note) if part))
    return "\n\n".join(blocks)


def _render_answer(answer: str, verification: str, note: str) -> str:
    parts = [_display(rf"\boxed{{{to_latex(answer)}}}"), f"✔️ *Checked with SymPy.* {verification}"]
    if note:
        parts.append(note)
    return "\n\n".join(parts)


def explain_solution(llm: BaseChatModel, math: dict, original: str = "", legend: list[str] | None = None) -> Lesson:
    """``original`` is the learner's own wording (e.g. "What is 25% of 40?"), which
    gives the tutor far better context than the normalized math alone."""
    steps = math["steps"]
    request = prompts.SOLVE_REQUEST.format(
        original=original or math["problem"], problem=math["problem"], answer=math["answer"], n=len(steps),
        steps=_numbered(steps) if steps else "(none: the answer comes straight from the problem)",
    )
    allowed = allowed_numbers(original, math["problem"], math["answer"], math["verification"], *steps)
    keys = ["asking", "picture", "step_notes", "answer_note", "encouragement"]
    data, removed, retried = _ask(llm, request, keys, len(steps), allowed)

    sections = [("What the question is asking", data["asking"] or prompts.DEFAULT_ASKING)]
    if data["picture"]:
        sections.append(("A picture to hold in your mind", data["picture"]))
    if steps:
        intro = f"Let {'; '.join(legend)}.\n\n" if legend else ""  # what each letter stands for
        sections.append(("Step by step", intro + _render_steps(steps, data["step_notes"], math.get("problem_latex", ""))))
    sections.append(("The answer", _render_answer(math["answer"], math["verification"], data["answer_note"])))
    sections.append(("You've got this", data["encouragement"] or prompts.DEFAULT_ENCOURAGEMENT))
    return Lesson(_markdown(sections), removed, retried)


def explain_topic(llm: BaseChatModel, topic: str, proxy: dict | None, situation: str, invitation: str) -> Lesson:
    """Fallback lesson: the core idea of ``topic``, then a verified practice problem if we have one."""
    steps = proxy["math"]["steps"] if proxy else []
    if proxy:
        m = proxy["math"]
        proxy_block = prompts.FALLBACK_PROXY_BLOCK.format(
            problem_text=proxy["problem_text"], answer=m["answer"],
            steps=_numbered(steps) if steps else "(none: the answer comes straight from the problem)",
        )
        proxy_keys = prompts.FALLBACK_PROXY_KEYS.format(n=len(steps))
        allowed = allowed_numbers(proxy["problem_text"], m["problem"], m["answer"], m["verification"], *steps)
        keys = ["big_idea", "step_notes", "answer_note", "encouragement"]
    else:
        proxy_block, proxy_keys, allowed = "", "", allowed_numbers()
        keys = ["big_idea", "encouragement"]
    context = prompts.FALLBACK_CONTEXT_UNCHECKABLE if situation == "uncheckable" else ""
    request = prompts.FALLBACK_REQUEST.format(topic=topic, context=context,
                                              proxy_block=proxy_block, proxy_keys=proxy_keys)
    data, removed, retried = _ask(llm, request, keys, len(steps), allowed)

    sections = [("What happened" if situation != "topic" else "Let's begin",
                 prompts.WHAT_HAPPENED[situation].format(topic=topic))]
    if data["big_idea"]:
        sections.append(("The big idea", data["big_idea"]))
    if proxy:
        practice = f"*{proxy['problem_text']}*"
        if steps:
            practice += "\n\n" + _render_steps(steps, data["step_notes"], proxy["math"].get("problem_latex", ""))
        sections.append(("Let's practice", practice))
        sections.append(("The answer", _render_answer(proxy["math"]["answer"], proxy["math"]["verification"],
                                                      data["answer_note"])))
    encouragement = data["encouragement"] or prompts.DEFAULT_ENCOURAGEMENT
    sections.append(("You've got this", f"{encouragement}\n\n{invitation}"))
    return Lesson(_markdown(sections), removed, retried)


def _markdown(sections: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"### {title}\n{body}" for title, body in sections)


_VERIFIED_NOTE = {
    "checked": "✔️ *Checked with SymPy:* my calculator solved it independently and got the same answer.",
    "corrected": ("✔️ *Checked with SymPy:* my first attempt didn't match my calculator, so I re-checked my "
                  "work and corrected it. Now both agree."),
}


def render_cloud_lesson(data: dict, verified: str, check_answer: str = "") -> str:
    """A cloud solution laid out like the whiteboard lessons, with an honest verification label."""
    clean = lambda latex: (latex or "").replace("$", "").strip()
    sections = []
    asking = data.get("problem", "").strip() or prompts.DEFAULT_ASKING
    if clean(data.get("problem_latex")):
        asking += "\n\n" + _display(clean(data["problem_latex"]))
    sections.append(("What the question is asking", asking))
    if data.get("picture", "").strip():
        sections.append(("A picture to hold in your mind", data["picture"].strip()))
    blocks = []
    for i, step in enumerate(data.get("steps") or [], 1):
        title = step.get("title", "").strip().rstrip(".:")
        parts = [f"**Step {i}: {title}**" if title else f"**Step {i}**"]
        if clean(step.get("latex")):
            parts.append(_display(clean(step["latex"])))
        if step.get("explanation", "").strip():
            parts.append(step["explanation"].strip())
        blocks.append("\n\n".join(parts))
    if blocks:
        sections.append(("Step by step", "\n\n".join(blocks)))
    answer_latex = clean(data.get("final_answer_latex")) or to_latex(data.get("final_answer", ""))
    if verified in _VERIFIED_NOTE:
        note = _VERIFIED_NOTE[verified]
    elif check_answer:
        note = (f"⚠️ *Not confirmed:* my calculator got **{check_answer}** when it re-solved the problem, which "
                "doesn't match. Please double-check this one with your teacher or textbook.")
    else:
        note = ("ℹ️ *Solved by the AI.* My calculator can't check this kind of problem on its own, so treat the "
                "answer with a little care.")
    sections.append(("The answer", f"{_display(rf'\boxed{{{answer_latex}}}')}\n\n{note}"))
    sections.append(("You've got this", data.get("encouragement", "").strip() or prompts.DEFAULT_ENCOURAGEMENT))
    return _markdown(sections)
