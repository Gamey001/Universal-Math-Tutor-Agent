"""Everything that talks to Ollama.

The graph only depends on the small ``TutorServices`` interface, so tests can
swap in a fake without a running model.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from . import config, prompts

LEGIBILITY_LEVELS = ("clear", "partial", "unreadable")
# Same context size and keep-alive on every call, so Ollama loads each model once and keeps it.
_SHARED = {"num_ctx": config.NUM_CTX, "keep_alive": config.KEEP_ALIVE}


class TutorServices(Protocol):
    vision_model: str
    backup_vision_model: str | None
    tutor_llm: BaseChatModel

    def read_image(self, model: str, image_png: bytes) -> dict: ...
    def transcribe_image(self, model: str, image_png: bytes) -> str: ...
    def describe_image(self, model: str, image_png: bytes) -> str: ...
    def extract_from_text(self, text: str, attempt: int = 0) -> dict: ...
    def make_proxy(self, topic: str) -> dict: ...
    def formulate(self, problem: str, diagram: str, think: bool = False) -> dict: ...


def parse_json(text: str) -> dict:
    """Parse a model's JSON reply, tolerating code fences or chatter around it."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_extraction(raw: dict, default_legibility: str = "unreadable") -> dict:
    """Coerce whatever a model returned into a predictable extraction dict."""
    legibility = _text(raw.get("legibility")).lower()
    if legibility not in LEGIBILITY_LEVELS:
        legibility = default_legibility
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "legibility": legibility,
        "problem_text": _text(raw.get("problem_text")),
        "expression": _text(raw.get("expression")),
        "topic": _text(raw.get("topic")),
        "confidence": confidence,
    }


def _number_or_none(value: Any) -> float | None:
    try:
        return float(str(value).replace("°", "").strip())
    except (TypeError, ValueError):
        return None


_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
# Function names the math tool understands; never renamed.
_KEEP_NAMES = {"sqrt", "sin", "cos", "tan", "log", "ln", "exp", "abs", "pi", "factorial", "binomial",
               "gcd", "lcm", "floor", "ceiling", "ceil", "mod"}


def normalize_setup(raw: dict) -> dict:
    """Coerce a formulate() reply into {"unknowns", "equations", "target", "choices"}.

    Models like descriptive names ("angle_PMQ", "apples"); the math tool only accepts
    single letters (anything longer could be read as implicit multiplication). So
    names are mapped to unused single letters here, keeping the name as the meaning.
    """
    meanings = {}
    for u in raw.get("unknowns") or []:
        if isinstance(u, dict) and _text(u.get("name")):
            meanings[_text(u["name"]).replace(" ", "_")] = _text(u.get("meaning"))
    equations = []
    for e in raw.get("equations") or []:
        # skip placeholders like "a = ?" (the model restating what it's looking for)
        if isinstance(e, dict) and _text(e.get("equation")) and "?" not in e["equation"]:
            text = _text(e["equation"])
            for name in sorted(meanings, key=len, reverse=True):  # "angle PMQ" -> "angle_PMQ"
                if "_" in name:
                    text = text.replace(name.replace("_", " "), name)
            equations.append({"equation": text, "reason": _text(e.get("reason"))})

    target_expression = _text(raw.get("target_expression")).replace(" ", " ")
    singles = {n for n in meanings if re.fullmatch(r"[a-z]", n)}
    for e in equations:
        for name in _IDENTIFIER.findall(e["equation"]) + _IDENTIFIER.findall(target_expression):
            if re.fullmatch(r"[a-z]", name):
                singles.add(name)

    def split_products(text: str) -> str:
        """'4pqxy' -> '4p*q*x*y' when every letter is already a one-letter unknown."""
        def fix(m: re.Match) -> str:
            word = m.group(0)
            if word.lower() in _KEEP_NAMES or word in meanings or not set(word) <= singles or len(word) < 2:
                return word
            return "*".join(word)
        return _IDENTIFIER.sub(fix, text)

    for e in equations:
        e["equation"] = split_products(e["equation"])
    target_expression = split_products(target_expression)

    names = []
    for e in equations + [{"equation": target_expression}]:
        for name in _IDENTIFIER.findall(e["equation"]):
            if name.lower() not in _KEEP_NAMES and name not in names:
                names.append(name)
    used = {n for n in names if re.fullmatch(r"[a-z]", n)}
    pool = [c for c in "xyzabcdfghjkmnpqrstuvw" if c not in used]
    rename = {}
    for name in names:
        if not re.fullmatch(r"[a-z]", name):
            if not pool:
                return {"unknowns": [], "equations": [], "target": "", "choices": []}  # too many unknowns
            rename[name] = pool.pop(0)

    def apply(text: str) -> str:
        return _IDENTIFIER.sub(lambda m: rename.get(m.group(0), m.group(0)), text)

    for e in equations:
        e["equation"] = apply(e["equation"])
    target_expression = apply(target_expression)
    unknowns = []
    for name in names:
        meaning = meanings.get(name) or name.replace("_", " ")
        unknowns.append({"name": rename.get(name, name), "meaning": meaning})
    target = _text(raw.get("target")).replace(" ", "_")
    choices = [n for n in (_number_or_none(c) for c in raw.get("choices") or []) if n is not None]
    return {"unknowns": unknowns, "equations": equations, "target": rename.get(target, target),
            "target_expression": target_expression, "choices": choices}


class OllamaServices:
    def __init__(
        self,
        vision_model: str = config.VISION_MODEL,
        backup_vision_model: str | None = config.BACKUP_VISION_MODEL,
        chat_model: str = config.CHAT_MODEL,
        reasoning_model: str = config.REASONING_MODEL,
        base_url: str = config.OLLAMA_HOST,
    ):
        self.vision_model = vision_model
        self.backup_vision_model = backup_vision_model
        self.chat_model = chat_model
        self.reasoning_model = reasoning_model
        self.base_url = base_url
        self._broken_models: dict[str, str] = {}
        no_think = False if chat_model.startswith("qwen3") else None  # quick replies; no thinking needed
        self._chat_json = ChatOllama(model=chat_model, base_url=base_url, format="json", temperature=0,
                                     reasoning=no_think, **_SHARED)
        # JSON mode: the tutor only writes prose fields; explain.py inserts all the math.
        self.tutor_llm = ChatOllama(model=chat_model, base_url=base_url, format="json", temperature=0.4,
                                    reasoning=no_think, **_SHARED)

    def _ask_vision(self, model: str, prompt: str, image_png: bytes, json_mode: bool) -> str:
        if model in self._broken_models:
            raise RuntimeError(self._broken_models[model])
        b64 = base64.b64encode(image_png).decode()
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ])
        # qwen3-vl thinks by default; reading a photo doesn't need it, and it's much slower.
        llm = ChatOllama(model=model, base_url=self.base_url, temperature=0, format="json" if json_mode else None,
                         reasoning=False if model.startswith("qwen3") else None, **_SHARED)
        try:
            return llm.invoke([message]).content
        except Exception as exc:
            # e.g. "unknown model architecture": the model can't load on this Ollama
            # version. Remember that so we don't retry it on every photo.
            if "error loading model" in str(exc) or "not found" in str(exc):
                self._broken_models[model] = str(exc)
            raise

    def read_image(self, model: str, image_png: bytes) -> dict:
        reply = self._ask_vision(model, prompts.VISION_PROMPT, image_png, json_mode=True)
        return normalize_extraction(parse_json(reply))

    def transcribe_image(self, model: str, image_png: bytes) -> str:
        return self._ask_vision(model, prompts.TRANSCRIBE_PROMPT, image_png, json_mode=False).strip()

    def describe_image(self, model: str, image_png: bytes) -> str:
        return self._ask_vision(model, prompts.DIAGRAM_PROMPT, image_png, json_mode=False).strip()

    def active_reasoning_model(self) -> str:
        """The reasoning model if it's installed, otherwise the chat model."""
        installed = installed_models(self.base_url) or set()
        return self.reasoning_model if self.reasoning_model in installed else self.chat_model

    def formulate(self, problem: str, diagram: str, think: bool = False) -> dict:
        """One independent setup attempt: unknowns, equations with reasons, target, choices.

        ``think`` turns on qwen3's thinking mode: much slower, used only when quick setups
        disagree. Thinking is capped so a confusing problem can't loop for minutes.
        """
        model = self.active_reasoning_model()
        qwen3 = model.startswith("qwen3")
        llm = ChatOllama(model=model, base_url=self.base_url, format="json", temperature=0.6,
                         reasoning=(think if qwen3 else None), num_predict=6000 if think else 1500, **_SHARED)
        diagram_block = f"\nWhat the diagram shows:\n{diagram}\n" if diagram else ""
        data = parse_json(llm.invoke(prompts.FORMULATE_PROMPT.format(problem=problem, diagram=diagram_block)).content)
        return normalize_setup(data) | {"model": model}

    def extract_from_text(self, text: str, attempt: int = 0) -> dict:
        """Words -> math. attempt > 0 samples an independent second translation to cross-check the first."""
        llm = self._chat_json if attempt == 0 else ChatOllama(
            model=self.chat_model, base_url=self.base_url, format="json", temperature=0.7,
            reasoning=False if self.chat_model.startswith("qwen3") else None, **_SHARED)
        reply = llm.invoke(prompts.TEXT_EXTRACTION_PROMPT.format(text=text))
        data = parse_json(reply.content)
        return {
            "expression": _text(data.get("expression")),
            "topic": _text(data.get("topic")),
            "problem_text": _text(data.get("problem_text")),
        }

    def make_proxy(self, topic: str) -> dict:
        reply = self._chat_json.invoke(prompts.PROXY_PROMPT.format(topic=topic))
        data = parse_json(reply.content)
        return {"problem_text": _text(data.get("problem_text")), "expression": _text(data.get("expression"))}


def installed_models(base_url: str = config.OLLAMA_HOST) -> set[str] | None:
    """Names of locally pulled models, or None if Ollama isn't reachable."""
    import ollama

    try:
        listing = ollama.Client(host=base_url).list()
    except Exception:
        return None
    names = set()
    for m in listing.models:
        names.add(m.model)
        if m.model.endswith(":latest"):
            names.add(m.model.removesuffix(":latest"))
    return names
