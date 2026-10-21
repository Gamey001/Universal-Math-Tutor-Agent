import io
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from PIL import Image, ImageDraw, ImageFilter


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def text_photo() -> bytes:
    img = Image.new("RGB", (600, 200), "white")
    ImageDraw.Draw(img).text((40, 80), "2x + 3 = 11", fill="black")
    return png_bytes(img.resize((1200, 400)))


@pytest.fixture
def blurry_photo() -> bytes:
    img = Image.new("RGB", (600, 200), "white")
    ImageDraw.Draw(img).rectangle((40, 60, 560, 140), fill="black")
    return png_bytes(img.filter(ImageFilter.GaussianBlur(6)))


@pytest.fixture
def blank_photo() -> bytes:
    return png_bytes(Image.new("RGB", (600, 400), (240, 238, 230)))


TUTOR_JSON = json.dumps({
    "asking": "We want to find the mystery number.",
    "picture": "Picture a balance scale.",
    "step_notes": ["Keep the scale level.", "Share it out evenly."],
    "answer_note": "That's the number in the bag.",
    "big_idea": "A fraction is a piece of a whole pizza.",
    "encouragement": "You did it. Why do we do the same thing to both sides?",
})


class FakeServices:
    """Scripted stand-in for OllamaServices. Records every call."""

    def __init__(self, reads=None, transcriptions=None, text_extractions=None, proxies=None,
                 backup=True, tutor_replies=None, setups=None, diagram="A diagram."):
        self.vision_model = "primary"
        self.backup_vision_model = "backup" if backup else None
        self.tutor_llm = FakeListChatModel(responses=tutor_replies or [TUTOR_JSON])
        self._reads = dict(reads or {})
        # By default a transcription agrees with the structured read.
        self._transcriptions = dict(transcriptions or {})
        self._setups = list(setups or [])
        self._diagram = diagram
        self._text = list(text_extractions or [])
        self._proxies = list(proxies or [])
        self.calls = []

    def read_image(self, model, image_png):
        self.calls.append(("read_image", model))
        result = self._reads.get(model)
        if isinstance(result, Exception):
            raise result
        return {"legibility": "unreadable", "problem_text": "", "expression": "",
                "topic": "", "confidence": 0.0, **(result or {})}

    def transcribe_image(self, model, image_png):
        self.calls.append(("transcribe_image", model))
        if model in self._transcriptions:
            return self._transcriptions[model]
        read = self._reads.get(model) or {}
        return read.get("problem_text") or read.get("expression", "")

    def describe_image(self, model, image_png):
        self.calls.append(("describe_image", model))
        return self._diagram

    def formulate(self, problem, diagram):
        self.calls.append(("formulate", problem))
        return self._setups.pop(0) if self._setups else {"unknowns": [], "equations": [], "target": "", "choices": []}

    def extract_from_text(self, text):
        self.calls.append(("extract_from_text", text))
        return self._text.pop(0) if self._text else {"expression": "", "topic": "", "problem_text": ""}

    def make_proxy(self, topic):
        self.calls.append(("make_proxy", topic))
        return self._proxies.pop(0) if self._proxies else {"problem_text": "", "expression": ""}


@pytest.fixture
def fake_services():
    return FakeServices
