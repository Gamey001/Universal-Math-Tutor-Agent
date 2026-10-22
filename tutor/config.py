"""Model names and thresholds. Override any of these with environment variables."""

import os
from pathlib import Path


def _load_dotenv(path: Path = Path(__file__).resolve().parent.parent / ".env") -> None:
    """Read KEY=value lines from the project's git-ignored .env, without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# --- Cloud (OpenAI): reads and solves; the local models below are the automatic fallback. ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# "auto" uses the cloud whenever a key is set; "off" forces local-only (e.g. for privacy or testing).
CLOUD = os.getenv("TUTOR_CLOUD", "auto")
OPENAI_MODEL = os.getenv("TUTOR_OPENAI_MODEL", "gpt-5.5")
# For reasoning models: "low" | "medium" | "high" ("" for models without reasoning, e.g. gpt-4.1).
OPENAI_REASONING_EFFORT = os.getenv("TUTOR_OPENAI_REASONING_EFFORT", "medium")
CLOUD_TIMEOUT = float(os.getenv("TUTOR_CLOUD_TIMEOUT", "180"))


def cloud_enabled() -> bool:
    return CLOUD != "off" and bool(OPENAI_API_KEY)


# --- Local models (Ollama) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Defaults are the setup measured on eval/cases.json: 46/60 correct, 0 safety failures, median 8.8s.
# qwen3-vl:30b-a3b was tried as a single all-in-one model, but that tag is the "thinking" edition: it
# reasons before every answer even with thinking off (~45s for a one-line reply), so a question took
# minutes. The non-thinking edition (qwen3-vl:30b-a3b-instruct) is the candidate to evaluate next.
VISION_MODEL = os.getenv("TUTOR_VISION_MODEL", "qwen2.5vl:7b")
# Second opinion, only used when the primary model can't read the photo clearly ("" turns it off).
BACKUP_VISION_MODEL = os.getenv("TUTOR_BACKUP_VISION_MODEL", "qwen2.5vl:3b") or None
# Writes the explanations, proxy problems, and parses typed questions (thinking off: fast).
CHAT_MODEL = os.getenv("TUTOR_CHAT_MODEL", "qwen3:30b-a3b")
# Turns word and diagram problems into equations for SymPy. Falls back to CHAT_MODEL if not installed.
REASONING_MODEL = os.getenv("TUTOR_REASONING_MODEL", "qwen3:30b-a3b")
# Every model call uses the same context size: Ollama reloads the whole model whenever it changes,
# which with a 20 GB model meant several reloads (minutes) per question. 16k fits every prompt here.
NUM_CTX = int(os.getenv("TUTOR_NUM_CTX", "16384"))
# Keep the model in memory between questions instead of reloading it after Ollama's 5-minute default.
KEEP_ALIVE = os.getenv("TUTOR_KEEP_ALIVE", "30m")

# Independent setups that must agree before a word/diagram answer is trusted. They run as a fast
# pass (thinking off, seconds) and, only if those disagree, a slow pass (thinking on, minutes).
SETUP_ATTEMPTS = int(os.getenv("TUTOR_SETUP_ATTEMPTS", "2"))

# Below this self-reported confidence, extracted numbers are not trusted enough to solve.
MIN_CONFIDENCE = float(os.getenv("TUTOR_MIN_CONFIDENCE", "0.6"))

# Images are downscaled to this longest side before being sent to a vision model.
MAX_IMAGE_SIDE = 1280
