"""Model names and thresholds. Override any of these with environment variables."""

import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Primary vision/OCR model. The 7b says "partial"/"unreadable" for blurry photos where
# the 3b confidently invents numbers, for ~2s more per read.
VISION_MODEL = os.getenv("TUTOR_VISION_MODEL", "qwen2.5vl:7b")
# Second opinion, only used when the primary model can't read the photo clearly.
BACKUP_VISION_MODEL = os.getenv("TUTOR_BACKUP_VISION_MODEL", "qwen2.5vl:3b")
# Writes the explanations, proxy problems, and parses typed questions.
# qwen3 (thinking off) writes clearer explanations than llama3.1, and sharing one model with
# REASONING_MODEL avoids Ollama swapping three large models in and out of GPU memory.
CHAT_MODEL = os.getenv("TUTOR_CHAT_MODEL", "qwen3:30b-a3b")
# Turns word and diagram problems into equations for SymPy. Falls back to CHAT_MODEL if not installed.
REASONING_MODEL = os.getenv("TUTOR_REASONING_MODEL", "qwen3:30b-a3b")
# Independent setups that must agree before a word/diagram answer is trusted. They run as a fast
# pass (thinking off, seconds) and, only if those disagree, a slow pass (thinking on, minutes).
SETUP_ATTEMPTS = int(os.getenv("TUTOR_SETUP_ATTEMPTS", "2"))

# Below this self-reported confidence, extracted numbers are not trusted enough to solve.
MIN_CONFIDENCE = float(os.getenv("TUTOR_MIN_CONFIDENCE", "0.6"))

# Images are downscaled to this longest side before being sent to a vision model.
MAX_IMAGE_SIDE = 1280
