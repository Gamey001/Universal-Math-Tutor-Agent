# 🐢 Slow & Steady: Universal Math Tutor Agent

A calm, patient, **local-first** math tutor for students, parents, and lifelong learners who feel anxious about math.
Snap a photo of a problem (or type it). The tutor reads it, checks the math with a deterministic calculator
(SymPy), and explains it slowly with an everyday analogy: sharing food, slicing pizza, balancing a scale.

Everything runs on your machine through [Ollama](https://ollama.com). Nothing leaves your computer.

## Quick start

```bash
# 1. Models (any vision model + any chat model work; these are the defaults)
ollama pull qwen2.5vl:7b        # reads photos
ollama pull llama3.1            # writes explanations
ollama pull qwen2.5vl:3b        # optional second opinion for hard photos

# 2. Python 3.10–3.13
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Run
.venv/bin/streamlit run app.py
```

Override models with environment variables: `TUTOR_VISION_MODEL`, `TUTOR_BACKUP_VISION_MODEL`, `TUTOR_CHAT_MODEL`,
`OLLAMA_HOST`, `TUTOR_MIN_CONFIDENCE` (see `tutor/config.py`).

## How it works

A LangGraph pipeline (`tutor/graph.py`):

```
image ─► precheck ─┬─ blank / not an image ─────────────────────► ask_context
                   └─► read_primary ─┬─ clear & confirmed ──────► solve ──► tutor
                                     └─► read_backup ─┐
text ──► read_text ───────────────────────────────────┤
                                                      ├─ clear & confirmed ► solve ──► tutor
                                                      ├─ topic known ──────► fallback ► tutor
                                                      └─ nothing usable ───► ask_context
topic button ──────────────────────────────────────────────────► fallback ► tutor
```

| Situation | What the tutor does |
|---|---|
| **A: clear photo or typed problem** | SymPy solves it and generates the steps. The tutor explains each verified step with one analogy. |
| **B: blurry photo, but the topic is visible** | It says kindly that it couldn't read the numbers, teaches the core idea, and walks through a **verified practice (proxy) problem** on the same topic. |
| **C: blank, fully blurred, unreadable, or no math found** | It **asks the learner** which area of math they need (topic buttons or free text) or to type the problem. Their answer continues the conversation. |
| **Typed topic** ("I don't get fractions") | Concept lesson plus a verified practice problem. |

### Accuracy safeguards

Small local models are good at language and unreliable at arithmetic and at reading blurry photos. So:

1. **The LLM never does math.** `tutor/math_tool.py` (SymPy) computes every answer *and* every step: balancing steps
   for linear equations and inequalities, common denominators for fractions, factoring for quadratics, and
   order of operations for arithmetic. It also checks answers by substituting them back in.
2. **Two independent reads before trusting a photo.** A structured read and a plain transcription must agree on
   the numbers. In testing, `qwen2.5vl:3b` confidently "read" `200 ÷ 20 × 100` from a photo with nothing legible
   on it; the transcription said `UNREADABLE`, so the app asked the learner instead. (`qwen2.5vl:7b`, now the
   default, admits blur far more often, which is why it reads first.)
3. **The tutor model only writes prose.** It returns JSON (analogy, why each step works, encouragement). The app
   inserts the verified math itself. Any number in the prose that the calculator didn't produce triggers a
   rewrite, and if it persists, that sentence is removed.
4. **The learner can always correct the reading.** Photo answers show "I read your problem as …" with a box to fix it.
5. **Blank-photo pre-check** (`tutor/images.py`): blank or completely flat images go straight to "ask for context"
   without waiting on the models.

## Project layout

```
app.py               Streamlit UI (conversation, topic buttons, corrections, "how I checked this")
tutor/graph.py       LangGraph state machine and routing
tutor/math_tool.py   SymPy engine: parse safely, solve, generate steps, verify
tutor/explain.py     Tutor prose → number guard → final Markdown lesson
tutor/services.py    Ollama calls (vision, transcription, text parsing, proxy problems)
tutor/topics.py      Known topics + ready-made, verified practice problems
tutor/images.py      Image loading, downscaling, blank detection
tutor/prompts.py     Every prompt and every learner-facing fixed message
tests/               61 tests; models are faked, so no Ollama needed
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Known limitations

- **A consistent misread can still get through.** If the vision model misreads a blurry `3/4` as `34` the same way
  on both reads, the reads agree. That's why the app always shows what it read and lets the learner correct it.
- **`llama3.2-vision` doesn't load on newer Ollama versions** (`unknown model architecture: 'mllama'`), so the
  default backup is `qwen2.5vl:7b`. If a configured model can't load, the app skips it and says so in the sidebar.
- Explanation quality depends on the chat model. `llama3.1` (8B) sometimes picks clumsy analogies, but the
  numbers it shows are always the verified ones.
