# 🐢 Slow & Steady: Universal Math Tutor Agent

A calm, patient, **local-first** math tutor for students, parents, and lifelong learners who feel anxious about math.
Snap a photo of a problem (or type it). The tutor reads it, checks the math with a deterministic calculator
(SymPy), and explains it slowly with an everyday analogy: sharing food, slicing pizza, balancing a scale.

Everything runs on your machine through [Ollama](https://ollama.com). Nothing leaves your computer.

## Quick start

```bash
# 1. Models (any vision model + any chat model work; these are the defaults)
ollama pull qwen2.5vl:7b        # reads photos
ollama pull qwen3:30b-a3b       # sets up word/diagram problems and writes explanations (~19 GB)
ollama pull qwen2.5vl:3b        # optional second opinion for hard photos

# 2. Python 3.10–3.13
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Run
.venv/bin/streamlit run app.py
```

Override models with environment variables: `TUTOR_VISION_MODEL`, `TUTOR_BACKUP_VISION_MODEL`, `TUTOR_CHAT_MODEL`,
`TUTOR_REASONING_MODEL`, `OLLAMA_HOST`, `TUTOR_MIN_CONFIDENCE`, `TUTOR_SETUP_ATTEMPTS` (see `tutor/config.py`).
On smaller machines, `TUTOR_CHAT_MODEL=llama3.1` works, but word and diagram problems will rarely be solved.

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
| **Word or diagram problem** (ages, mixtures, angles in a figure…) | The reasoning model writes equations, each with its reason; SymPy solves them. Two independent setups must agree (and match an answer choice, if any). If a diagram can't be set up, the learner sees the diagram description and can correct it. |

### What it can solve (with worked steps)

| Area | Examples you can type (or photograph) |
|---|---|
| Arithmetic, fractions, percentages | `3/4 + 1/8`, `25% of 40`, `2 * (6 + 4)` |
| Factorials and counting | `5!`, `5C2`, `10 choose 3`, `17 mod 5`, `gcd(12, 18)` |
| Equations and inequalities | `2x + 3 = 11`, `-3x <= 9`, `x^2 - 5x + 6 = 0` (factoring or the quadratic formula) |
| Simultaneous equations | `x + y = 10; x - y = 2` (also with commas or "and") |
| Complex numbers | `(2 + 3i)(1 - i)`, `(1 + 2i) / (3 - i)`, `abs(3 + 4i)`, `polar(1 + i)`, `i^7`, `x^2 + 2x + 5 = 0` |
| Matrices | `det [[1, 2], [3, 4]]`, `inverse of [[4, 7], [2, 6]]`, `[[1, 2], [3, 4]] * [[5, 6], [7, 8]]`, `eigenvalues of [[2, 1], [1, 2]]`, `rank`, `rref`, `trace`, `transpose`, `A^3`, `solve [[2, 1], [1, 3]] x = [3, 5]` |
| Derivatives | `derivative of x^2 sin(x)`, `d/dx (3x^4 - 2x)`, `second derivative of x^3`, `f'(x) for f(x) = sqrt(x^2 + 1)` (power, product, quotient, chain rules named) |
| Integrals | `integral of x e^x dx`, `integral from 0 to 1 of x^2 dx`, `∫ x ln(x) dx` (substitution, integration by parts, …) |
| Limits | `limit of sin(x)/x as x -> 0`, `limit of (1 + 1/n)^n as n -> infinity`, one-sided limits (L'Hôpital's rule shown) |
| Differential equations, series | `y' = 2y, y(0) = 3`, `y'' + y = 0`, `y' + y = x`, `taylor series of e^x`, `maclaurin series of sin(x) to order 5` |
| Word and diagram problems | "Maria is 3 times as old as her son…", angle problems from a photo |

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
4. **The learner can always correct the reading.** Photo answers show "I read your problem as …" with a box to fix it,
   and diagram descriptions are editable when a setup fails.
5. **Setups are cross-checked.** Word and diagram problems are set up twice, independently: first quickly (thinking
   off, seconds), then, only if those disagree, carefully (thinking on, capped). Only an agreed answer is shown.
   In testing, a quick setup wrongly treated vertically opposite angles as adding to 180° (x = 40); the
   disagreement triggered the careful pass, which found x = 50.
6. **Calculus answers are checked a second, independent way**: integrals by differentiating the answer (plus a
   numerical area for definite ones), derivatives by a numerical slope, limits by evaluating right next to the point,
   differential equations by substituting the solution back in, Taylor polynomials against SymPy's own series.
7. **Blank-photo pre-check** (`tutor/images.py`): blank or completely flat images go straight to "ask for context"
   without waiting on the models.

## Project layout

```
app.py               Streamlit UI (conversation, topic buttons, corrections, "how I checked this")
tutor/graph.py       LangGraph state machine and routing
tutor/math_tool.py   SymPy engine: parse safely, solve, generate steps, verify (incl. complex numbers, systems)
tutor/matrix_tool.py Matrix operations with their own eval-free parser
tutor/calculus_tool.py Derivatives, integrals, limits, ODEs, Taylor series, each checked a second way
tutor/latex.py       Step text → typeset math (fractions, ∫, matrices) for the whiteboard view
tutor/ui.py          Page style and the handwritten-whiteboard lesson card
tutor/explain.py     Tutor prose → number guard → final Markdown lesson
tutor/services.py    Ollama calls (vision, transcription, text parsing, proxy problems)
tutor/topics.py      Known topics + ready-made, verified practice problems
tutor/images.py      Image loading, downscaling, blank detection
tutor/prompts.py     Every prompt and every learner-facing fixed message
tests/               223 tests; models are faked, so no Ollama needed
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
- **Diagrams are the weak spot.** `qwen2.5vl:7b` often misdescribes figures (e.g. inventing a point), and the
  setup can only be as good as the description. The app then asks the learner to check the description.
- **Careful (thinking) setups take 2–3 minutes** on an M1 Max. Most word problems finish in the quick pass (~25s).
