"""Run the tutor on the evaluation set and report accuracy, wrong answers and speed.

    .venv/bin/python -m eval.run_eval --label baseline
    .venv/bin/python -m eval.run_eval --label quick --category calculus --limit 3
    .venv/bin/python -m eval.run_eval compare eval/results/A.json eval/results/B.json

Models come from the usual TUTOR_* environment variables, so comparing models is
just running this twice with different settings. Results are saved to
eval/results/<timestamp>-<label>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
OUTCOMES = ("correct", "wrong", "declined", "error")
ICONS = {"correct": "✅", "wrong": "❌", "declined": "⚪", "error": "💥"}


def load_cases(path: Path, only: set[str] | None, category: str | None, limit: int | None) -> list[dict]:
    cases = json.loads(path.read_text())["cases"]
    if only:
        cases = [c for c in cases if c["id"] in only]
    if category:
        cases = [c for c in cases if c["category"] == category]
    return cases[:limit] if limit else cases


def run_case(graph, case: dict) -> dict:
    from eval.scoring import score

    inputs = dict(case["input"])
    if "image" in inputs:
        inputs["image"] = (HERE / inputs["image"]).read_bytes()
    start = time.time()
    state, error = None, None
    try:
        state = graph.invoke(inputs)
    except Exception as exc:  # recorded, never fatal: one bad case mustn't stop the run
        error = f"{type(exc).__name__}: {exc}"
    seconds = round(time.time() - start, 1)
    outcome = score(state, case["expect"], error)
    state = state or {}
    attempts = state.get("attempts") or []
    return {
        "id": case["id"], "category": case["category"], "outcome": outcome, "seconds": seconds,
        "status": state.get("status"), "answer": (state.get("math") or {}).get("answer"),
        "expected": case["expect"], "error": error,
        "read": attempts[0]["extraction"].get("expression") if attempts else None,
        "setups": [s["result"].get("answer") or s["result"].get("error") for s in state.get("setups") or []],
    }


def summarize(results: list[dict]) -> str:
    total = len(results)
    counts = Counter(r["outcome"] for r in results)
    lines = [f"**{counts['correct']}/{total} correct ({100 * counts['correct'] / max(total, 1):.0f}%)** · "
             f"❌ wrong: {counts['wrong']} · ⚪ declined: {counts['declined']} · 💥 errors: {counts['error']} · "
             f"median time: {sorted(r['seconds'] for r in results)[total // 2] if total else 0}s", "",
             "| Category | Correct | Wrong | Declined | Error |", "|---|---|---|---|---|"]
    by_cat = defaultdict(Counter)
    for r in results:
        by_cat[r["category"]][r["outcome"]] += 1
    for cat, c in sorted(by_cat.items()):
        lines.append(f"| {cat} | {c['correct']}/{sum(c.values())} | {c['wrong']} | {c['declined']} | {c['error']} |")
    wrong = [r for r in results if r["outcome"] in ("wrong", "error")]
    if wrong:
        lines += ["", "Wrong answers and errors (fix these first):"]
        lines += [f"- `{r['id']}`: got {r['answer'] or r['error']!r}, expected {r['expected']}" for r in wrong]
    return "\n".join(lines)


def compare(path_a: str, path_b: str) -> None:
    a, b = (json.loads(Path(p).read_text()) for p in (path_a, path_b))
    print(f"A = {a['label']} ({a['models']})\nB = {b['label']} ({b['models']})\n")
    print(f"A: {summarize(a['results']).splitlines()[0]}\nB: {summarize(b['results']).splitlines()[0]}\n")
    rb = {r["id"]: r for r in b["results"]}
    for ra in a["results"]:
        other = rb.get(ra["id"])
        if other and other["outcome"] != ra["outcome"]:
            print(f"{ra['id']:28s} {ICONS[ra['outcome']]} {ra['outcome']:9s} → {ICONS[other['outcome']]} {other['outcome']}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="run")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases", default=str(HERE / "cases.json"))
    args = parser.parse_args()

    from tutor import config
    from tutor.graph import build_graph

    cases = load_cases(Path(args.cases), set(args.only.split(",")) if args.only else None, args.category, args.limit)
    models = {"vision": config.VISION_MODEL, "backup_vision": config.BACKUP_VISION_MODEL,
              "chat": config.CHAT_MODEL, "reasoning": config.REASONING_MODEL}
    print(f"Running {len(cases)} cases with {models}\n", flush=True)
    graph = build_graph()
    results = []
    for i, case in enumerate(cases, 1):
        r = run_case(graph, case)
        results.append(r)
        print(f"[{i:2d}/{len(cases)}] {ICONS[r['outcome']]} {r['id']:28s} {r['seconds']:6.1f}s  "
              f"{(r['answer'] or r['status'] or r['error'] or '')[:70]}", flush=True)

    out = HERE / "results" / f"{datetime.now():%Y%m%d-%H%M}-{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "models": models, "when": datetime.now().isoformat(),
                               "results": results}, indent=2, default=str))
    print("\n" + summarize(results) + f"\n\nSaved {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
