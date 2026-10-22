"""Check the cloud setup without printing the key:  .venv/bin/python -m tutor.cloud_check"""

from __future__ import annotations

import time

from . import cloud, config


def main() -> None:
    if not config.OPENAI_API_KEY:
        print("✘ No OPENAI_API_KEY found. Add it to .env (see .env.example).")
        return
    print(f"✔ Key found (ends …{config.OPENAI_API_KEY[-4:]}). Configured model: {config.OPENAI_MODEL}")
    from openai import OpenAI
    try:
        ids = sorted(m.id for m in OpenAI(api_key=config.OPENAI_API_KEY, timeout=30).models.list())
    except Exception as exc:
        print(f"✘ Couldn't reach OpenAI with this key: {type(exc).__name__}: {exc}")
        return
    chat_models = [i for i in ids if i.startswith(("gpt-", "o1", "o3", "o4"))]
    print(f"✔ Key works. {len(chat_models)} chat models available to it, e.g.: {', '.join(chat_models[-12:])}")
    if config.OPENAI_MODEL not in ids:
        print(f"✘ '{config.OPENAI_MODEL}' isn't available to this key. Set TUTOR_OPENAI_MODEL in .env to one above.")
        return
    start = time.time()
    try:
        solution = cloud.solve(text="Solve 2x + 3 = 11")
    except cloud.CloudUnavailable as exc:
        print(f"✘ Test solve failed: {exc}")
        return
    print(f"✔ Test solve in {time.time() - start:.0f}s: {solution.data.get('final_answer')!r}, "
          f"verification: {solution.verified}")


if __name__ == "__main__":
    main()
