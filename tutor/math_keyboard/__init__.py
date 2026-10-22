"""A math input box with an on-screen keyboard for every area of math (MathLive, bundled locally).

    latex = math_input("fix-1", label="Type the problem exactly", button="Check this")
    if latex: ...  # returned once per press of the button

The frontend is plain HTML + MathLive (no build step); MathLive's MIT licence is in frontend/.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_component = components.declare_component("math_keyboard", path=str(Path(__file__).parent / "frontend"))


def math_input(key: str, label: str = "", button: str = "Solve this", value: str = "",
               placeholder: str = "") -> str | None:
    """Show the math box; return the LaTeX the learner submitted, once per submission (else None)."""
    result = _component(key=key, label=label, button=button, value=value, placeholder=placeholder, default=None)
    seen_key = f"_math_input_seen_{key}"
    if result and result.get("nonce") != st.session_state.get(seen_key):
        st.session_state[seen_key] = result["nonce"]
        return result.get("latex") or None
    return None
