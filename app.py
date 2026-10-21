"""Slow & Steady Math Tutor: Streamlit front end.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import uuid

import streamlit as st

from tutor import config
from tutor.graph import ServiceUnavailable, build_graph, unconfirmed_readings
from tutor.services import OllamaServices, installed_models
from tutor.topics import TOPIC_CHOICES

st.set_page_config(page_title="Slow & Steady Math Tutor", page_icon="🐢", layout="centered")

st.markdown(
    """
    <style>
      .block-container { max-width: 760px; padding-top: 2.5rem; }
      .stMarkdown p, .stMarkdown li { font-size: 1.06rem; line-height: 1.75; }
      h3 { margin-top: 1.4rem !important; color: #2f5d50; }
      .tutor-note { color: #5b6b66; font-size: 0.95rem; }
      .you-said { background: #eef4f1; border-radius: 12px; padding: 0.6rem 0.9rem; margin: 0.4rem 0 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Friendly progress lines, shown while each graph node runs.
PROGRESS = {
    "precheck": "Taking a first look at your photo…",
    "read_primary": "Reading your problem carefully…",
    "read_backup": "Taking a second, closer look…",
    "read_text": "Reading your question…",
    "solve": "Checking the math with my calculator…",
    "fallback": "Choosing a practice example for us…",
    "ask_context": "Thinking about how best to help…",
    "tutor": "Writing your explanation, one step at a time…",
}


@st.cache_resource
def get_services() -> OllamaServices:
    return OllamaServices()


@st.cache_resource
def get_graph():
    return build_graph(get_services())


def sidebar() -> None:
    st.sidebar.markdown("### 🐢 Slow & Steady")
    st.sidebar.markdown(
        "<span class='tutor-note'>Snap a photo or type a question. "
        "Every number is checked by a calculator (SymPy), never guessed by the AI.</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    models = installed_models()
    if models is None:
        st.sidebar.error("Ollama isn't running. Start it with `ollama serve`, then refresh.")
        return
    broken = get_services()._broken_models
    for label, name in [("Chat model", config.CHAT_MODEL), ("Vision model", config.VISION_MODEL),
                        ("Backup vision", config.BACKUP_VISION_MODEL)]:
        if name in broken:
            st.sidebar.warning(f"{label} can't load · `{name}`  \nSkipping it; photos use the main vision model only.")
        elif name in models or name.removesuffix(":latest") in models:
            st.sidebar.success(f"{label} ready · `{name}`")
        else:
            st.sidebar.warning(f"{label} not installed · `{name}`  \nRun `ollama pull {name}`")
    st.sidebar.divider()
    if st.sidebar.button("Start fresh", width="stretch"):
        st.session_state.turns = []
        st.rerun()


def queue(inputs: dict, label: str) -> None:
    """Remember a request; it runs on the next script pass, with progress shown."""
    st.session_state.pending = {"inputs": inputs, "label": label}


def run_pending() -> None:
    pending = st.session_state.pop("pending", None)
    if not pending:
        return
    turn = {"id": uuid.uuid4().hex, "label": pending["label"], "image": pending["inputs"].get("image")}
    with st.status("Let's take this slowly…", expanded=False) as status:
        try:
            state: dict = {}
            for update in get_graph().stream(pending["inputs"], stream_mode="updates"):
                for node, values in update.items():
                    status.update(label=PROGRESS.get(node, "Working…"))
                    for key, value in (values or {}).items():
                        state[key] = state.get(key, []) + value if key == "attempts" else value
            turn["state"] = state
            status.update(label="Ready", state="complete")
        except ServiceUnavailable as exc:
            turn["error"] = str(exc)
            status.update(label="I couldn't reach the AI models", state="error")
        except Exception as exc:  # keep the page calm even if something unexpected breaks
            turn["error"] = f"{type(exc).__name__}: {exc}"
            status.update(label="Something went wrong", state="error")
    st.session_state.turns.append(turn)
    st.rerun()


def render_reading(turn: dict, state: dict, is_latest: bool) -> None:
    """For photos: show what was read, and let the learner correct it."""
    math = state.get("math")
    if math:
        st.markdown("**I read your problem as:**")
        st.latex(math["problem_latex"])
    readings = unconfirmed_readings(state)
    if readings and is_latest:
        st.markdown("I *think* I might have seen this, but I couldn't confirm it. Is it right?")
        cols = st.columns(min(len(readings), 3))
        for col, reading in zip(cols, readings):
            if col.button(f"Yes, it's {reading}", key=f"confirm-{turn['id']}-{reading}"):
                queue({"text": reading}, reading)
                st.rerun()
    if is_latest and (math or readings):
        with st.form(key=f"fix-{turn['id']}", border=False):
            fixed = st.text_input("Not quite right? Type the problem exactly as it's written:",
                                  placeholder="for example 3/4 + 1/8")
            if st.form_submit_button("Check this instead") and fixed.strip():
                queue({"text": fixed.strip()}, fixed.strip())
                st.rerun()


def render_context_request(turn: dict, state: dict) -> None:
    st.info(state["message"])
    cols = st.columns(3)
    for i, topic in enumerate(TOPIC_CHOICES):
        if cols[i % 3].button(topic, key=f"topic-{turn['id']}-{i}", width="stretch"):
            queue({"requested_topic": topic}, f"Help with {topic.lower()}")
            st.rerun()
    with st.form(key=f"context-{turn['id']}", border=False):
        reply = st.text_input("Or tell me in your own words:",
                              placeholder="e.g. adding fractions, or 2x + 3 = 11")
        if st.form_submit_button("Continue") and reply.strip():
            queue({"text": reply.strip()}, reply.strip())
            st.rerun()


def render_details(state: dict) -> None:
    attempts = state.get("attempts") or []
    guard = state.get("guard") or {}
    if not (attempts or state.get("math") or guard.get("retried") or guard.get("removed_numbers")):
        return  # nothing worth showing, e.g. a typed topic question
    with st.expander("How I checked this"):
        for a in attempts:
            if a["error"]:
                st.markdown(f"- `{a['model']}` couldn't run: {a['error'][:160]}")
                continue
            ex = a["extraction"]
            line = f"- `{a['model']}` read **{ex['legibility']}**"
            if ex.get("expression") or ex.get("unconfirmed"):
                line += f": `{ex.get('expression') or ex.get('unconfirmed')}`"
            if a["transcription"] is not None:
                verdict = "agreed ✔️" if a["agreed"] else "did **not** agree, so I didn't trust the numbers"
                line += f"; a second independent read (`{a['transcription']}`) {verdict}"
            st.markdown(line)
        if state.get("math"):
            st.markdown("- Every calculation and the answer came from SymPy, not from the AI.")
        if guard.get("retried"):
            st.markdown("- The explanation first included a number the calculator hadn't produced, "
                        "so I asked for a rewrite.")
        if guard.get("removed_numbers"):
            st.markdown(f"- Removed sentences containing unverified numbers: {', '.join(guard['removed_numbers'])}.")


def render_turn(turn: dict, is_latest: bool) -> None:
    with st.chat_message("user", avatar="🙂"):
        if turn.get("image"):
            st.image(turn["image"], width=280)
        else:
            st.markdown(turn["label"])
    with st.chat_message("assistant", avatar="🐢"):
        if "error" in turn:
            st.error("I couldn't reach the local AI models. Please check that Ollama is running "
                     f"(`ollama serve`) and try again.\n\nDetails: {turn['error'][:300]}")
            return
        state = turn["state"]
        if turn.get("image"):
            render_reading(turn, state, is_latest)
        if state.get("status") == "needs_context":
            if is_latest:
                render_context_request(turn, state)
            else:
                st.markdown(state["message"])
        else:
            st.markdown(state.get("explanation", ""))
        render_details(state)


def input_area() -> None:
    photo_tab, type_tab = st.tabs(["📷  Snap or upload a photo", "✍️  Type a question"])
    with photo_tab:
        uploaded = st.file_uploader("A photo of your math problem (handwritten or printed)",
                                    type=["png", "jpg", "jpeg", "webp"], key="uploader")
        # camera_input turns the camera on as soon as it renders (even in a collapsed
        # expander), so only render it once the learner opts in.
        snapped = None
        if st.toggle("Use my camera instead", key="use-camera"):
            snapped = st.camera_input("Take a picture", key="camera", label_visibility="collapsed")
        image = snapped or uploaded
        if st.button("Help me understand this", type="primary", disabled=image is None, key="go-photo"):
            queue({"image": image.getvalue()}, "📷 photo")
            st.rerun()
    with type_tab:
        with st.form("typed", border=False, clear_on_submit=True):
            text = st.text_area("Type a problem, a word problem, or a topic you'd like to understand",
                                placeholder="e.g.  2x + 3 = 11   ·   What is 25% of 40?   ·   I don't get fractions",
                                height=100)
            if st.form_submit_button("Help me understand this", type="primary") and text.strip():
                queue({"text": text.strip()}, text.strip())
                st.rerun()


def main() -> None:
    st.session_state.setdefault("turns", [])
    sidebar()
    st.title("🐢 Slow & Steady Math Tutor")
    st.markdown("<span class='tutor-note'>No rush, no judgement. We'll go one small step at a time.</span>",
                unsafe_allow_html=True)

    turns = st.session_state.turns
    for i, turn in enumerate(turns):
        render_turn(turn, is_latest=i == len(turns) - 1)
    run_pending()

    st.divider()
    st.markdown("#### " + ("Ask about something else" if turns else "What are we working on today?"))
    input_area()


main()
