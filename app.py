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
from tutor.latex_input import LatexInputError, latex_to_problem
from tutor.math_keyboard import math_input
from tutor.ui import PAGE_CSS, lesson_html

st.set_page_config(page_title="Slow & Steady Math Tutor", page_icon="🐢", layout="centered")

st.markdown(PAGE_CSS, unsafe_allow_html=True)

# Friendly progress lines, shown while each graph node runs.
PROGRESS = {
    "precheck": "Taking a first look at your photo…",
    "read_primary": "Reading your problem carefully…",
    "read_backup": "Taking a second, closer look…",
    "read_text": "Reading your question…",
    "solve": "Checking the math with my calculator…",
    "formulate": "Setting the problem up as equations, twice, to be sure… (this one takes a little longer)",
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
    # Hidden for now; uncomment to show the intro line in the sidebar again.
    # st.sidebar.markdown(
    #     "<span class='tutor-note'>Snap a photo or type a question. "
    #     "Every number is checked by a calculator (SymPy), never guessed by the AI.</span>",
    #     unsafe_allow_html=True,
    # )
    st.sidebar.divider()
    models = installed_models()
    if models is None:
        st.sidebar.error("Ollama isn't running. Start it with `ollama serve`, then refresh.")
        return
    broken = get_services()._broken_models
    for label, name in [("Chat model", config.CHAT_MODEL), ("Vision model", config.VISION_MODEL),
                        ("Backup vision", config.BACKUP_VISION_MODEL)]:
        if not name:
            continue  # no backup vision model configured
        if name in broken:
            st.sidebar.warning(f"{label} can't load · `{name}`  \nSkipping it; photos use the main vision model only.")
        elif name in models or name.removesuffix(":latest") in models:
            st.sidebar.success(f"{label} ready · `{name}`")
        else:
            st.sidebar.warning(f"{label} not installed · `{name}`  \nRun `ollama pull {name}`")
    st.sidebar.divider()
    st.sidebar.toggle("Handwritten whiteboard style", value=True, key="handwritten")
    if st.sidebar.button("Start fresh", width="stretch"):
        st.session_state.turns = []
        st.rerun()


def queue(inputs: dict, label: str, latex: str = "") -> None:
    """Remember a request; it runs on the next script pass, with progress shown."""
    st.session_state.pending = {"inputs": inputs, "label": label, "latex": latex}


def keyboard_entry(key: str, label: str, button: str = "Solve this") -> None:
    """The math keyboard; a submission is converted and queued like a typed question."""
    latex = math_input(key, label=label, button=button)
    if not latex:
        return
    try:
        problem = latex_to_problem(latex)
    except LatexInputError as exc:
        st.warning(f"{exc} You can also type it in words instead.")
        return
    queue({"text": problem}, problem, latex=latex)
    st.rerun()


def run_pending() -> None:
    pending = st.session_state.pop("pending", None)
    if not pending:
        return
    turn = {"id": uuid.uuid4().hex, "label": pending["label"], "image": pending["inputs"].get("image"),
            "latex": pending.get("latex", "")}
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
        with st.expander("Not quite right? Write it with the math keyboard"):
            keyboard_entry(f"fix-kb-{turn['id']}", "Write the problem exactly as it is in your photo:", "Check this instead")
        with st.form(key=f"fix-{turn['id']}", border=False):
            fixed = st.text_input("Or type it:",
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
    with st.expander("Write the problem with the math keyboard (∫, √, Σ, matrices…)"):
        keyboard_entry(f"context-kb-{turn['id']}", "Write your problem here:")
    with st.form(key=f"context-{turn['id']}", border=False):
        reply = st.text_input("Or tell me in your own words:",
                              placeholder="e.g. adding fractions, or 2x + 3 = 11")
        if st.form_submit_button("Continue") and reply.strip():
            queue({"text": reply.strip()}, reply.strip())
            st.rerun()


def render_details(state: dict) -> None:
    attempts = state.get("attempts") or []
    guard = state.get("guard") or {}
    if not (attempts or state.get("math") or state.get("setups") or guard.get("retried")
            or guard.get("removed_numbers")):
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
        if state.get("diagram"):
            st.markdown("- What I saw in the diagram:")
            st.code(state["diagram"], language=None)
        for i, setup in enumerate(state.get("setups") or [], 1):
            result = setup.get("result", {})
            if setup.get("error"):
                st.markdown(f"- Setup {i} couldn't run: {setup['error'][:160]}")
            elif result.get("ok"):
                st.markdown(f"- Setup {i} (`{setup.get('model', '?')}`): {result['answer']}")
            else:
                st.markdown(f"- Setup {i} (`{setup.get('model', '?')}`) didn't work out: {result.get('error', '')}")
        if state.get("setups"):
            verdict = ("All setups agreed" + (" and the answer matches one of the choices" if any(
                s.get("choices") for s in state["setups"]) else "") + ", so I trusted it.") \
                if state.get("setup") else "The setups didn't agree, so I didn't trust any answer."
            st.markdown(f"- {verdict}")
        if state.get("math"):
            st.markdown("- Every calculation and the answer came from SymPy, not from the AI.")
        if guard.get("retried"):
            st.markdown("- The explanation first included a number the calculator hadn't produced, "
                        "so I asked for a rewrite.")
        if guard.get("removed_numbers"):
            st.markdown(f"- Removed sentences containing unverified numbers: {', '.join(guard['removed_numbers'])}.")


def render_diagram_fix(turn: dict, state: dict) -> None:
    """Small vision models misread diagrams; let the learner correct the description and retry."""
    setups = state.get("setups") or []
    if any(s.get("result", {}).get("ok") for s in setups):
        why = "My two attempts at setting it up came out differently, so I didn't trust either answer."
    else:
        why = "I couldn't turn it into something my calculator can check."
    st.markdown(f"{why} **Here's what I read from your photo.** If anything is wrong or missing, "
                "fix it and I'll try again:")
    with st.form(key=f"diagram-{turn['id']}", border=False):
        fixed = st.text_area("What I read", value=state["diagram"], height=220,
                             label_visibility="collapsed")
        if st.form_submit_button("Try again with this", type="primary") and fixed.strip():
            problem = state["extraction"].get("problem_text") or "Solve the problem in the diagram."
            queue({"text": problem, "diagram": fixed.strip()}, "📐 corrected diagram description")
            st.rerun()


def render_turn(turn: dict, is_latest: bool) -> None:
    with st.chat_message("user", avatar="🙂"):
        if turn.get("image"):
            st.image(turn["image"], width=280)
        elif turn.get("latex"):
            st.latex(turn["latex"])  # what the learner wrote with the math keyboard, as real notation
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
        if is_latest and state.get("diagram") and state.get("setups") and state.get("status") != "solved":
            render_diagram_fix(turn, state)
        if state.get("status") == "needs_context":
            if is_latest:
                render_context_request(turn, state)
            else:
                st.markdown(state["message"])
        else:
            st.markdown(lesson_html(state.get("explanation", ""), st.session_state.get("handwritten", True)),
                        unsafe_allow_html=True)
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
        if st.toggle("Use the math keyboard (∫, d/dx, lim, Σ, √, matrices, Greek…)", key="use-keyboard"):
            keyboard_entry("typed-kb", "Write your problem. Tap ⌨️ Keyboard for every math symbol:")
            return
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
