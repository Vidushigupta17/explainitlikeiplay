"""
app.py
Streamlit entry point for "Explain It Like I Play".

Run with:  streamlit run app.py
"""

import streamlit as st

from config import GAMES, SUGGESTED_TOPICS, DIFFICULTIES
from llm_client import generate_explanation, generate_quiz, LLMConfigError
import prompts

st.set_page_config(
    page_title="Explain It Like I Play",
    page_icon="🎮",
    layout="centered",
)

# ----------------------------------------------------------------------
# Session state setup — this is what lets buttons re-run the script
# without wiping out previous results or re-calling the LLM by accident.
# ----------------------------------------------------------------------
defaults = {
    "explanation": None,       # dict: game_analogy / technical_mapping / technical_explanation / key_takeaways
    "locked_game": None,       # game used for the current explanation
    "locked_topic": None,      # topic used for the current explanation
    "locked_difficulty": None, # difficulty used for the current explanation
    "choosing_game": True,     # controls whether the game selector is active (for "Change Game")
    "error": None,
    # --- Quiz ("Challenge") mode state ---
    "quiz": None,               # list of 5 question dicts once generated
    "quiz_active": False,       # True while the challenge UI is showing
    "quiz_index": 0,            # index of the question currently being shown
    "quiz_answers": {},         # {question_index: selected_option_index}
    "quiz_score": 0,            # correct count, set once the quiz completes
    "quiz_complete": False,     # True once all 5 questions are answered
    "quiz_error": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def run_llm(system_prompt: str, user_prompt: str):
    """Calls the LLM, stores result or error in session_state."""
    st.session_state.error = None
    try:
        with st.spinner("Thinking through the analogy..."):
            result = generate_explanation(system_prompt, user_prompt)
        st.session_state.explanation = result
    except LLMConfigError as e:
        st.session_state.error = str(e)
    except Exception as e:  # noqa: BLE001 — surface any API/parsing error to the user
        st.session_state.error = f"Something went wrong generating the explanation: {e}"


def start_quiz(game: str, topic: str, difficulty: str, explanation: dict):
    """Generates a fresh 5-question quiz and resets all quiz progress state."""
    st.session_state.quiz_error = None
    try:
        with st.spinner("Writing your challenge questions..."):
            questions = generate_quiz(
                prompts.QUIZ_SYSTEM_PROMPT,
                prompts.build_quiz_prompt(game, topic, difficulty, explanation),
            )
        st.session_state.quiz = questions
        st.session_state.quiz_active = True
        st.session_state.quiz_index = 0
        st.session_state.quiz_answers = {}
        st.session_state.quiz_score = 0
        st.session_state.quiz_complete = False
    except LLMConfigError as e:
        st.session_state.quiz_error = str(e)
    except Exception as e:  # noqa: BLE001 — surface any API/parsing error to the user
        st.session_state.quiz_error = f"Something went wrong generating the challenge: {e}"


def reset_quiz():
    """Clears quiz state and returns to the explanation view."""
    st.session_state.quiz = None
    st.session_state.quiz_active = False
    st.session_state.quiz_index = 0
    st.session_state.quiz_answers = {}
    st.session_state.quiz_score = 0
    st.session_state.quiz_complete = False
    st.session_state.quiz_error = None


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🎮 Explain It Like I Play")
st.caption(
    "Pick a game you know and a CS/engineering topic you don't (yet). "
    "Get an explanation grounded in the game's real mechanics — without losing technical accuracy."
)
st.divider()

# ----------------------------------------------------------------------
# Selectors
# ----------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    game = st.selectbox(
        "🕹️ Choose a game",
        GAMES,
        index=GAMES.index(st.session_state.locked_game) if st.session_state.locked_game in GAMES else 0,
        disabled=not st.session_state.choosing_game and st.session_state.explanation is not None,
    )

with col2:
    difficulty = st.selectbox(
        "📈 Difficulty",
        DIFFICULTIES,
        index=DIFFICULTIES.index(st.session_state.locked_difficulty)
        if st.session_state.locked_difficulty in DIFFICULTIES
        else 0,
    )

topic_choice = st.selectbox("🧠 Topic", SUGGESTED_TOPICS + ["Custom topic..."])
if topic_choice == "Custom topic...":
    topic = st.text_input(
        "Enter your own topic",
        value=st.session_state.locked_topic if st.session_state.locked_topic not in SUGGESTED_TOPICS else "",
        placeholder="e.g. Race conditions, REST APIs, Big-O notation...",
    )
else:
    topic = topic_choice

explain_clicked = st.button("✨ Explain", type="primary", use_container_width=True)

if explain_clicked:
    if not topic or not topic.strip():
        st.session_state.error = "Please enter or select a topic first."
    else:
        st.session_state.locked_game = game
        st.session_state.locked_topic = topic.strip()
        st.session_state.locked_difficulty = difficulty
        st.session_state.choosing_game = False
        run_llm(
            prompts.SYSTEM_PROMPT,
            prompts.build_initial_prompt(game, topic.strip(), difficulty),
        )

# ----------------------------------------------------------------------
# Error display
# ----------------------------------------------------------------------
if st.session_state.error:
    st.error(st.session_state.error)

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
if st.session_state.explanation:
    data = st.session_state.explanation
    g = st.session_state.locked_game
    t = st.session_state.locked_topic
    d = st.session_state.locked_difficulty

    st.divider()
    st.subheader(f"{t} — explained through {g} ({d})")

    st.markdown("### 🎮 Game Analogy")
    st.write(data.get("game_analogy", ""))

    st.markdown("### 🔗 Technical Mapping")
    st.markdown(data.get("technical_mapping", ""))

    st.markdown("### 📚 Actual Technical Explanation")
    st.write(data.get("technical_explanation", ""))

    st.markdown("### 💡 Key Takeaways")
    st.markdown(data.get("key_takeaways", ""))

    st.divider()
    st.caption("Not quite right, or want another angle?")

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        if st.button("🔄 Explain Again", use_container_width=True):
            reset_quiz()  # explanation is changing, so any quiz tied to it is stale
            run_llm(
                prompts.SYSTEM_PROMPT,
                prompts.build_regenerate_prompt(g, t, d, data.get("game_analogy", "")),
            )
            st.rerun()

    with b2:
        if st.button("🧒 Explain Simpler", use_container_width=True):
            reset_quiz()
            run_llm(
                prompts.SYSTEM_PROMPT,
                prompts.build_simplify_prompt(g, t, d, data),
            )
            st.rerun()

    with b3:
        if st.button("📖 Explain Technically", use_container_width=True):
            reset_quiz()
            run_llm(
                prompts.SYSTEM_PROMPT,
                prompts.build_technical_prompt(g, t, d, data),
            )
            st.rerun()

    with b4:
        if st.button("🎮 Change Game", use_container_width=True):
            st.session_state.explanation = None
            st.session_state.choosing_game = True
            st.session_state.error = None
            reset_quiz()
            st.rerun()

    # --------------------------------------------------------------
    # Challenge / Quiz mode
    # --------------------------------------------------------------
    if not st.session_state.quiz_active:
        st.divider()
        if st.button("🎯 Start Challenge", use_container_width=True):
            start_quiz(g, t, d, data)
            st.rerun()

    if st.session_state.quiz_error:
        st.error(st.session_state.quiz_error)

    if st.session_state.quiz_active and st.session_state.quiz:
        st.divider()

        if not st.session_state.quiz_complete:
            # ---- Question view ----
            idx = st.session_state.quiz_index
            total = len(st.session_state.quiz)
            q = st.session_state.quiz[idx]

            st.subheader("🎯 Challenge")
            st.caption(f"Question {idx + 1} of {total}")
            st.markdown(f"**{q['question']}**")

            choice = st.radio(
                "Choose an answer:",
                q["options"],
                index=None,
                key=f"quiz_choice_{idx}",
                label_visibility="collapsed",
            )

            is_last = idx == total - 1
            if st.button("Finish Challenge" if is_last else "Next Question", use_container_width=True):
                if choice is None:
                    st.warning("Pick an answer before continuing.")
                else:
                    st.session_state.quiz_answers[idx] = q["options"].index(choice)
                    if is_last:
                        score = sum(
                            1
                            for i, qq in enumerate(st.session_state.quiz)
                            if st.session_state.quiz_answers.get(i) == qq["correct_index"]
                        )
                        st.session_state.quiz_score = score
                        st.session_state.quiz_complete = True
                    else:
                        st.session_state.quiz_index += 1
                    st.rerun()

        else:
            # ---- Results view ----
            total = len(st.session_state.quiz)
            score = st.session_state.quiz_score
            accuracy = round((score / total) * 100) if total else 0

            st.subheader("🏆 CHALLENGE COMPLETE")
            st.markdown(f"**Score:** {score}/{total}")
            st.markdown(f"**Accuracy:** {accuracy}%")

            incorrect = [
                (i, qq)
                for i, qq in enumerate(st.session_state.quiz)
                if st.session_state.quiz_answers.get(i) != qq["correct_index"]
            ]

            if incorrect:
                st.markdown("#### Review — Questions to revisit")
                for i, qq in incorrect:
                    your_idx = st.session_state.quiz_answers.get(i)
                    your_answer = qq["options"][your_idx] if your_idx is not None else "(no answer)"
                    correct_answer = qq["options"][qq["correct_index"]]
                    with st.expander(f"Q{i + 1}. {qq['question']}"):
                        st.write(f"❌ Your answer: {your_answer}")
                        st.write(f"✅ Correct answer: {correct_answer}")
                        st.write(qq.get("explanation", ""))
            else:
                st.success("Perfect score — nice work! 🎉")

            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("🔁 Retake Challenge", use_container_width=True):
                    start_quiz(g, t, d, data)
                    st.rerun()
            with rc2:
                if st.button("✖️ Close Challenge", use_container_width=True):
                    reset_quiz()
                    st.rerun()
elif not st.session_state.error:
    st.info("Pick a game, a topic, and a difficulty, then hit **Explain** to get started.")
