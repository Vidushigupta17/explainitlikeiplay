"""
app.py
Streamlit entry point for "Explain It Like I Play".

Run with:  streamlit run app.py
"""

import streamlit as st

from config import (
    GAMES,
    SUGGESTED_TOPICS,
    DIFFICULTIES,
    XP_CORRECT_NO_HINT,
    XP_CORRECT_WITH_HINT,
    XP_MISSION_COMPLETE_BONUS,
    get_level_info,
)
from llm_client import generate_explanation, generate_quiz, LLMConfigError
import prompts
import mission_engine

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
    # --- Challenge (quiz) mode state ---
    "quiz": None,               # list of 5 question dicts once generated
    "quiz_active": False,       # True while the challenge UI is showing
    "quiz_index": 0,            # index of the question currently being shown
    "quiz_answers": {},         # {question_index: selected_option_index}
    "quiz_score": 0,            # correct count, set once the challenge completes
    "quiz_complete": False,     # True once all 5 questions are answered
    "quiz_error": None,
    # --- Gamification state ---
    "xp": 0,                    # total XP, persists across challenges this session
    "quiz_hints_used": {},      # {question_index: True} if hint was revealed
    "quiz_streak": 0,           # current run of consecutive correct answers
    "quiz_best_streak": 0,      # longest streak reached during this challenge
    "quiz_xp_earned": 0,        # XP earned during the current/most recent challenge
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
    """Generates a fresh 5-question challenge and resets all progress + gamification state."""
    st.session_state.quiz_error = None
    try:
        with st.spinner("Preparing your challenge questions..."):
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
        st.session_state.quiz_hints_used = {}
        st.session_state.quiz_streak = 0
        st.session_state.quiz_best_streak = 0
        st.session_state.quiz_xp_earned = 0
    except LLMConfigError as e:
        st.session_state.quiz_error = str(e)
    except Exception as e:  # noqa: BLE001 — surface any API/parsing error to the user
        st.session_state.quiz_error = f"Something went wrong generating the challenge: {e}"


def reset_quiz():
    """Clears challenge state and returns to the explanation view. XP earned is kept."""
    st.session_state.quiz = None
    st.session_state.quiz_active = False
    st.session_state.quiz_index = 0
    st.session_state.quiz_answers = {}
    st.session_state.quiz_score = 0
    st.session_state.quiz_complete = False
    st.session_state.quiz_error = None
    st.session_state.quiz_hints_used = {}
    st.session_state.quiz_streak = 0
    st.session_state.quiz_best_streak = 0
    st.session_state.quiz_xp_earned = 0


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🎮 Explain It Like I Play")
st.caption(
    "Pick a game you know and a CS/engineering topic you don't (yet). "
    "Get an explanation grounded in the game's real mechanics — without losing technical accuracy."
)

# --- Player status HUD (XP / Level / Mastery Progress) ---
level_info = get_level_info(st.session_state.xp)
hud1, hud2 = st.columns(2)
with hud1:
    st.metric("⭐ XP", st.session_state.xp)
with hud2:
    st.metric("🏅 Level", f"{level_info['level_num']} — {level_info['level_name']}")

if level_info["next_threshold"] is not None:
    span = level_info["next_threshold"] - level_info["current_threshold"]
    progress = (st.session_state.xp - level_info["current_threshold"]) / span if span else 1.0
    st.progress(
        min(max(progress, 0.0), 1.0),
        text=f"Mastery Progress — {st.session_state.xp - level_info['current_threshold']}/{span} XP "
        f"to next level",
    )
else:
    st.progress(1.0, text="Mastery Progress — Max level reached! 🏅")

st.divider()

# ----------------------------------------------------------------------
# Top-level mode tabs — the original explain/quiz flow lives untouched
# inside the first tab. The second tab is the universal Game Arena: a
# topic router that launches CPU Arena for CPU Scheduling and an
# AI-generated mission (mission_engine.py) for every other topic, so
# this tab is no longer CPU-Arena-only.
# ----------------------------------------------------------------------
tab_learn, tab_arena = st.tabs(["🎓 Explain It Like I Play", "🎮 Game Arena"])

with tab_arena:
    mission_engine.render_game_arena(SUGGESTED_TOPICS, DIFFICULTIES)

with tab_learn:
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
                hint_used = st.session_state.quiz_hints_used.get(idx, False)

                st.subheader("🎯 Challenge")
                st.caption(f"Question {idx + 1} of {total}  •  🔥 Streak: {st.session_state.quiz_streak}")
                st.progress((idx) / total, text=f"Mastery Progress — {idx}/{total} answered")
                st.markdown(f"**{q['question']}**")

                choice = st.radio(
                    "Choose an answer:",
                    q["options"],
                    index=None,
                    key=f"quiz_choice_{idx}",
                    label_visibility="collapsed",
                )

                hint_col, _ = st.columns([1, 3])
                with hint_col:
                    if st.button("💡 Hint", key=f"hint_btn_{idx}", disabled=hint_used):
                        st.session_state.quiz_hints_used[idx] = True
                        st.rerun()
                if hint_used:
                    fallback_hint = (
                        "Think back to how this concept was mapped onto the game analogy above."
                    )
                    st.info(f"💡 {q.get('hint') or fallback_hint}")

                is_last = idx == total - 1
                button_label = "🏁 Complete Mission" if is_last else "➡️ Next Question"
                if st.button(button_label, use_container_width=True):
                    if choice is None:
                        st.warning("Pick an answer before continuing.")
                    else:
                        selected_index = q["options"].index(choice)
                        st.session_state.quiz_answers[idx] = selected_index
                        is_correct = selected_index == q["correct_index"]

                        # --- XP + streak scoring for this question ---
                        if is_correct:
                            earned = XP_CORRECT_WITH_HINT if hint_used else XP_CORRECT_NO_HINT
                            st.session_state.xp += earned
                            st.session_state.quiz_xp_earned += earned
                            st.session_state.quiz_streak += 1
                            st.session_state.quiz_best_streak = max(
                                st.session_state.quiz_best_streak, st.session_state.quiz_streak
                            )
                        else:
                            st.session_state.quiz_streak = 0

                        if is_last:
                            score = sum(
                                1
                                for i, qq in enumerate(st.session_state.quiz)
                                if st.session_state.quiz_answers.get(i) == qq["correct_index"]
                            )
                            st.session_state.quiz_score = score
                            # --- Mission Complete bonus, awarded once ---
                            st.session_state.xp += XP_MISSION_COMPLETE_BONUS
                            st.session_state.quiz_xp_earned += XP_MISSION_COMPLETE_BONUS
                            st.session_state.quiz_complete = True
                        else:
                            st.session_state.quiz_index += 1
                        st.rerun()

            else:
                # ---- Results view ----
                total = len(st.session_state.quiz)
                score = st.session_state.quiz_score
                accuracy = round((score / total) * 100) if total else 0
                level_info = get_level_info(st.session_state.xp)

                st.subheader("🏆 MISSION COMPLETE")
                st.markdown(f"⭐ **XP Earned:** +{st.session_state.quiz_xp_earned}")
                st.markdown(f"🎯 **Accuracy:** {accuracy}% ({score}/{total} correct)")
                st.markdown(f"🔥 **Current Streak:** {st.session_state.quiz_best_streak}")
                st.markdown(f"🏅 **Level:** {level_info['level_name']}")

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
                    if st.button("🔁 Retry Mission", use_container_width=True):
                        start_quiz(g, t, d, data)
                        st.rerun()
                with rc2:
                    if st.button("✖️ Close Challenge", use_container_width=True):
                        reset_quiz()
                        st.rerun()
    else:
        st.info("Pick a game, a topic, and a difficulty, then hit **Explain** to get started.")