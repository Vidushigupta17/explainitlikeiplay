"""
mission_engine.py
Game Arena — a topic-agnostic mission system.

  User Topic
    -> AI analyzes the topic & designs a mission (ONE-TIME content call)
    -> Student plays the mission (pure Python — no LLM in the loop)
    -> Game records performance (pure Python)
    -> AI analyzes which concept the player is weak on -> targeted "Explain Again"
    -> Retry Mission
    -> Mastery Score

--------------------------------------------------------------------------
UNIVERSAL MISSION INTERFACE
--------------------------------------------------------------------------
Every mission — whether it's the hand-built CPU Arena or an AI-generated
mission for an arbitrary topic — is described by the same shape, so the
rest of the app (HUD, XP, "Explain Again", mastery tracking) can treat
them uniformly:

    {
        "topic":        str,          # the subject being taught
        "game_title":   str,          # display name of the mission
        "objective":    str,          # one-line learning objective
        "instructions": str,          # how to play
        "concepts":     [str, ...],   # sub-concepts this mission tests
        "metrics":      [str, ...],   # what the mission measures
    }

CPU Arena exposes this shape via cpu_arena.get_universal_mission_meta().
AI-generated missions carry these same fields (plus "steps", the
decision-game content) inside the dict returned by
llm_client.generate_mission().

--------------------------------------------------------------------------
ROUTING
--------------------------------------------------------------------------
    Topic
      -> Check specialized modules
      -> CPU Scheduling -> CPU Arena (existing, hand-tuned, deterministic)
      -> Anything else  -> Generic Mission Engine (this file)

To add a future specialized game for another topic, add its normalized
name(s) to SPECIALIZED_TOPICS below and route it to its own module the
same way CPU Arena is routed — no other file needs to change.

--------------------------------------------------------------------------
WHAT THE LLM DOES vs. WHAT PYTHON DOES
--------------------------------------------------------------------------
The LLM is called exactly ONCE per mission (plus once more only if the
player asks for a fresh "Retry Mission", and once more only if they ask
for a weak-concept "Explain Again"). At mission-design time, the LLM
supplies the scenario text, the answer options, and which option is
correct (`correct_index`) — but it never sees or judges a live player
action. Every real-time check — was this choice correct, what's the
score, which concept is weakest, what's the mastery % — is plain
Python comparing the player's click to the pre-recorded correct_index.
"""

import streamlit as st

from config import XP_CORRECT_NO_HINT, XP_CORRECT_WITH_HINT, XP_MISSION_COMPLETE_BONUS
from llm_client import (
    generate_mission,
    generate_decision_mission,
    generate_ordering_mission,
    generate_simulation_mission,
    generate_explanation,
    LLMConfigError,
)
import prompts
import cpu_arena
import game_state
import decision_game
import ordering_game
import simulation_game

# Topic keywords that suggest the Simulation Game (perform operations on a
# live state, e.g. a stack or queue) is the best fit. Checked BEFORE the
# Ordering Game keywords below, since these topics are more specifically
# "live state you operate on" than "sequence you arrange". Purely a hint
# for which mission generator to try first; if generation fails or
# doesn't validate, _start_mission() falls back gracefully, same as the
# existing Decision Game / Ordering Game fallback chain.
SIMULATION_TOPIC_KEYWORDS = ("stack", "queue", "buffer", "cache")

# Topic keywords that suggest the Ordering Game (arrange items into a
# correct sequence) is a better fit than the default Decision Game —
# e.g. sorting algorithms, CPU scheduling, algorithm steps, networking
# sequences, mathematical procedures. Purely a hint for which mission
# generator to try first; if generation fails or doesn't validate,
# _start_mission() falls back the same way it already does for Decision
# Game failures.
ORDERING_TOPIC_KEYWORDS = ("sort", "order", "schedul", "sequence", "step", "procedure", "algorithm")

# Normalized (lowercased, stripped) topic strings that have a specialized,
# hand-built game module instead of the generic AI-generated one.
SPECIALIZED_TOPICS = {
    "cpu scheduling": cpu_arena,
    "os scheduling": cpu_arena,
    "process scheduling": cpu_arena,
    "cpu arena": cpu_arena,
}

GAME_ARENA_KEYS = [
    "mission_spec",
    "mission_index",
    "mission_answers",
    "mission_concept_stats",
    "mission_score",
    "mission_hints_used",
    "mission_complete",
    "mission_error",
    "mission_last_feedback",
    "mission_weak_concept",
    "mission_focus_explanation",
    "mission_focus_error",
    "mission_xp_earned",
]


def _normalize(topic: str) -> str:
    return (topic or "").strip().lower()


def is_specialized(topic: str) -> bool:
    return _normalize(topic) in SPECIALIZED_TOPICS


def _prefers_simulation(topic: str) -> bool:
    """Heuristic only — decides which mission generator to TRY FIRST, not
    whether one is allowed to run. Never affects scoring/correctness."""
    normalized = _normalize(topic)
    return any(keyword in normalized for keyword in SIMULATION_TOPIC_KEYWORDS)


def _prefers_ordering(topic: str) -> bool:
    """Heuristic only — decides which mission generator to TRY FIRST, not
    whether one is allowed to run. Never affects scoring/correctness."""
    normalized = _normalize(topic)
    return any(keyword in normalized for keyword in ORDERING_TOPIC_KEYWORDS)


def _init_router_state():
    st.session_state.setdefault("game_arena_locked_topic", None)
    st.session_state.setdefault("game_arena_locked_difficulty", None)
    st.session_state.setdefault("game_arena_choosing", True)


def _init_mission_state():
    for key in GAME_ARENA_KEYS:
        st.session_state.setdefault(key, None)
    st.session_state.setdefault("mission_index", 0)
    st.session_state.setdefault("mission_answers", {})
    st.session_state.setdefault("mission_concept_stats", {})
    st.session_state.setdefault("mission_score", 0)
    st.session_state.setdefault("mission_hints_used", {})
    st.session_state.setdefault("mission_complete", False)
    st.session_state.setdefault("mission_xp_earned", 0)


def _reset_mission_progress():
    """Clears in-progress mission state but keeps the loaded spec (used by Retry)."""
    st.session_state.mission_index = 0
    st.session_state.mission_answers = {}
    st.session_state.mission_concept_stats = {}
    st.session_state.mission_score = 0
    st.session_state.mission_hints_used = {}
    st.session_state.mission_complete = False
    st.session_state.mission_last_feedback = None
    st.session_state.mission_weak_concept = None
    st.session_state.mission_focus_explanation = None
    st.session_state.mission_focus_error = None
    st.session_state.mission_xp_earned = 0
    simulation_game.reset_runtime()


def _clear_mission():
    """Fully clears the mission, including the loaded spec (used by Change Topic)."""
    for key in GAME_ARENA_KEYS:
        st.session_state[key] = None
    _init_mission_state()
    game_state.reset_game_state()
    simulation_game.reset_runtime()


def _change_topic():
    _clear_mission()
    st.session_state.game_arena_locked_topic = None
    st.session_state.game_arena_locked_difficulty = None
    st.session_state.game_arena_choosing = True
    if "arena_started" in st.session_state:
        cpu_arena.reset_arena()


# ----------------------------------------------------------------------
# Deterministic mission logic — NO LLM involved anywhere below
# ----------------------------------------------------------------------
def _start_mission(topic: str, difficulty: str, previous_title: str = None):
    """Calls the LLM to design mission content, then hands control fully
    to deterministic Python for gameplay.

    Tries the small, single-scenario Decision Game schema first
    (llm_client.generate_decision_mission). If the LLM's response
    doesn't validate — missing fields, wrong game_type, empty actions,
    or a correct_action that isn't one of the offered actions — that
    failure is swallowed here and generation falls back to the original
    5-step mission generator, so a bad Decision Game response degrades
    gracefully instead of showing an error or crashing."""
    st.session_state.mission_error = None
    try:
        spec = None
        with st.spinner("Designing your mission..."):
            if _prefers_simulation(topic):
                try:
                    simulation_prompt = prompts.build_simulation_mission_prompt(topic, difficulty, previous_title)
                    spec = generate_simulation_mission(
                        prompts.SIMULATION_MISSION_SYSTEM_PROMPT, simulation_prompt, topic, difficulty
                    )
                except LLMConfigError:
                    raise  # missing/invalid API key — nothing to fall back to, surface it
                except Exception:
                    spec = None  # invalid Simulation Game spec — fall back below, don't crash

            if spec is None and _prefers_ordering(topic):
                try:
                    ordering_prompt = prompts.build_ordering_mission_prompt(topic, difficulty, previous_title)
                    spec = generate_ordering_mission(
                        prompts.ORDERING_MISSION_SYSTEM_PROMPT, ordering_prompt, topic, difficulty
                    )
                except LLMConfigError:
                    raise  # missing/invalid API key — nothing to fall back to, surface it
                except Exception:
                    spec = None  # invalid Ordering Game spec — fall back below, don't crash

            if spec is None:
                try:
                    decision_prompt = prompts.build_decision_mission_prompt(topic, difficulty, previous_title)
                    spec = generate_decision_mission(
                        prompts.DECISION_MISSION_SYSTEM_PROMPT, decision_prompt, topic, difficulty
                    )
                except LLMConfigError:
                    raise  # missing/invalid API key — nothing to fall back to, surface it
                except Exception:
                    spec = None  # invalid Decision Game spec — fall back below, don't crash

            if spec is None:
                if previous_title:
                    user_prompt = prompts.build_retry_mission_prompt(topic, difficulty, previous_title)
                else:
                    user_prompt = prompts.build_mission_prompt(topic, difficulty)
                spec = generate_mission(prompts.MISSION_SYSTEM_PROMPT, user_prompt)

        st.session_state.mission_spec = spec
        _reset_mission_progress()
        # Generic game-state layer (game_state.py) — drives the Decision
        # Game HUD regardless of which schema produced this spec.
        game_state.init_game_state(
            mission_id=spec.get("game_title", "mission"),
            topic=topic,
            game_type=spec.get("game_type", "decision"),
            objective=spec.get("objective", ""),
        )
    except LLMConfigError as e:
        st.session_state.mission_error = str(e)
    except Exception as e:  # noqa: BLE001 — surface any API/parsing error, never crash
        st.session_state.mission_error = f"Couldn't design a mission for this topic: {e}"


def _record_decision(step_idx: int, chosen_index: int):
    """Deterministically scores the player's choice against the
    pre-generated correct_index. No LLM call happens here."""
    spec = st.session_state.mission_spec
    step = spec["steps"][step_idx]
    result = decision_game.evaluate_action(step, chosen_index)
    is_correct = result["is_correct"]
    correct_index = result["correct_index"]
    concept = step["concept"]
    hint_used = st.session_state.mission_hints_used.get(step_idx, False)

    st.session_state.mission_answers[step_idx] = chosen_index

    stats = st.session_state.mission_concept_stats.setdefault(concept, {"correct": 0, "wrong": 0})
    if is_correct:
        stats["correct"] += 1
        st.session_state.mission_score += 1
        earned = XP_CORRECT_WITH_HINT if hint_used else XP_CORRECT_NO_HINT
        st.session_state.xp += earned
        st.session_state.mission_xp_earned += earned
    else:
        stats["wrong"] += 1

    st.session_state.mission_last_feedback = {
        "is_correct": is_correct,
        "correct_index": correct_index,
        "explanation": result["explanation"],
    }

    # Keep the generic game-state layer (game_state.py) in sync with real
    # gameplay — this is what now drives the Decision Game HUD (score,
    # attempts, progress), instead of just a preview panel.
    game_state.record_player_action(
        action=step["options"][chosen_index],
        is_correct=is_correct,
        detail=concept,
    )

    is_last = step_idx == len(spec["steps"]) - 1
    if is_last:
        st.session_state.xp += XP_MISSION_COMPLETE_BONUS
        st.session_state.mission_xp_earned += XP_MISSION_COMPLETE_BONUS
        st.session_state.mission_complete = True
        st.session_state.mission_weak_concept = _compute_weak_concept()
        game_state.complete_game()
    else:
        st.session_state.mission_index += 1


def _record_ordering(step_idx: int, chosen_order: list):
    """Deterministically scores the player's arrangement against the
    pre-generated correct_order. No LLM call happens here — mirrors
    _record_decision() above but for the Ordering Game."""
    spec = st.session_state.mission_spec
    step = spec["steps"][step_idx]
    result = ordering_game.evaluate_order(step, chosen_order)
    is_correct = result["is_correct"]
    concept = step["concept"]
    hint_used = st.session_state.mission_hints_used.get(step_idx, False)

    st.session_state.mission_answers[step_idx] = chosen_order

    stats = st.session_state.mission_concept_stats.setdefault(concept, {"correct": 0, "wrong": 0})
    if is_correct:
        stats["correct"] += 1
        st.session_state.mission_score += 1
        earned = XP_CORRECT_WITH_HINT if hint_used else XP_CORRECT_NO_HINT
        st.session_state.xp += earned
        st.session_state.mission_xp_earned += earned
    else:
        stats["wrong"] += 1

    st.session_state.mission_last_feedback = {
        "is_correct": is_correct,
        "correct_order": result["correct_order"],
        "explanation": result["explanation"],
    }

    # Keep the generic game-state layer (game_state.py) in sync, same as
    # the Decision Game does.
    items = step.get("items", [])
    game_state.record_player_action(
        action=" → ".join(items[i] for i in chosen_order),
        is_correct=is_correct,
        detail=concept,
    )

    is_last = step_idx == len(spec["steps"]) - 1
    if is_last:
        st.session_state.xp += XP_MISSION_COMPLETE_BONUS
        st.session_state.mission_xp_earned += XP_MISSION_COMPLETE_BONUS
        st.session_state.mission_complete = True
        st.session_state.mission_weak_concept = _compute_weak_concept()
        game_state.complete_game()
    else:
        st.session_state.mission_index += 1


def _record_simulation(step_idx: int, final_state: list):
    """Deterministically scores the player's final state against the
    pre-generated target_state. No LLM call happens here — mirrors
    _record_decision()/_record_ordering() above but for the Simulation
    Game."""
    spec = st.session_state.mission_spec
    step = spec["steps"][step_idx]
    result = simulation_game.evaluate_result(step, final_state)
    is_correct = result["is_correct"]
    concept = step["concept"]
    hint_used = st.session_state.mission_hints_used.get(step_idx, False)

    st.session_state.mission_answers[step_idx] = final_state

    stats = st.session_state.mission_concept_stats.setdefault(concept, {"correct": 0, "wrong": 0})
    if is_correct:
        stats["correct"] += 1
        st.session_state.mission_score += 1
        earned = XP_CORRECT_WITH_HINT if hint_used else XP_CORRECT_NO_HINT
        st.session_state.xp += earned
        st.session_state.mission_xp_earned += earned
    else:
        stats["wrong"] += 1

    st.session_state.mission_last_feedback = {
        "is_correct": is_correct,
        "target_state": result["target_state"],
        "explanation": result["explanation"],
    }

    # Keep the generic game-state layer (game_state.py) in sync, same as
    # the Decision Game and Ordering Game do.
    game_state.record_player_action(
        action=f"final state {final_state}",
        is_correct=is_correct,
        detail=concept,
    )

    is_last = step_idx == len(spec["steps"]) - 1
    if is_last:
        st.session_state.xp += XP_MISSION_COMPLETE_BONUS
        st.session_state.mission_xp_earned += XP_MISSION_COMPLETE_BONUS
        st.session_state.mission_complete = True
        st.session_state.mission_weak_concept = _compute_weak_concept()
        game_state.complete_game()
    else:
        st.session_state.mission_index += 1
        simulation_game.reset_runtime()  # next step (if any) starts with a fresh live state


def _compute_weak_concept():
    """Weakest concept = most wrong answers (ties broken by worst accuracy).
    Returns None if the player made no mistakes — i.e. full mastery."""
    stats = st.session_state.mission_concept_stats
    candidates = [(concept, s) for concept, s in stats.items() if s["wrong"] > 0]
    if not candidates:
        return None

    def accuracy(s):
        total = s["correct"] + s["wrong"]
        return s["correct"] / total if total else 0.0

    candidates.sort(key=lambda item: (-item[1]["wrong"], accuracy(item[1])))
    return candidates[0][0]


def _mastery_percent():
    spec = st.session_state.mission_spec
    total = len(spec["steps"])
    return round(100 * st.session_state.mission_score / total) if total else 0


# ----------------------------------------------------------------------
# Rendering — Generic Mission Engine
# ----------------------------------------------------------------------
def _render_mission_briefing(meta: dict):
    with st.expander("📋 Mission Briefing", expanded=False):
        if meta.get("objective"):
            st.markdown(f"**Objective:** {meta['objective']}")
        if meta.get("instructions"):
            st.markdown(f"**How to play:** {meta['instructions']}")
        if meta.get("concepts"):
            st.markdown("**Concepts covered:** " + ", ".join(meta["concepts"]))
        if meta.get("metrics"):
            st.markdown("**Tracked metrics:** " + ", ".join(meta["metrics"]))


def _render_generic_mission(topic: str, difficulty: str):
    _init_mission_state()

    if st.session_state.mission_error:
        st.error(st.session_state.mission_error)
        if st.button("🔁 Try Again", use_container_width=True):
            _start_mission(topic, difficulty)
            st.rerun()
        return

    if not st.session_state.mission_spec:
        with st.spinner("Designing your mission..."):
            pass  # spinner is shown inside _start_mission itself
        _start_mission(topic, difficulty)
        st.rerun()
        return

    spec = st.session_state.mission_spec
    st.markdown(f"### 🎮 {spec['game_title']}")
    if spec.get("mission"):
        st.caption(spec["mission"])

    _render_mission_briefing(spec)

    total_steps = len(spec["steps"])
    completed_steps = total_steps if st.session_state.mission_complete else min(
        st.session_state.mission_index, total_steps
    )
    active_step = (
        "Complete"
        if st.session_state.mission_complete
        else f"{min(completed_steps + 1, total_steps) if total_steps else 0}/{total_steps}"
    )
    overall_progress = round(100 * completed_steps / total_steps) if total_steps else 0

    hud1, hud2, hud3, hud4 = st.columns(4)
    with hud1:
        st.metric("🏆 Score", f"{st.session_state.mission_score}/{total_steps}")
    with hud2:
        st.metric("📍 Current Step", active_step)
    with hud3:
        st.metric("✅ Completed Steps", f"{completed_steps}/{total_steps}")
    with hud4:
        st.metric("📊 Overall Progress", f"{overall_progress}%")

    st.caption(f"🧠 Mastery: {_mastery_percent()}%" if st.session_state.mission_complete else "🧠 Mastery: —")

    st.divider()

    if st.session_state.mission_complete:
        _render_mission_results(topic, difficulty)
    else:
        _render_mission_step()


# Maps a mission's game_type to the module that implements it. Every
# module in this map exposes the same render_hud()/render_feedback()
# signatures (see decision_game.py / ordering_game.py / simulation_game.py),
# which is what lets _render_mission_step() stay generic below instead of
# growing a new branch of HUD code per interaction type.
_GAME_TYPE_ENGINES = {
    "ordering": ordering_game,
    "simulation": simulation_game,
}


def _render_mission_step():
    spec = st.session_state.mission_spec
    idx = st.session_state.mission_index
    total_steps = len(spec["steps"])
    step = spec["steps"][idx]
    hint_used = st.session_state.mission_hints_used.get(idx, False)
    state = game_state.get_game_state()
    game_type = spec.get("game_type", "decision")
    engine = _GAME_TYPE_ENGINES.get(game_type, decision_game)

    st.subheader("🎯 Challenge")
    st.caption(f"Step {idx + 1} of {total_steps} — concept: {step.get('concept', '')}")

    engine.render_hud(
        objective=spec.get("objective", ""),
        score=state["score"] if state else 0,
        attempts=state["attempts"] if state else 0,
        current_step=state["current_step"] if state else idx,
        total_steps=total_steps,
    )

    # Immediate feedback for the action the player just took, shown above
    # the next scenario — same pattern cpu_arena.py already uses.
    fb = st.session_state.mission_last_feedback
    if fb is not None:
        engine.render_feedback(fb["is_correct"])

    st.divider()

    hint_col, _ = st.columns([1, 3])
    with hint_col:
        if st.button("💡 Hint", key=f"mission_hint_{idx}", disabled=hint_used):
            st.session_state.mission_hints_used[idx] = True
            st.rerun()
    if hint_used:
        st.info(f"💡 {step.get('hint') or 'Think carefully about what this concept requires.'}")

    if game_type == "simulation":
        final_state = simulation_game.render_step(step, key_prefix=f"mission_action_{idx}")
        if final_state is not None:
            _record_simulation(idx, final_state)
            st.rerun()
    elif game_type == "ordering":
        chosen_order = ordering_game.render_step(step, key_prefix=f"mission_action_{idx}")
        if chosen_order is not None:
            _record_ordering(idx, chosen_order)
            st.rerun()
    else:
        chosen_index = decision_game.render_step(step, key_prefix=f"mission_action_{idx}")
        if chosen_index is not None:
            _record_decision(idx, chosen_index)
            st.rerun()


def _render_mission_results(topic: str, difficulty: str):
    spec = st.session_state.mission_spec
    total = len(spec["steps"])
    score = st.session_state.mission_score
    accuracy = round(100 * score / total) if total else 0

    st.subheader("🏆 MISSION COMPLETE")
    st.markdown(f"🏆 **Score:** {score}/{total}")
    st.markdown(f"⭐ **XP Earned:** +{st.session_state.mission_xp_earned}")
    st.markdown(f"🎯 **Accuracy:** {accuracy}% ({score}/{total} correct)")

    st.markdown("#### Concept Breakdown")
    rows = []
    for concept, s in st.session_state.mission_concept_stats.items():
        attempted = s["correct"] + s["wrong"]
        acc = round(100 * s["correct"] / attempted) if attempted else 0
        rows.append({"Concept": concept, "Correct": s["correct"], "Wrong": s["wrong"], "Accuracy": f"{acc}%"})
    if rows:
        st.table(rows)

    weak_concept = st.session_state.mission_weak_concept

    st.markdown("#### 🧠 AI Feedback")
    if weak_concept:
        st.warning(f"⚠️ Weak Concept: **{weak_concept}**")
    else:
        st.success("🏆 Full mastery — no weak concepts detected this run!")

    if st.session_state.mission_focus_error:
        st.error(st.session_state.mission_focus_error)

    if st.session_state.mission_focus_explanation:
        data = st.session_state.mission_focus_explanation
        st.markdown("##### 🔄 Focused Explanation")
        st.markdown("**Analogy**")
        st.write(data.get("game_analogy", ""))
        st.markdown("**Technical Mapping**")
        st.markdown(data.get("technical_mapping", ""))
        st.markdown("**Technical Explanation**")
        st.write(data.get("technical_explanation", ""))
        st.markdown("**Key Takeaways**")
        st.markdown(data.get("key_takeaways", ""))

    b1, b2, b3 = st.columns(3)
    with b1:
        if weak_concept and st.button("🔄 Explain Again", use_container_width=True,
                                       help=f"Focus: {weak_concept}"):
            st.session_state.mission_focus_error = None
            try:
                with st.spinner("Preparing a focused explanation..."):
                    result = generate_explanation(
                        prompts.CONCEPT_FOCUS_SYSTEM_PROMPT,
                        prompts.build_concept_focus_prompt(topic, weak_concept, difficulty),
                    )
                st.session_state.mission_focus_explanation = result
            except LLMConfigError as e:
                st.session_state.mission_focus_error = str(e)
            except Exception as e:  # noqa: BLE001
                st.session_state.mission_focus_error = f"Couldn't generate a focused explanation: {e}"
            st.rerun()
    with b2:
        if st.button("⚔️ Retry Mission", use_container_width=True):
            _start_mission(topic, difficulty, previous_title=spec.get("game_title"))
            st.rerun()
    with b3:
        if st.button("🔁 Change Topic", use_container_width=True):
            _change_topic()
            st.rerun()


# ----------------------------------------------------------------------
# Rendering — top-level Game Arena entry point (called from app.py)
# ----------------------------------------------------------------------
def render_game_arena(topics: list, difficulties: list):
    """
    Renders the whole Game Arena tab: topic/difficulty selection, then
    routes to either the specialized CPU Arena or the generic AI-designed
    mission engine.
    """
    _init_router_state()

    st.subheader("🎮 Game Arena")
    st.caption(
        "Turn any topic into a short interactive mission. The AI designs the mission once; "
        "every score, timing, and correctness check while you play is computed deterministically."
    )

    locked = not st.session_state.game_arena_choosing

    topic_options = ["CPU Scheduling"] + [t for t in topics if t != "CPU Scheduling"] + ["Custom topic..."]
    col1, col2 = st.columns(2)
    with col1:
        topic_choice = st.selectbox("🧠 Mission Topic", topic_options, disabled=locked, key="game_arena_topic_choice")
        if topic_choice == "Custom topic...":
            topic = st.text_input(
                "Enter your own topic",
                placeholder="e.g. Binary Search, Stack, SQL Joins, Newton's Laws...",
                disabled=locked,
                key="game_arena_custom_topic",
            )
        else:
            topic = topic_choice
    with col2:
        difficulty = st.selectbox("📈 Difficulty", difficulties, disabled=locked, key="game_arena_difficulty_choice")

    if not locked:
        if is_specialized(topic):
            st.info("✅ A specialized mission module exists for this topic — launching CPU Arena.")
            if st.button("🎮 Enter Mission", type="primary", use_container_width=True, disabled=not topic):
                st.session_state.game_arena_locked_topic = topic
                st.session_state.game_arena_locked_difficulty = difficulty
                st.session_state.game_arena_choosing = False
                st.rerun()
        else:
            if st.button("🎮 Start Mission", type="primary", use_container_width=True,
                         disabled=not topic or not topic.strip()):
                st.session_state.game_arena_locked_topic = topic.strip()
                st.session_state.game_arena_locked_difficulty = difficulty
                st.session_state.game_arena_choosing = False
                _clear_mission()
                st.rerun()
        return

    st.divider()
    locked_topic = st.session_state.game_arena_locked_topic
    locked_difficulty = st.session_state.game_arena_locked_difficulty

    if is_specialized(locked_topic):
        module = SPECIALIZED_TOPICS[_normalize(locked_topic)]
        _render_mission_briefing(module.get_universal_mission_meta())
        module.render_arena()
        st.divider()
        if st.button("🔁 Change Topic", use_container_width=True):
            _change_topic()
            st.rerun()
    else:
        _render_generic_mission(locked_topic, locked_difficulty)
