"""
decision_game.py
The Decision Game — the first concrete, reusable game mechanic built on
top of the generic game-state framework (game_state.py).

Pattern:
    Mission Scenario
      -> Show current game state (objective, score, attempts, progress)
      -> Show 2-5 possible actions
      -> Player chooses an action
      -> Python evaluates the action        <- deterministic, no LLM
      -> Game state changes
      -> Show immediate feedback
      -> Continue until mission completion

A "decision step" is any dict shaped like the ones mission_engine.py
already gets back from llm_client.generate_mission():
    {
        "scenario": str (optional),   # current-state flavor text
        "prompt": str,                # what the player is being asked
        "options": [str, ...],        # 2-5 available actions
        "correct_index": int,         # the deterministic correct action,
                                       # set by the LLM at mission-DESIGN
                                       # time only — never at play time
        "explanation": str (optional),
    }

CRITICAL: this module never asks the LLM to judge a player's action. The
LLM's only involvement was supplying `correct_index` once, up front, when
the mission was designed (see llm_client.generate_mission /
prompts.MISSION_SYSTEM_PROMPT). Everything below is plain Python.

This module is intentionally generic about WHERE its steps come from —
it doesn't know or care whether the mission was AI-generated. CPU Arena
is a separate, already-deterministic game and is not wired to this file.
"""

import streamlit as st


def evaluate_action(step: dict, chosen_index: int) -> dict:
    """
    Pure, deterministic evaluation of a player's chosen action against the
    step's pre-recorded correct_index. No LLM call, no randomness.
    """
    correct_index = step.get("correct_index")
    is_correct = chosen_index == correct_index
    return {
        "is_correct": is_correct,
        "correct_index": correct_index,
        "explanation": step.get("explanation", ""),
    }


def render_hud(objective: str, score: int, attempts: int, current_step: int, total_steps: int):
    """Renders score plus distinct current/completed/progress indicators."""
    if objective:
        st.markdown(f"🎯 **Current Objective:** {objective}")

    completed_steps = min(max(current_step, 0), total_steps)
    active_step = min(completed_steps + 1, total_steps) if total_steps else 0
    overall_progress = round(100 * completed_steps / total_steps) if total_steps else 0

    h1, h2, h3 = st.columns(3)
    with h1:
        st.metric("⭐ Current Score", score)
    with h2:
        st.metric("🔥 Attempts", attempts)
    with h3:
        st.metric("📍 Current Step", f"{active_step}/{total_steps}")

    p1, p2 = st.columns(2)
    with p1:
        st.metric("✅ Completed Steps", f"{completed_steps}/{total_steps}")
    with p2:
        st.metric("📊 Overall Progress", f"{overall_progress}%")


def render_feedback(is_correct: bool):
    """
    Immediate, concise feedback after an action — does NOT reveal the
    correct action, only whether the player's choice was right.
    """
    if is_correct:
        st.success("✅ Good choice!")
    else:
        st.error("❌ Not quite.")


def render_step(step: dict, key_prefix: str):
    """
    Renders the scenario + 2-5 clickable action buttons for one decision
    step ("🎮 Available Actions"). Returns the index of the action the
    player just clicked this run, or None if nothing was clicked yet.
    Clicking an action IS the submission — no separate confirm step.
    """
    if step.get("scenario"):
        st.write(step["scenario"])
    if step.get("prompt"):
        st.markdown(f"**{step['prompt']}**")

    st.markdown("##### 🎮 Available Actions")
    options = step.get("options", [])
    cols = st.columns(len(options))
    for i, (col, label) in enumerate(zip(cols, options)):
        with col:
            if st.button(label, key=f"{key_prefix}_{i}", use_container_width=True):
                return i
    return None
