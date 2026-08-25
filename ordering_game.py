"""
ordering_game.py
The Ordering Game — a second reusable interaction type built on top of
the generic game-state framework (game_state.py), alongside the
existing Decision Game (decision_game.py).

Pattern:
    Mission Scenario
      -> Show current game state (objective, score, attempts, progress)
      -> Show a set of items the player must arrange
      -> Player picks the items in the order they believe is correct
      -> Python checks the resulting order         <- deterministic, no LLM
      -> Game state changes
      -> Show immediate feedback
      -> Continue until mission completion

An "ordering step" is any dict shaped like the ones mission_engine.py
gets back from llm_client.generate_ordering_mission():
    {
        "items": [str, ...],          # 3-6 items, presented scrambled
        "correct_order": [int, ...],  # indices into items, the correct
                                       # sequence, set by the LLM at
                                       # mission-DESIGN time only — never
                                       # at play time
        "concept": str (optional),
        "hint": str (optional),
        "explanation": str (optional),
    }

CRITICAL: this module never asks the LLM to judge a player's arrangement.
The LLM's only involvement was supplying `correct_order` once, up front,
when the mission was designed (see llm_client.generate_ordering_mission /
prompts.ORDERING_MISSION_SYSTEM_PROMPT). Everything below is plain Python.

This module deliberately mirrors decision_game.py's shape (same HUD,
same feedback style) so the two interaction types feel consistent to the
player, without either module depending on the other.
"""

import streamlit as st


def evaluate_order(step: dict, chosen_order: list) -> dict:
    """
    Pure, deterministic evaluation of a player's chosen item order
    against the step's pre-recorded correct_order. No LLM call, no
    randomness.
    """
    correct_order = step.get("correct_order", [])
    is_correct = list(chosen_order) == list(correct_order)
    return {
        "is_correct": is_correct,
        "correct_order": correct_order,
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
    Immediate, concise feedback after a submitted order — does NOT
    reveal the correct order, only whether the player's arrangement was
    right.
    """
    if is_correct:
        st.success("✅ Correct order!")
    else:
        st.error("❌ Not quite the right order yet.")


def render_step(step: dict, key_prefix: str):
    """
    Renders the scrambled items ("🔢 Arrange In Order") and lets the
    player build their sequence by picking items, in order, via a
    multiselect (selection order = the player's arrangement). Returns
    the list of chosen item INDICES the moment the player submits a
    complete order, or None otherwise.
    """
    items = step.get("items", [])

    st.markdown("##### 🔢 Arrange In Order")
    st.caption("Pick the items below, in order, to build your sequence — then submit.")

    # Use indices as the widget values so duplicate display labels remain
    # separate, selectable items. ``format_func`` keeps the player-facing UI
    # exactly as before: it shows only the original item text.
    item_indices = list(range(len(items)))
    chosen_order = st.multiselect(
        "Your order:",
        options=item_indices,
        format_func=lambda item_index: items[item_index],
        key=f"{key_prefix}_order",
        label_visibility="collapsed",
    )

    st.write(" → ".join(items[i] for i in chosen_order) if chosen_order else "*(nothing selected yet)*")

    submit_disabled = len(chosen_order) != len(items)
    if st.button(
        "✅ Submit Order",
        key=f"{key_prefix}_submit",
        disabled=submit_disabled,
        use_container_width=True,
    ):
        return chosen_order
    return None
