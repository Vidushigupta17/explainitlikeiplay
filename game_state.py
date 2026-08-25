"""
game_state.py
Generic, game-type-agnostic interactive game state.

This is the first layer of a broader "Game Engine" architecture that sits
underneath the existing step-based Game Arena missions:

    Mission
      -> Game State            (this module)
      -> Player Action
      -> Game Engine evaluates the action   (deterministic Python)
      -> Game State updates
      -> Next action
      -> Mission complete

This shared state is actively consumed by the generic Game Arena mission
flow in mission_engine.py. Decision, Ordering, and Simulation missions all
record each submitted result here, and their HUDs read its score, attempts,
and completed-step count. Game-specific mechanics remain in their own
modules (decision_game.py, ordering_game.py, and simulation_game.py), while
this module provides the common state and action history they share.

CRITICAL (same rule as cpu_arena.py and the existing mission step flow):
every function in this file is plain deterministic Python. The LLM is
NEVER asked to evaluate a player action, compute score, or decide
correctness here.

State lives in st.session_state under a single key, same pattern as the
rest of the app — no database needed.
"""

import streamlit as st

GAME_STATE_KEY = "generic_game_state"


def _blank_state(mission_id: str, topic: str, game_type: str, objective: str) -> dict:
    return {
        "mission_id": mission_id,
        "topic": topic,
        "game_type": game_type,
        "objective": objective,
        "current_step": 0,
        "score": 0,
        "attempts": 0,
        "actions_taken": [],   # ordered list of {step, action, is_correct, detail}
        "correct_actions": 0,
        "mistakes": 0,
        "game_completed": False,
    }


def init_game_state(mission_id: str, topic: str, game_type: str, objective: str) -> dict:
    """Starts a fresh generic game state for a new mission (overwrites any
    previous one — only one generic game is active at a time)."""
    state = _blank_state(mission_id, topic, game_type, objective)
    st.session_state[GAME_STATE_KEY] = state
    return state


def get_game_state():
    """Returns the current generic game state dict, or None if no generic
    game has been started (or it's been reset)."""
    return st.session_state.get(GAME_STATE_KEY)


def has_active_game() -> bool:
    state = get_game_state()
    return state is not None and not state["game_completed"]


def record_player_action(action: str, is_correct: bool, detail: str = "") -> dict:
    """
    Deterministically records one player action against the generic game
    state: logs it, updates attempts/correct_actions/mistakes/score, and
    advances current_step. Score is the number of correct answers, matching
    ``mission_score`` and every user-facing mission HUD. This is the ONLY
    place that mutates those
    fields — no LLM call involved, same as cpu_arena.make_decision() and
    mission_engine._record_decision().

    No-ops (and returns None) if there's no active game state, or the
    game is already marked complete.
    """
    state = get_game_state()
    if state is None or state["game_completed"]:
        return None

    state["attempts"] += 1
    state["actions_taken"].append(
        {
            "step": state["current_step"],
            "action": action,
            "is_correct": is_correct,
            "detail": detail,
        }
    )

    if is_correct:
        state["correct_actions"] += 1
        state["score"] += 1
    else:
        state["mistakes"] += 1

    state["current_step"] += 1
    return state


def complete_game() -> dict | None:
    """Marks the generic game as complete. Idempotent."""
    state = get_game_state()
    if state is not None:
        state["game_completed"] = True
    return state


def reset_game_state():
    """Clears the generic game state entirely (used when leaving/changing
    the mission that owns it)."""
    st.session_state.pop(GAME_STATE_KEY, None)
