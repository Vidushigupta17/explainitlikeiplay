"""
simulation_game.py
The Simulation Game — a third reusable interaction type, alongside the
Decision Game (decision_game.py) and Ordering Game (ordering_game.py).

Pattern:
    Mission Scenario
      -> Show current game state (objective, score, attempts, progress)
      -> Show the CURRENT SIMULATION STATE (a generic ordered list of values)
      -> Show 2-5 available operations (e.g. PUSH / POP / PEEK)
      -> Player performs an operation
      -> Python applies it deterministically and updates the state  <- no LLM
      -> UI re-renders the updated state immediately
      -> Player clicks Finish whenever they believe they've reached the goal
      -> Python checks the final state against the target state     <- no LLM

A "simulation step" is any dict shaped like the ones mission_engine.py
gets back from llm_client.generate_simulation_mission():
    {
        "initial_state": [value, ...],        # starting state, LLM-designed
        "available_operations": [
            {"name": str, "effect": str, "requires_value": bool},
            ...
        ],
        "target_state": [value, ...],         # the state the player must reach
        "max_operations": int (optional),
        "concept": str (optional),
        "hint": str (optional),
        "explanation": str (optional),
    }

--------------------------------------------------------------------------
WHERE EACH OF THE REQUESTED CONCEPTS LIVES
--------------------------------------------------------------------------
- initial_state         -> step["initial_state"]           (LLM-designed, fixed)
- available_operations   -> step["available_operations"]     (LLM-designed, fixed)
- current_state          -> runtime, tracked in st.session_state by this
                            module; starts as a copy of initial_state and
                            changes as the player acts
- operation              -> whichever entry of available_operations the
                            player clicks
- state_update            -> apply_operation() below: the ONE deterministic,
                            generic function that turns (current_state,
                            operation) into a new current_state, for EVERY
                            topic
- success_condition       -> step's human-readable description, checked
                            against the machine-checkable step["target_state"]
                            by evaluate_result()
- concepts                -> step["concept"] (+ the mission-level
                            "concepts" list)

CRITICAL: exactly like decision_game.py and ordering_game.py, the LLM never
executes an operation or judges a live player action. It designs the
scenario/spec ONCE, up front (initial_state, available_operations,
target_state, success_condition, concepts). apply_operation() and
evaluate_result() below are the ONLY code that ever touches live state, and
both are plain, generic, topic-agnostic Python — the same two functions run
for a Stack mission, a Queue mission, or any other linear-structure topic.
Adding a new simulation topic never requires new Python — only new DATA
(a different initial_state / available_operations / target_state).
"""

import streamlit as st

RUNTIME_KEY = "simulation_runtime"

# The fixed, generic vocabulary of state-changing "effects" every operation
# must resolve to. This IS the reusable engine — supporting a new
# simulation topic means mapping its operations onto these effects, never
# writing new per-topic code.
#   push_back / pop_back / peek_back   -> act on the END of the list
#                                          (e.g. "top of stack", "back of queue")
#   push_front / pop_front / peek_front -> act on the START of the list
#                                          (e.g. "front of queue")
#   clear                                -> empties the whole state
VALID_EFFECTS = {
    "push_back",
    "push_front",
    "pop_back",
    "pop_front",
    "peek_back",
    "peek_front",
    "clear",
}


def apply_operation(current_state: list, op: dict, value=None) -> dict:
    """
    THE generic state-update function. Deterministically applies one
    operation to the current state and returns the new state plus
    anything the operation "observed" (what a POP/PEEK saw), so the UI can
    show feedback like "Popped 30" or "Top is 20".

    No LLM call, no randomness, no topic-specific branching — every
    simulation topic (Stack, Queue, ...) is just a different set of
    available_operations mapped onto this same handful of effects.
    """
    effect = op.get("effect")
    state = list(current_state)
    observed = None

    if effect == "push_back":
        state.append(value)
    elif effect == "push_front":
        state.insert(0, value)
    elif effect == "pop_back":
        if state:
            observed = state.pop()
    elif effect == "pop_front":
        if state:
            observed = state.pop(0)
    elif effect == "peek_back":
        if state:
            observed = state[-1]
    elif effect == "peek_front":
        if state:
            observed = state[0]
    elif effect == "clear":
        state = []
    # An unrecognized effect is a no-op rather than a crash — mission
    # validation (llm_client) already rejects any effect outside
    # VALID_EFFECTS before a step ever reaches this function.

    return {"new_state": state, "observed": observed}


def evaluate_result(step: dict, final_state: list) -> dict:
    """
    Pure, deterministic evaluation of the player's final state against the
    step's pre-recorded target_state. No LLM call, no randomness.
    """
    target_state = step.get("target_state", [])
    is_correct = list(final_state) == list(target_state)
    return {
        "is_correct": is_correct,
        "target_state": target_state,
        "explanation": step.get("explanation", ""),
    }


def init_runtime(step: dict):
    """Starts (or restarts) the live runtime state for one simulation
    step: a copy of initial_state, plus an operation counter."""
    st.session_state[RUNTIME_KEY] = {
        "current_state": list(step.get("initial_state", [])),
        "operations_used": 0,
        "last_observed": None,
        "last_op_name": None,
    }


def get_runtime():
    return st.session_state.get(RUNTIME_KEY)


def reset_runtime():
    """Clears the live runtime state (used when leaving/restarting a
    simulation mission). Safe to call even if no simulation is active."""
    st.session_state.pop(RUNTIME_KEY, None)


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
    """Immediate, concise feedback after the player finishes — does NOT
    reveal the target state, only whether they reached it."""
    if is_correct:
        st.success("✅ Target state reached!")
    else:
        st.error("❌ Not quite the target state.")


def _render_state(state: list):
    st.markdown("##### 📦 Current State")
    if not state:
        st.code("(empty)")
    else:
        # Drawn end-first (e.g. "top of stack" shown at the top), matching
        # the reference example: Current Stack: [30] [20] [10].
        st.code("\n".join(f"[{v}]" for v in reversed(state)))


def _mission_values(step: dict) -> list:
    """Returns distinct state values in a stable order for typed PUSH input."""
    values = []
    for value in [*step.get("initial_state", []), *step.get("target_state", [])]:
        if not any(type(value) is type(existing) and value == existing for existing in values):
            values.append(value)
    return values


def _render_value_input(step: dict, key_prefix: str):
    """Renders a value control that preserves the schema's value types.

    Purely numeric missions retain the free-form ``number_input`` used by
    existing simulations. Missions containing strings use a deterministic
    selector of the values declared by the mission; these are the same values
    the reachability validator considers, so every validated target value can
    be supplied without coercing it to a number or string.
    """
    values = _mission_values(step)
    if all(type(value) is int for value in values):
        return st.number_input(
            "Value (for operations that need one):",
            step=1,
            key=f"{key_prefix}_value",
        )

    return st.selectbox(
        "Value (for operations that need one):",
        options=values,
        format_func=str,
        key=f"{key_prefix}_value",
    )


def render_step(step: dict, key_prefix: str):
    """
    Renders the current simulation state, the available operations (with
    a shared value input for any operation whose requires_value is true),
    and a Finish button. Every operation click immediately updates the
    runtime state and reruns, so the state shown always reflects the
    player's latest action.

    Returns the FINAL current_state list the moment the player clicks
    "Finish", or None while they're still acting.
    """
    runtime = get_runtime()
    if runtime is None:
        init_runtime(step)
        runtime = get_runtime()

    _render_state(runtime["current_state"])

    if runtime["last_op_name"] is not None:
        if runtime["last_observed"] is not None:
            st.caption(f"Last operation: **{runtime['last_op_name']}** → observed `{runtime['last_observed']}`")
        else:
            st.caption(f"Last operation: **{runtime['last_op_name']}**")

    max_ops = step.get("max_operations")
    ops_caption = f"Operations used: {runtime['operations_used']}"
    if max_ops:
        ops_caption += f" / {max_ops}"
    st.caption(ops_caption)

    st.markdown("##### ⚙️ Available Operations")
    operations = step.get("available_operations", [])

    value = None
    if any(op.get("requires_value") for op in operations):
        value = _render_value_input(step, key_prefix)

    ops_disabled = bool(max_ops) and runtime["operations_used"] >= max_ops
    cols = st.columns(len(operations)) if operations else []
    for op_index, (col, op) in enumerate(zip(cols, operations)):
        with col:
            if st.button(
                op.get("name", "OP"),
                # The display name can legitimately repeat in generated
                # content, so use its stable position for the widget key.
                key=f"{key_prefix}_operation_{op_index}",
                use_container_width=True,
                disabled=ops_disabled,
            ):
                result = apply_operation(
                    runtime["current_state"], op, value=value if op.get("requires_value") else None
                )
                runtime["current_state"] = result["new_state"]
                runtime["operations_used"] += 1
                runtime["last_observed"] = result["observed"]
                runtime["last_op_name"] = op.get("name", "OP")
                st.rerun()

    st.divider()
    if st.button("🏁 Finish", key=f"{key_prefix}_finish", use_container_width=True, type="primary"):
        return list(runtime["current_state"])
    return None
