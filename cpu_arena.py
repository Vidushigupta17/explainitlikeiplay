"""
cpu_arena.py
🎮 CPU ARENA — a small interactive game that teaches CPU scheduling
(FCFS, SJF, Priority, Round Robin) by having the player act as the
scheduler: at each decision point, pick which process runs next.

CRITICAL DESIGN RULE (per product requirement):
All scheduling math — ready-queue membership, which process is
"correct" under the selected algorithm, waiting time, turnaround
time, and score — is computed here in plain deterministic Python.
The LLM is never asked to calculate any of this; it is only ever
used (elsewhere, e.g. quiz/explanation features) for natural-language
explanation, never for arithmetic or scheduling decisions.

State lives entirely in st.session_state (prefixed "arena_"), same
pattern as the rest of the app — no database needed.
"""

import streamlit as st

# ----------------------------------------------------------------------
# Fixed process set for the MVP. Numbers were chosen deliberately so
# that FCFS, SJF, and Priority each produce a DIFFERENT execution
# order — useful both for teaching and for testing (see README).
#   priority: lower number = higher priority (runs first), standard
#   convention used by most OS textbooks.
# ----------------------------------------------------------------------
PROCESSES = [
    {"pid": "P1", "emoji": "⚔️", "label": "Enemy Battle", "arrival": 0, "burst": 5, "priority": 3},
    {"pid": "P2", "emoji": "⛏️", "label": "Mining", "arrival": 1, "burst": 3, "priority": 1},
    {"pid": "P3", "emoji": "🏠", "label": "Building", "arrival": 2, "burst": 8, "priority": 4},
    {"pid": "P4", "emoji": "🌾", "label": "Farming", "arrival": 3, "burst": 2, "priority": 2},
]

ALGORITHMS = ["FCFS", "SJF", "Priority", "Round Robin"]
DEFAULT_QUANTUM = 2


def get_universal_mission_meta() -> dict:
    """
    Describes CPU Arena using the same universal mission shape that the
    generic Game Arena engine (mission_engine.py) uses for every other
    topic — purely for a consistent "Mission Briefing" display. This is
    additive documentation only; it does not change any of the game's
    internal deterministic logic below.
    """
    return {
        "topic": "CPU Scheduling",
        "game_title": "CPU Arena",
        "objective": "Learn how CPU scheduling algorithms decide execution order by acting as the scheduler yourself.",
        "instructions": "Pick an algorithm (and quantum, for Round Robin), then choose which process runs next at each decision point.",
        "concepts": ["FCFS", "SJF", "Priority Scheduling", "Round Robin", "Waiting Time", "Turnaround Time"],
        "metrics": ["Waiting Time", "Turnaround Time", "Score"],
    }

ARENA_KEYS = [
    "arena_started",
    "arena_algorithm",
    "arena_quantum",
    "arena_time",
    "arena_procs",
    "arena_rr_queue",
    "arena_completed_order",
    "arena_score",
    "arena_log",
    "arena_last_feedback",
    "arena_game_over",
]


def init_arena_state():
    st.session_state.setdefault("arena_started", False)
    st.session_state.setdefault("arena_algorithm", None)
    st.session_state.setdefault("arena_quantum", DEFAULT_QUANTUM)


def _proc_label(pid: str) -> str:
    p = st.session_state.arena_procs[pid]
    return f"{p['emoji']} {p['pid']} — {p['label']}"


# ----------------------------------------------------------------------
# Game setup / teardown
# ----------------------------------------------------------------------
def start_arena(algorithm: str, quantum: int = DEFAULT_QUANTUM):
    st.session_state.arena_started = True
    st.session_state.arena_algorithm = algorithm
    st.session_state.arena_quantum = quantum
    st.session_state.arena_time = 0
    st.session_state.arena_procs = {
        p["pid"]: {**p, "remaining": p["burst"], "completed": False,
                   "completion_time": None, "waiting_time": None, "turnaround_time": None}
        for p in PROCESSES
    }
    st.session_state.arena_rr_queue = []
    st.session_state.arena_completed_order = []
    st.session_state.arena_score = 0
    st.session_state.arena_log = []
    st.session_state.arena_last_feedback = None
    st.session_state.arena_game_over = False
    if algorithm == "Round Robin":
        _rr_update_queue()


def reset_arena():
    for key in ARENA_KEYS:
        st.session_state.pop(key, None)
    init_arena_state()


# ----------------------------------------------------------------------
# Deterministic scheduling logic (NO LLM involved anywhere below)
# ----------------------------------------------------------------------
def _rr_update_queue():
    """Append newly-arrived processes (not yet queued, not completed) to
    the Round Robin FIFO queue, ordered by (arrival_time, pid)."""
    procs = st.session_state.arena_procs
    queue = st.session_state.arena_rr_queue
    t = st.session_state.arena_time
    already = set(queue) | {pid for pid, p in procs.items() if p["completed"]}
    newcomers = [pid for pid, p in procs.items() if p["arrival"] <= t and pid not in already]
    newcomers.sort(key=lambda pid: (procs[pid]["arrival"], pid))
    queue.extend(newcomers)


def get_ready_pids():
    """Non-RR algorithms: pids that have arrived and are not yet completed."""
    procs = st.session_state.arena_procs
    t = st.session_state.arena_time
    ready = [pid for pid, p in procs.items() if not p["completed"] and p["arrival"] <= t]
    return sorted(ready, key=lambda pid: (procs[pid]["arrival"], pid))


def advance_idle_if_needed():
    """If the CPU would be idle (nothing ready) but the game isn't over,
    fast-forward the clock to the next arrival — no decision needed
    while idle."""
    procs = st.session_state.arena_procs
    if all(p["completed"] for p in procs.values()):
        st.session_state.arena_game_over = True
        return

    algorithm = st.session_state.arena_algorithm
    if algorithm == "Round Robin":
        _rr_update_queue()
        if not st.session_state.arena_rr_queue:
            future = [p["arrival"] for p in procs.values()
                      if not p["completed"] and p["arrival"] > st.session_state.arena_time]
            if future:
                st.session_state.arena_time = min(future)
                _rr_update_queue()
    else:
        if not get_ready_pids():
            future = [p["arrival"] for p in procs.values() if not p["completed"]]
            if future:
                st.session_state.arena_time = min(future)


def determine_correct_pid():
    """The scheduling-theory-correct choice at this exact decision point,
    per the selected algorithm. Pure arithmetic — no LLM call."""
    algorithm = st.session_state.arena_algorithm
    procs = st.session_state.arena_procs

    if algorithm == "Round Robin":
        queue = st.session_state.arena_rr_queue
        return queue[0] if queue else None

    ready = get_ready_pids()
    if not ready:
        return None
    if algorithm == "FCFS":
        return min(ready, key=lambda pid: (procs[pid]["arrival"], pid))
    if algorithm == "SJF":
        return min(ready, key=lambda pid: (procs[pid]["remaining"], procs[pid]["arrival"], pid))
    if algorithm == "Priority":
        return min(ready, key=lambda pid: (procs[pid]["priority"], procs[pid]["arrival"], pid))
    return None


def make_decision(chosen_pid: str):
    """Executes the player's choice, scores it against the deterministically
    correct choice, updates waiting/turnaround time on completion, and logs
    the outcome. All math here — none of it is delegated to the LLM."""
    algorithm = st.session_state.arena_algorithm
    procs = st.session_state.arena_procs
    correct_pid = determine_correct_pid()
    is_correct = chosen_pid == correct_pid

    if is_correct:
        st.session_state.arena_score += 10
    else:
        st.session_state.arena_score = max(0, st.session_state.arena_score - 5)

    p = procs[chosen_pid]

    if algorithm == "Round Robin":
        quantum = st.session_state.arena_quantum
        run_time = min(quantum, p["remaining"])
        st.session_state.arena_time += run_time
        p["remaining"] -= run_time

        # IMPORTANT: update arrivals-during-this-quantum BEFORE removing the
        # just-run process from the queue, so it isn't mistaken for a fresh
        # arrival and re-added twice.
        _rr_update_queue()

        queue = st.session_state.arena_rr_queue
        if queue and queue[0] == chosen_pid:
            queue.pop(0)
        elif chosen_pid in queue:
            queue.remove(chosen_pid)  # safety net; shouldn't normally happen

        if p["remaining"] <= 0:
            p["completed"] = True
            p["completion_time"] = st.session_state.arena_time
            p["turnaround_time"] = p["completion_time"] - p["arrival"]
            p["waiting_time"] = p["turnaround_time"] - p["burst"]
            st.session_state.arena_completed_order.append(chosen_pid)
        else:
            queue.append(chosen_pid)
    else:
        # Non-preemptive: FCFS / SJF / Priority run the chosen process to completion.
        run_time = p["remaining"]
        st.session_state.arena_time += run_time
        p["remaining"] = 0
        p["completed"] = True
        p["completion_time"] = st.session_state.arena_time
        p["turnaround_time"] = p["completion_time"] - p["arrival"]
        p["waiting_time"] = p["turnaround_time"] - p["burst"]
        st.session_state.arena_completed_order.append(chosen_pid)

    st.session_state.arena_last_feedback = {
        "chosen": chosen_pid,
        "correct": correct_pid,
        "is_correct": is_correct,
    }
    verdict = "✅ Correct!" if is_correct else f"❌ Not quite — {algorithm} would run {correct_pid} next."
    st.session_state.arena_log.append(f"t={st.session_state.arena_time}: chose {chosen_pid} — {verdict}")

    advance_idle_if_needed()


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def render_arena():
    init_arena_state()

    st.subheader("🎮 CPU ARENA")
    st.caption("You are the scheduler. Pick which process runs next — the game checks your "
               "decision against the real algorithm and tracks waiting time, turnaround time, and score.")

    if not st.session_state.arena_started:
        _render_setup()
        return

    advance_idle_if_needed()
    st.markdown("━━━━━━━━━━━━━━")

    top1, top2, top3 = st.columns(3)
    with top1:
        st.metric("🧮 Algorithm", st.session_state.arena_algorithm)
    with top2:
        st.metric("⏱️ Clock", st.session_state.arena_time)
    with top3:
        st.metric("🏆 Score", st.session_state.arena_score)

    if st.session_state.arena_game_over:
        _render_results()
        return

    _render_in_progress()


def _render_setup():
    st.markdown("#### Choose your scheduling algorithm")
    algorithm = st.radio("Algorithm", ALGORITHMS, horizontal=True, label_visibility="collapsed")

    quantum = DEFAULT_QUANTUM
    if algorithm == "Round Robin":
        quantum = st.number_input("⏳ Time Quantum", min_value=1, max_value=10, value=DEFAULT_QUANTUM, step=1)

    st.markdown("#### The processes waiting to run")
    st.table(
        [
            {
                "Process": f"{p['emoji']} {p['pid']} — {p['label']}",
                "Arrival Time": p["arrival"],
                "Burst Time": p["burst"],
                "Priority": p["priority"],
            }
            for p in PROCESSES
        ]
    )
    st.caption("Lower priority number = higher priority (runs first). Priority only matters for Priority Scheduling.")

    if st.button("▶️ Start CPU Arena", type="primary", use_container_width=True):
        start_arena(algorithm, quantum)
        st.rerun()


def _render_in_progress():
    procs = st.session_state.arena_procs
    algorithm = st.session_state.arena_algorithm

    # ---- Ready Queue ----
    st.markdown("#### 🧾 Ready Queue")
    if algorithm == "Round Robin":
        ready_pids = list(st.session_state.arena_rr_queue)
    else:
        ready_pids = get_ready_pids()

    if ready_pids:
        st.write(" → ".join(_proc_label(pid) + f" (remaining: {procs[pid]['remaining']})" for pid in ready_pids))
    else:
        st.write("_(empty)_")

    # ---- Last decision feedback ----
    fb = st.session_state.arena_last_feedback
    if fb:
        if fb["is_correct"]:
            st.success(f"✅ {fb['chosen']} was the correct pick under {algorithm}.")
        else:
            st.error(f"❌ You picked {fb['chosen']}, but {algorithm} would have run {fb['correct']} next.")

    # ---- Decision prompt ----
    st.markdown("#### Choose the next process to execute")
    if ready_pids:
        cols = st.columns(len(ready_pids))
        for col, pid in zip(cols, ready_pids):
            p = procs[pid]
            with col:
                btn_label = f"{p['emoji']} {p['pid']}"
                help_text = f"{p['label']} — arrival {p['arrival']}, burst {p['remaining']} left"
                if algorithm == "Priority":
                    help_text += f", priority {p['priority']}"
                if st.button(btn_label, key=f"arena_choice_{pid}_{st.session_state.arena_time}",
                             help=help_text, use_container_width=True):
                    make_decision(pid)
                    st.rerun()

    # ---- Completed Processes ----
    st.markdown("#### ✅ Completed Processes")
    completed = st.session_state.arena_completed_order
    if completed:
        st.table(
            [
                {
                    "Process": pid,
                    "Completion Time": procs[pid]["completion_time"],
                    "Waiting Time": procs[pid]["waiting_time"],
                    "Turnaround Time": procs[pid]["turnaround_time"],
                }
                for pid in completed
            ]
        )
    else:
        st.write("_(none yet)_")

    st.divider()
    if st.button("🔁 Restart Arena", use_container_width=True):
        reset_arena()
        st.rerun()


def _render_results():
    procs = st.session_state.arena_procs
    completed = st.session_state.arena_completed_order
    algorithm = st.session_state.arena_algorithm

    st.success("🏁 All processes completed!")

    st.markdown("#### 📊 Final Results")
    rows = [
        {
            "Process": pid,
            "Arrival": procs[pid]["arrival"],
            "Burst": procs[pid]["burst"],
            "Completion": procs[pid]["completion_time"],
            "Waiting Time": procs[pid]["waiting_time"],
            "Turnaround Time": procs[pid]["turnaround_time"],
        }
        for pid in completed
    ]
    st.table(rows)

    avg_wait = sum(procs[pid]["waiting_time"] for pid in completed) / len(completed)
    avg_turnaround = sum(procs[pid]["turnaround_time"] for pid in completed) / len(completed)

    m1, m2, m3 = st.columns(3)
    m1.metric("🏆 Final Score", st.session_state.arena_score)
    m2.metric("⏳ Avg Waiting Time", f"{avg_wait:.2f}")
    m3.metric("🔁 Avg Turnaround Time", f"{avg_turnaround:.2f}")

    with st.expander("🧾 Decision log"):
        for line in st.session_state.arena_log:
            st.write(line)

    st.caption(f"Order completed under {algorithm}: {' → '.join(completed)}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 Replay Same Algorithm", use_container_width=True):
            start_arena(algorithm, st.session_state.arena_quantum)
            st.rerun()
    with c2:
        if st.button("🧮 Try a Different Algorithm", use_container_width=True):
            reset_arena()
            st.rerun()