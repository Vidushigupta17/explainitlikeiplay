"""
config.py
Static configuration data for the app: supported games, sample topics,
and difficulty levels. Keeping this separate from app.py keeps the UI
file clean and makes it trivial to add a new game or topic later.
"""

GAMES = [
    "Minecraft",
    "Valorant",
    "Pokémon",
    "FIFA",
    "Roblox",
    "Among Us",
]

# Suggested topics shown as quick-pick chips. The user can still type
# any topic of their own — this is just to help them get started.
SUGGESTED_TOPICS = [
    "Recursion",
    "Hash Tables",
    "TCP/IP Networking",
    "Multithreading",
    "Binary Search",
    "Client-Server Architecture",
    "Caching",
    "State Machines",
    "Graph Traversal (BFS/DFS)",
    "Load Balancing",
    "Object-Oriented Programming",
    "Database Indexing",
    "CPU Scheduling",
]

DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]

# Model + generation defaults (can be overridden via .env with GEMINI_MODEL)
# NOTE: llm_client.py calls the Gemini API (google-genai SDK), so this must be
# a valid Gemini model id — not an Anthropic model name.
# gemini-2.5-flash is unavailable to new Gemini API accounts. Keep the
# default aligned with Gemini's supported successor; .env may still override
# it through GEMINI_MODEL when a project needs a different available model.
DEFAULT_MODEL = "gemini-3.6-flash"
# Explanations have 4 free-text fields — 2048 tokens is comfortable.
DEFAULT_MAX_TOKENS = 2048
# Quiz responses hold 5 full MCQ questions (question + 4 options + a
# correct-answer explanation each), which is significantly more JSON than
# a single explanation — give it more headroom so it doesn't get cut off
# mid-response.
QUIZ_MAX_TOKENS = 4096
# Mission generation returns a title/objective/instructions + 5 decision
# steps (each with up to 4 options, a hint, and an explanation) — similar
# order of magnitude to the quiz, so it gets the same headroom.
MISSION_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.8

# ----------------------------------------------------------------------
# Gamification — XP awards and level thresholds for the Challenge mode.
# Everything here is deterministic (pure math on total XP), so no
# database or auth is needed; state lives in Streamlit session_state.
# ----------------------------------------------------------------------
XP_CORRECT_NO_HINT = 20   # correct answer, no hint used
XP_CORRECT_WITH_HINT = 10  # correct answer, hint used
XP_MISSION_COMPLETE_BONUS = 50  # awarded once for finishing all 5 questions

# (xp_threshold, level_number, level_name) — sorted ascending by threshold.
LEVELS = [
    (0, 1, "Rookie"),
    (100, 2, "Learner"),
    (250, 3, "Explorer"),
    (450, 4, "Strategist"),
    (700, 5, "Master"),
]


def get_level_info(xp: int) -> dict:
    """
    Deterministically maps total XP to a level. Returns a dict with the
    current level number/name plus the XP bounds of the current level
    and the threshold for the next one (None if already at max level) —
    handy for rendering a "Mastery Progress" bar toward the next level.
    """
    current = LEVELS[0]
    next_threshold = None
    for i, entry in enumerate(LEVELS):
        threshold, _, _ = entry
        if xp >= threshold:
            current = entry
            next_threshold = LEVELS[i + 1][0] if i + 1 < len(LEVELS) else None
        else:
            break

    threshold, level_num, level_name = current
    return {
        "level_num": level_num,
        "level_name": level_name,
        "current_threshold": threshold,
        "next_threshold": next_threshold,
    }
