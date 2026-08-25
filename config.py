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
]

DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]

# Model + generation defaults (can be overridden via .env with GEMINI_MODEL)
# NOTE: llm_client.py calls the Gemini API (google-genai SDK), so this must be
# a valid Gemini model id — not an Anthropic model name.
DEFAULT_MODEL = "gemini-2.5-flash"
# Explanations have four JSON fields.  Some topics (especially at the
# Intermediate/Advanced levels) can exceed 2,048 tokens before the closing
# brace is emitted, which leaves the app with invalid, truncated JSON.
# 4,096 gives the model enough room to complete the response.
DEFAULT_MAX_TOKENS = 4096
# Quiz responses hold 5 full MCQ questions (question + 4 options + a
# correct-answer explanation each), which is significantly more JSON than
# a single explanation — give it more headroom so it doesn't get cut off
# mid-response.
QUIZ_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.8
