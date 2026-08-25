# 🎮 Explain It Like I Play

A small Streamlit app that explains CS/engineering concepts using accurate
mechanics and analogies from a video game you already know — Minecraft,
Valorant, Pokémon, FIFA, Roblox, or Among Us — without sacrificing
technical correctness.

## Project structure

```
explain-it-like-i-play/
├── app.py              # Streamlit UI + session state + orchestration
├── prompts.py           # System + user prompt construction for each mode
├── llm_client.py         # Gemini API wrapper (reads key from env, parses JSON response)
├── config.py             # Static lists: games, topic suggestions, difficulty levels
├── requirements.txt
├── .env.example           # Copy to .env and add your API key
└── .gitignore
```

## Setup

1. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API key**

   ```bash
   cp .env.example .env
   ```

   Open `.env` and paste your Gemini API key:

   ```
   GEMINI_API_KEY=your-api-key
   ```

   The key is loaded from the environment at runtime — it is never
   hardcoded anywhere in the source.

## Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How it works

1. Pick a **game**, a **topic** (from the suggested list or your own), and
   a **difficulty**, then click **Explain**.
2. The app sends a structured prompt to the LLM asking for strict JSON
   with four fields: `game_analogy`, `technical_mapping`,
   `technical_explanation`, and `key_takeaways`. These render as the four
   labeled sections in the UI.
3. Follow-up actions reuse the locked-in game/topic/difficulty and only
   call the LLM when clicked (Streamlit session state prevents
   unnecessary regeneration on unrelated reruns):
   - **🔄 Explain Again** — same topic/game/difficulty, a fresh analogy.
   - **🧒 Explain Simpler** — simplifies wording, keeps the same meaning.
   - **📖 Explain Technically** — adds technical depth and precision.
   - **🎮 Change Game** — clears the result and unlocks the game selector.

## Notes on accuracy

The system prompt explicitly instructs the model to:
- never invent or misstate game mechanics,
- keep the game analogy clearly separate from the real technical
  explanation, and
- never sacrifice technical correctness for the sake of the analogy.

This is prompt-level guidance, not a guarantee — treat generated
explanations as a study aid, not a substitute for a textbook.

## Notes on scope

This is intentionally an MVP: no authentication, no database, no
payments. State lives only in the Streamlit session (resets on page
refresh), which keeps the app simple and easy to extend later.
