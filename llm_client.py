"""
llm_client.py
Thin wrapper around Google's Gemini API using the `google-genai` SDK
(the current, actively-maintained client — successor to the older
`google-generativeai` package).

The API key is NEVER hardcoded — it is read from the environment
(loaded from a local .env file via python-dotenv). If the key is
missing, callers get a clear RuntimeError instead of a confusing
SDK stack trace.
"""

import os
import json
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import DEFAULT_MODEL, DEFAULT_MAX_TOKENS, QUIZ_MAX_TOKENS, DEFAULT_TEMPERATURE

load_dotenv()  # reads .env if present; no-op in production envs with real env vars set


class LLMConfigError(RuntimeError):
    """Raised when the API key / client cannot be set up."""


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMConfigError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return genai.Client(api_key=api_key)


def _check_not_truncated(response, config_var_name: str = "DEFAULT_MAX_TOKENS") -> None:
    """
    Raises a clear, actionable error if the model's response was cut off
    before finishing (e.g. hit max_output_tokens) rather than letting a
    truncated-JSON parse failure surface as a confusing stack trace.
    """
    try:
        finish_reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return  # SDK shape changed or unavailable — don't block on this check

    reason_name = getattr(finish_reason, "name", str(finish_reason))
    if reason_name in ("MAX_TOKENS", "2"):  # "2" covers older SDKs returning raw enum ints
        raise ValueError(
            "The model's response was cut off before it finished (hit the output token "
            f"limit). Try increasing {config_var_name} in config.py, or simplify the "
            "request, and try again."
        )


def _extract_json(text: str) -> dict:
    """
    The model is instructed to return raw JSON, but LLMs sometimes wrap
    output in ```json fences or add stray text. This defensively pulls
    out the first valid JSON object it can find.
    """
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from model response:\n{text}")


def generate_explanation(system_prompt: str, user_prompt: str) -> dict:
    """
    Calls the LLM and returns a dict with keys:
    game_analogy, technical_mapping, technical_explanation, key_takeaways
    """
    client = _get_client()
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            response_mime_type="application/json",
        ),
    )

    _check_not_truncated(response)
    raw_text = response.text
    data = _extract_json(raw_text)

    # Guarantee all expected keys exist so the UI never KeyErrors.
    for key in ("game_analogy", "technical_mapping", "technical_explanation", "key_takeaways"):
        data.setdefault(key, "")

    return data


def generate_quiz(system_prompt: str, user_prompt: str) -> list:
    """
    Calls the LLM to generate a 5-question MCQ quiz and returns a list of
    dicts, each shaped like:
        {
            "question": str,
            "options": [str, str, str, str],
            "correct_index": int,   # 0-3
            "explanation": str,
        }
    Raises ValueError if the model's response doesn't validate into that
    shape, so the caller (app.py) can surface a clear error instead of
    silently rendering a broken quiz.
    """
    client = _get_client()
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=QUIZ_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            response_mime_type="application/json",
        ),
    )

    _check_not_truncated(response, config_var_name="QUIZ_MAX_TOKENS")
    raw_text = response.text
    data = _extract_json(raw_text)

    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Quiz response did not contain a non-empty 'questions' list.")

    validated = []
    for i, q in enumerate(questions[:5]):  # defensively cap at 5
        question_text = q.get("question")
        options = q.get("options")
        correct_index = q.get("correct_index")
        explanation = q.get("explanation", "")

        if not question_text or not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"Question {i + 1} is malformed (needs text + exactly 4 options).")
        if not isinstance(correct_index, int) or not (0 <= correct_index <= 3):
            raise ValueError(f"Question {i + 1} has an invalid correct_index: {correct_index!r}.")

        validated.append(
            {
                "question": question_text,
                "options": options,
                "correct_index": correct_index,
                "explanation": explanation,
            }
        )

    if len(validated) < 5:
        raise ValueError(f"Expected 5 quiz questions, got {len(validated)}.")

    return validated