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
from collections import deque

from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import DEFAULT_MODEL, DEFAULT_MAX_TOKENS, QUIZ_MAX_TOKENS, MISSION_MAX_TOKENS, DEFAULT_TEMPERATURE
from simulation_game import VALID_EFFECTS, apply_operation

load_dotenv()  # reads .env if present; no-op in production envs with real env vars set


class LLMConfigError(RuntimeError):
    """Raised when the API key / client cannot be set up."""


class LLMServiceError(LLMConfigError):
    """Raised when the model service cannot currently accept a request."""


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMConfigError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return genai.Client(api_key=api_key)


def _generate_content(client: genai.Client, **kwargs):
    """Calls Gemini while keeping transient provider payloads out of the UI."""
    try:
        return client.models.generate_content(**kwargs)
    except Exception as error:  # noqa: BLE001 — provider SDK exceptions vary by version
        message = str(error)
        if "RESOURCE_EXHAUSTED" in message.upper() or "429" in message:
            raise LLMServiceError(
                "Gemini's request quota is currently exhausted. Wait for the quota to reset or use an API "
                "key/project with available quota, then try again."
            ) from error
        if ("NOT_FOUND" in message.upper() or "404" in message) and "model" in message.lower():
            raise LLMServiceError(
                f"The configured Gemini model '{kwargs.get('model')}' is unavailable. Update GEMINI_MODEL in "
                ".env or use the current default model, then try again."
            ) from error
        raise


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

    response = _generate_content(
        client,
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
            "hint": str,
            "explanation": str,
        }
    Raises ValueError if the model's response doesn't validate into that
    shape, so the caller (app.py) can surface a clear error instead of
    silently rendering a broken quiz.
    """
    client = _get_client()
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    response = _generate_content(
        client,
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
        hint = q.get("hint", "") or ""
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
                "hint": hint,
                "explanation": explanation,
            }
        )

    if len(validated) < 5:
        raise ValueError(f"Expected 5 quiz questions, got {len(validated)}.")

    return validated


def generate_mission(system_prompt: str, user_prompt: str) -> dict:
    """
    Calls the LLM to design a Game Arena mission (see prompts.py's
    MISSION_SYSTEM_PROMPT) and returns a validated dict shaped like:
        {
            "game_title": str, "mission": str, "objective": str,
            "game_type": str, "instructions": str, "concepts": [str, ...],
            "difficulty": str, "success_condition": str, "metrics": [str, ...],
            "steps": [
                {"id": str, "scenario": str, "prompt": str, "options": [str, ...],
                 "correct_index": int, "concept": str, "hint": str, "explanation": str},
                ...
            ],
        }

    IMPORTANT: this function only asks the LLM to design mission *content*
    (including which option is correct). It never asks the LLM to judge a
    live player action, compute scores, or compute timing/performance
    metrics — that is all done deterministically in mission_engine.py using
    the "correct_index" values validated here.

    Raises ValueError with a clear message if the response doesn't validate,
    so the caller can show a graceful fallback instead of crashing.
    """
    client = _get_client()
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    response = _generate_content(
        client,
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=MISSION_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            response_mime_type="application/json",
        ),
    )

    _check_not_truncated(response, config_var_name="MISSION_MAX_TOKENS")
    raw_text = response.text
    data = _extract_json(raw_text)
    if not isinstance(data, dict):
        raise ValueError("Mission response must be a JSON object.")

    required_top_level = ("game_title", "mission", "objective", "game_type", "instructions", "concepts", "steps")
    for key in required_top_level:
        if key not in data or data.get(key) in (None, "", []):
            raise ValueError(f"Mission response is missing or has an empty '{key}' field.")

    # This is the fallback *Decision* schema. Ordering and Simulation have
    # their own validated generators above, so never let an incompatible AI
    # response reach their renderers with decision-shaped steps.
    if str(data.get("game_type", "")).strip().lower() != "decision":
        raise ValueError(f"Fallback mission must have game_type 'decision', got {data.get('game_type')!r}.")

    concepts = data.get("concepts")
    if not isinstance(concepts, list) or not all(isinstance(c, str) and c.strip() for c in concepts):
        raise ValueError("Mission 'concepts' must be a non-empty list of non-empty strings.")

    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        raise ValueError("Mission response did not contain a non-empty 'steps' list.")

    validated_steps = []
    for i, s in enumerate(steps[:5]):  # defensively cap at 5
        prompt_text = s.get("prompt")
        options = s.get("options")
        correct_index = s.get("correct_index")
        concept = s.get("concept", "")
        hint = s.get("hint", "") or ""
        explanation = s.get("explanation", "")
        scenario = s.get("scenario", "") or ""

        if not prompt_text or not isinstance(options, list) or not (2 <= len(options) <= 4):
            raise ValueError(f"Step {i + 1} is malformed (needs a prompt + 2-4 options).")
        if not isinstance(correct_index, int) or not (0 <= correct_index < len(options)):
            raise ValueError(f"Step {i + 1} has an invalid correct_index: {correct_index!r}.")
        if not concept or concept not in concepts:
            # Be lenient: fall back to the first declared concept rather than failing the
            # whole mission over a mislabeled step, since this only affects analytics.
            concept = concepts[0]

        validated_steps.append(
            {
                "id": s.get("id") or f"s{i + 1}",
                "scenario": scenario,
                "prompt": prompt_text,
                "options": options,
                "correct_index": correct_index,
                "concept": concept,
                "hint": hint,
                "explanation": explanation,
            }
        )

    if len(validated_steps) < 3:
        raise ValueError(f"Expected at least 3 mission steps, got {len(validated_steps)}.")

    return {
        "game_title": data.get("game_title", "Untitled Mission"),
        "mission": data.get("mission", ""),
        "objective": data.get("objective", ""),
        "game_type": "decision",
        "instructions": data.get("instructions", ""),
        "concepts": concepts,
        "difficulty": data.get("difficulty", ""),
        "success_condition": data.get("success_condition", ""),
        "metrics": data.get("metrics", []) if isinstance(data.get("metrics"), list) else [],
        "steps": validated_steps,
    }


# ----------------------------------------------------------------------
# Small Decision Game mission schema — one scenario, one set of actions,
# one correct action. Deliberately much smaller than generate_mission()'s
# 5-step schema (see prompts.DECISION_MISSION_JSON_SHAPE). The LLM only
# ever designs this content once, up front; it is converted below into
# the SAME universal mission-spec shape generate_mission() returns
# (game_title/objective/game_type/instructions/concepts/difficulty/
# success_condition/steps), so mission_engine.py and decision_game.py
# don't need to know or care which schema produced it.
# ----------------------------------------------------------------------
def _validate_and_convert_decision_spec(data: dict, topic: str, difficulty: str) -> dict:
    """
    Pure validation + conversion, no network call — kept separate from
    generate_decision_mission() so it's trivial to unit-test and reuse.
    Raises ValueError (with a specific reason) on anything invalid:
    missing/empty required fields, a game_type other than "decision",
    an available_actions list that isn't 2-5 non-empty strings, or a
    correct_action that isn't exactly one of the offered actions.

    Python decides validity here — the LLM's response is just data.
    """
    required = ("title", "objective", "game_type", "scenario", "available_actions", "correct_action", "concepts", "hint")
    for key in required:
        if key not in data or data.get(key) in (None, "", []):
            raise ValueError(f"Decision mission response is missing or has an empty '{key}' field.")

    if str(data.get("game_type", "")).strip().lower() != "decision":
        raise ValueError(f"Expected game_type 'decision', got {data.get('game_type')!r}.")

    actions = data.get("available_actions")
    if not isinstance(actions, list) or not (2 <= len(actions) <= 5):
        raise ValueError("Decision mission 'available_actions' must be a list of 2-5 actions.")
    if any(not isinstance(a, str) or not a.strip() for a in actions):
        raise ValueError("Decision mission 'available_actions' must all be non-empty strings.")

    correct_action = data.get("correct_action")
    if not isinstance(correct_action, str) or correct_action not in actions:
        raise ValueError(
            f"Decision mission 'correct_action' ({correct_action!r}) is not one of the offered "
            "'available_actions'."
        )

    concepts = data.get("concepts")
    if not isinstance(concepts, list) or not all(isinstance(c, str) and c.strip() for c in concepts):
        raise ValueError("Decision mission 'concepts' must be a non-empty list of non-empty strings.")

    hint = data.get("hint")
    if not isinstance(hint, str) or not hint.strip():
        raise ValueError("Decision mission 'hint' must be a non-empty string.")

    correct_index = actions.index(correct_action)
    success_condition = data.get("success_condition", "") or ""

    return {
        "game_title": data.get("title", "Untitled Mission"),
        "mission": "",
        "objective": data.get("objective", ""),
        "game_type": "decision",
        "instructions": (
            f"Read the scenario and choose the action that best achieves: {success_condition}"
            if success_condition
            else "Read the scenario and choose the best action."
        ),
        "concepts": concepts,
        "difficulty": data.get("difficulty", difficulty),
        "success_condition": success_condition,
        "metrics": [],
        "steps": [
            {
                "id": "s1",
                "scenario": data.get("scenario", ""),
                "prompt": "Choose your action:",
                "options": actions,
                "correct_index": correct_index,
                "concept": concepts[0],
                "hint": hint.strip(),
                "explanation": success_condition,
            }
        ],
    }


def generate_decision_mission(system_prompt: str, user_prompt: str, topic: str, difficulty: str) -> dict:
    """
    Calls the LLM to design a small, single-scenario Decision Game mission
    (see prompts.DECISION_MISSION_SYSTEM_PROMPT) and returns it already
    converted into the universal mission-spec shape.

    IMPORTANT: the LLM only ever designs mission CONTENT here (the
    scenario, the actions, and which action is correct). It never judges
    a live player action — decision_game.evaluate_action() does that,
    deterministically, using the correct_index this function resolves
    from correct_action below.

    Raises ValueError if the response doesn't validate, so the caller
    (mission_engine._start_mission) can fall back to the existing
    multi-step mission generator instead of crashing.
    """
    client = _get_client()
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    response = _generate_content(
        client,
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

    return _validate_and_convert_decision_spec(data, topic, difficulty)


# ----------------------------------------------------------------------
# Small Ordering Game mission schema — one "arrange these items in the
# correct order" challenge. A second reusable interaction type alongside
# the Decision Game above. Just like the Decision Game, this is converted
# below into the SAME universal mission-spec shape generate_mission()
# returns, so mission_engine.py doesn't need to know or care which schema
# produced it.
# ----------------------------------------------------------------------
def _validate_and_convert_ordering_spec(data: dict, topic: str, difficulty: str) -> dict:
    """
    Pure validation + conversion, no network call — kept separate from
    generate_ordering_mission() so it's trivial to unit-test and reuse.
    Raises ValueError (with a specific reason) on anything invalid:
    missing/empty required fields, a game_type other than "ordering", an
    items list that isn't 3-6 non-empty strings, or a correct_order that
    isn't a permutation of the items' indices.

    Python decides validity here — the LLM's response is just data.
    """
    required = ("title", "objective", "game_type", "items", "correct_order", "concepts", "hint")
    for key in required:
        if key not in data or data.get(key) in (None, "", []):
            raise ValueError(f"Ordering mission response is missing or has an empty '{key}' field.")

    if str(data.get("game_type", "")).strip().lower() != "ordering":
        raise ValueError(f"Expected game_type 'ordering', got {data.get('game_type')!r}.")

    items = data.get("items")
    if not isinstance(items, list) or not (3 <= len(items) <= 6):
        raise ValueError("Ordering mission 'items' must be a list of 3-6 items.")
    if any(not isinstance(i, str) or not i.strip() for i in items):
        raise ValueError("Ordering mission 'items' must all be non-empty strings.")

    correct_order = data.get("correct_order")
    if (
        not isinstance(correct_order, list)
        or not all(isinstance(i, int) for i in correct_order)
        or sorted(correct_order) != list(range(len(items)))
    ):
        raise ValueError(
            f"Ordering mission 'correct_order' must be a permutation of item indices 0..{len(items) - 1}."
        )

    concepts = data.get("concepts")
    if not isinstance(concepts, list) or not all(isinstance(c, str) and c.strip() for c in concepts):
        raise ValueError("Ordering mission 'concepts' must be a non-empty list of non-empty strings.")

    hint = data.get("hint")
    if not isinstance(hint, str) or not hint.strip():
        raise ValueError("Ordering mission 'hint' must be a non-empty string.")

    success_condition = data.get("success_condition", "") or ""

    return {
        "game_title": data.get("title", "Untitled Mission"),
        "mission": "",
        "objective": data.get("objective", ""),
        "game_type": "ordering",
        "instructions": data.get("instructions") or "Arrange the items into the correct order, then submit.",
        "concepts": concepts,
        "difficulty": data.get("difficulty", difficulty),
        "success_condition": success_condition,
        "metrics": [],
        "steps": [
            {
                "id": "s1",
                "items": items,
                "correct_order": correct_order,
                "concept": concepts[0],
                "hint": hint.strip(),
                "explanation": success_condition,
            }
        ],
    }


def generate_ordering_mission(system_prompt: str, user_prompt: str, topic: str, difficulty: str) -> dict:
    """
    Calls the LLM to design a small, single-challenge Ordering Game
    mission (see prompts.ORDERING_MISSION_SYSTEM_PROMPT) and returns it
    already converted into the universal mission-spec shape.

    IMPORTANT: the LLM only ever designs mission CONTENT here (the
    items and which sequence is correct). It never judges a live
    player's arrangement — ordering_game.evaluate_order() does that,
    deterministically, using the correct_order validated below.

    Raises ValueError if the response doesn't validate, so the caller
    (mission_engine._start_mission) can fall back to another mission
    generator instead of crashing.
    """
    client = _get_client()
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    response = _generate_content(
        client,
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

    return _validate_and_convert_ordering_spec(data, topic, difficulty)


# ----------------------------------------------------------------------
# Small Simulation Game mission schema — one "perform operations on a
# live state until it matches a target" challenge. A third reusable
# interaction type alongside the Decision Game and Ordering Game above.
# Converted below into the SAME universal mission-spec shape
# generate_mission() returns, so mission_engine.py doesn't need to know
# or care which schema produced it. simulation_game.apply_operation() —
# imported at the top of this file, never duplicated — is the single
# source of truth for what each "effect" does, so validation and
# gameplay can never disagree about what an operation means.
# ----------------------------------------------------------------------
def _simulation_target_reachable(initial_state: list, target_state: list, operations: list, max_ops: int) -> bool:
    """
    Bounded breadth-first search using the SAME apply_operation() the
    live game uses, so a mission is only accepted if a real player could
    actually reach target_state with the given operations. Bounded by
    max_ops moves and a visited-state cap so this can never hang.

    Values tried for any operation that "requires_value" are limited to
    the values that actually appear in initial_state/target_state (the
    only values that could ever matter for reaching an exact target),
    keeping the search finite and fast.
    """
    start = tuple(initial_state)
    goal = tuple(target_state)
    if start == goal:
        return False  # nothing to do isn't a valid mission either

    candidate_values = list({*initial_state, *target_state}) or [None]
    visited = {start}
    queue = deque([(start, 0)])
    max_visited = 5000

    while queue:
        state, depth = queue.popleft()
        if depth >= max_ops:
            continue
        for op in operations:
            values_to_try = candidate_values if op.get("requires_value") else [None]
            for value in values_to_try:
                new_state = tuple(apply_operation(list(state), op, value=value)["new_state"])
                if new_state == goal:
                    return True
                if new_state not in visited and len(visited) < max_visited:
                    visited.add(new_state)
                    queue.append((new_state, depth + 1))
    return False


def _is_supported_simulation_value(value) -> bool:
    """Returns whether a state value has a lossless playable UI path.

    Integers keep the existing numeric input. Non-empty strings are selected
    as their original string values by the simulation UI. ``bool`` is
    deliberately excluded even though it is an ``int`` subclass: Streamlit's
    numeric input cannot produce a boolean value.
    """
    return type(value) is int or (isinstance(value, str) and bool(value.strip()))


def _validate_and_convert_simulation_spec(data: dict, topic: str, difficulty: str) -> dict:
    """
    Pure validation + conversion, no network call — kept separate from
    generate_simulation_mission() so it's trivial to unit-test and reuse.
    Raises ValueError (with a specific reason) on anything invalid:
    missing/empty required fields, a game_type other than "simulation",
    an operation whose "effect" isn't in VALID_EFFECTS, or a target_state
    that isn't actually reachable from initial_state using the given
    operations within max_operations moves.

    Python decides validity here — the LLM's response is just data.
    """
    required_text_fields = ("title", "objective", "game_type", "success_condition", "hint")
    for key in required_text_fields:
        if key not in data or data.get(key) in (None, ""):
            raise ValueError(f"Simulation mission response is missing or has an empty '{key}' field.")

    if str(data.get("game_type", "")).strip().lower() != "simulation":
        raise ValueError(f"Expected game_type 'simulation', got {data.get('game_type')!r}.")

    hint = data.get("hint")
    if not isinstance(hint, str) or not hint.strip():
        raise ValueError("Simulation mission 'hint' must be a non-empty string.")

    initial_state = data.get("initial_state")
    if not isinstance(initial_state, list) or len(initial_state) > 8:
        raise ValueError("Simulation mission 'initial_state' must be a list of at most 8 values.")
    if any(not _is_supported_simulation_value(v) for v in initial_state):
        raise ValueError(
            "Simulation mission 'initial_state' values must be integers or non-empty strings."
        )

    target_state = data.get("target_state")
    if not isinstance(target_state, list) or not target_state or len(target_state) > 8:
        raise ValueError("Simulation mission 'target_state' must be a non-empty list of at most 8 values.")
    if any(not _is_supported_simulation_value(v) for v in target_state):
        raise ValueError(
            "Simulation mission 'target_state' values must be integers or non-empty strings."
        )
    if target_state == initial_state:
        raise ValueError("Simulation mission 'target_state' must differ from 'initial_state'.")

    operations = data.get("available_operations")
    if not isinstance(operations, list) or not (2 <= len(operations) <= 5):
        raise ValueError("Simulation mission 'available_operations' must be a list of 2-5 operations.")

    validated_ops = []
    for i, op in enumerate(operations):
        name = op.get("name") if isinstance(op, dict) else None
        effect = op.get("effect") if isinstance(op, dict) else None
        if not name or not isinstance(name, str):
            raise ValueError(f"Operation {i + 1} is missing a valid 'name'.")
        if effect not in VALID_EFFECTS:
            raise ValueError(f"Operation {i + 1} has an invalid 'effect' {effect!r}.")
        validated_ops.append(
            {
                "name": name,
                "effect": effect,
                "requires_value": bool(op.get("requires_value", False)),
            }
        )

    concepts = data.get("concepts")
    if not isinstance(concepts, list) or not all(isinstance(c, str) and c.strip() for c in concepts):
        raise ValueError("Simulation mission 'concepts' must be a non-empty list of non-empty strings.")

    max_operations = data.get("max_operations")
    if not isinstance(max_operations, int) or not (1 <= max_operations <= 12):
        max_operations = 8  # sane default if the model omits/mis-types it

    if not _simulation_target_reachable(initial_state, target_state, validated_ops, max_operations):
        raise ValueError(
            "Simulation mission 'target_state' is not reachable from 'initial_state' using the given "
            "'available_operations' within 'max_operations' moves."
        )

    success_condition = data.get("success_condition", "") or ""

    return {
        "game_title": data.get("title", "Untitled Mission"),
        "mission": "",
        "objective": data.get("objective", ""),
        "game_type": "simulation",
        "instructions": data.get("instructions") or "Perform operations to reach the target state, then click Finish.",
        "concepts": concepts,
        "difficulty": data.get("difficulty", difficulty),
        "success_condition": success_condition,
        "metrics": [],
        "steps": [
            {
                "id": "s1",
                "initial_state": initial_state,
                "available_operations": validated_ops,
                "target_state": target_state,
                "max_operations": max_operations,
                "concept": concepts[0],
                "hint": hint.strip(),
                "explanation": success_condition,
            }
        ],
    }


def generate_simulation_mission(system_prompt: str, user_prompt: str, topic: str, difficulty: str) -> dict:
    """
    Calls the LLM to design a small, single-challenge Simulation Game
    mission (see prompts.SIMULATION_MISSION_SYSTEM_PROMPT) and returns it
    already converted into the universal mission-spec shape.

    IMPORTANT: the LLM only ever designs mission CONTENT here (the
    initial state, the operations, and the target state). It never
    executes an operation or judges a live player action —
    simulation_game.apply_operation() and .evaluate_result() do that,
    deterministically, using the spec validated below.

    Raises ValueError if the response doesn't validate, so the caller
    (mission_engine._start_mission) can fall back to another mission
    generator instead of crashing.
    """
    client = _get_client()
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    response = _generate_content(
        client,
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

    return _validate_and_convert_simulation_spec(data, topic, difficulty)
