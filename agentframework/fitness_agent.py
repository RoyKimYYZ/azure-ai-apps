import logging
import mimetypes
import os
import sys
import json
import inspect
import hashlib
from pathlib import Path
from typing import Any

from agent_framework import ChatAgent, ChatMessage, DataContent, TextContent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from openai import RateLimitError
from prompt_toolkit import PromptSession

from config import get_config
from fitness_memory import (
    DatabaseContextProvider,
    PhotoSubmissionStructuredOutput,
    TextTurnStructuredOutput,
    extract_idempotency_key,
    get_fitness_repository,
)
from run_utils import format_usage, run_with_retry, run_with_stream


logger = logging.getLogger(__name__)
_prompt_session = PromptSession()


async def _prompt_text(message: str) -> str:
    try:
        return (await _prompt_session.prompt_async(message)).strip()
    except Exception:
        return input(message).strip()


async def _resolve_maybe_awaitable(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _status(message: str) -> None:
    logger.info(message)
    print(f"⏳ {message}")


def _coerce_payload(result: Any) -> tuple[PhotoSubmissionStructuredOutput, dict[str, Any]]:
    parsed = getattr(result, "parsed", None)
    if isinstance(parsed, PhotoSubmissionStructuredOutput):
        payload = parsed
        raw = payload.model_dump()
        return payload, raw
    if isinstance(parsed, dict):
        payload = PhotoSubmissionStructuredOutput.model_validate(parsed)
        return payload, parsed
    text = getattr(result, "text", "")
    if isinstance(text, str):
        loaded = json.loads(text)
        payload = PhotoSubmissionStructuredOutput.model_validate(loaded)
        return payload, loaded
    raise ValueError("Unable to parse structured photo submission output.")


def _print_help() -> None:
    print("\nCommands:")
    print("  /photo <path>   Analyze meal photo and persist memory")
    print("  /summary        Print durable read model summary from database")
    print("  /help           Show commands")
    print("  /exit           Exit chat\n")
    print("Note: Run /photo inside the fitness> prompt. Do not run /photo in bash.\n")


def _extract_photo_path(command_text: str) -> str:
    return command_text.removeprefix("/photo").strip()


def _coerce_text_turn_payload(result: Any) -> tuple[TextTurnStructuredOutput, dict[str, Any]]:
    parsed = getattr(result, "parsed", None)
    if isinstance(parsed, TextTurnStructuredOutput):
        payload = parsed
        raw = payload.model_dump()
        return payload, raw
    if isinstance(parsed, dict):
        payload = TextTurnStructuredOutput.model_validate(parsed)
        return payload, parsed
    text = getattr(result, "text", "")
    if isinstance(text, str):
        loaded = json.loads(text)
        payload = TextTurnStructuredOutput.model_validate(loaded)
        return payload, loaded
    raise ValueError("Unable to parse structured text turn output.")


async def _persist_text_turn_memory(
    *,
    agent: ChatAgent,
    repo: Any,
    user_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    extraction_prompt = (
        "Extract durable memory updates from this conversation turn and return strict JSON only.\n"
        "Conversation turn:\n"
        f"USER: {user_text}\n"
        f"ASSISTANT: {assistant_text}\n\n"
        "Schema:\n"
        "{\n"
        '  "profile_updates": [{"field": "name|birthday_mmddyyyy|height_value|height_unit|city|country|sex|timezone|external_user_key", "value": "string|number|null"}],\n'
        '  "body_metric_events_insert": [{"metric_type": "weight|waist|blood_pressure", "value_primary": 0.0, "value_secondary": 0.0, "unit": "lbs|kg|in|cm|mmHg", "observed_at": "ISO-8601|null", "source": "manual|assistant|null", "confidence": 0.0, "notes": "string|null"}],\n'
        '  "persistence_ops": [{"operation": "insert|update|upsert", "target": "users|body_metric_events", "idempotency_key": "string|null"}]\n'
        "}\n"
        "Rules: only include explicit facts from this turn; do not infer missing values; return empty arrays if nothing to store."
    )

    try:
        result = await run_with_retry(
            agent,
            extraction_prompt,
            response_format=TextTurnStructuredOutput,
        )
        payload, raw_output = _coerce_text_turn_payload(result)
        if not payload.profile_updates and not payload.body_metric_events_insert:
            return

        idempotency_key = hashlib.sha256(
            f"{user_id}:{user_text}:{assistant_text}".encode("utf-8")
        ).hexdigest()
        persist_result = repo.apply_text_turn_submission(
            user_id=user_id,
            payload=payload,
            raw_structured_output=raw_output,
            idempotency_key=idempotency_key,
        )
        logger.info("Text-turn memory persisted: %s", persist_result)
    except Exception as exc:
        logger.warning("Could not persist text-turn memory: %s", exc)


async def _persist_thread_state(repo: Any, *, user_id: str, session_key: str, agent_name: str, thread: Any, summary_text: str | None) -> None:
    try:
        resolved_thread = await _resolve_maybe_awaitable(thread)
        serialized_thread = await resolved_thread.serialize()
        repo.upsert_thread_state(
            user_id=user_id,
            session_key=session_key,
            agent_name=agent_name,
            session_state=serialized_thread,
            summary_text=summary_text,
        )
    except Exception as exc:
        logger.error(
            "Could not save conversation memory to database. Continuing without persistence for this turn. Error: %s",
            exc,
        )


async def _handle_photo_submission(
    *,
    agent: ChatAgent,
    thread: Any,
    repo: Any,
    user_id: str,
    image_path: str,
) -> None:
    thread = await _resolve_maybe_awaitable(thread)
    _status("Preparing photo analysis request...")
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Image not found: {image_file}")
        return

    mime_type, _ = mimetypes.guess_type(image_file.name)
    mime_type = mime_type or "application/octet-stream"
    image_bytes = image_file.read_bytes()

    extraction_prompt = (
        "Analyze this meal image and return a strict JSON object matching this schema exactly:\n"
        "{\n"
        '  "profile_updates": [{"field": "name|birthday_mmddyyyy|height_value|height_unit|city|country|sex|timezone|external_user_key", "value": "string|number|null"}],\n'
        '  "meal_upsert": {"occurred_at": "ISO-8601|null", "meal_type": "breakfast|lunch|dinner|snack|other|null", "source_image_uri": "string|null", "source_hash": "string|null", "unit_system": "metric|imperial|null", "notes": "string|null"},\n'
        '  "meal_items_upsert": [{"food_label": "string", "quantity_value": 0.0, "quantity_unit": "string|null", "confidence": 0.0, "notes": "string|null"}],\n'
        '  "macro_events_insert": [{"calories_kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0, "sugar_g": 0.0, "sodium_mg": 0.0, "confidence": 0.0, "model_name": "string|null", "model_version": "string|null", "prompt_version": "string|null", "notes": "string|null"}],\n'
        '  "body_metric_events_insert": [{"metric_type": "weight|waist|blood_pressure", "value_primary": 0.0, "value_secondary": 0.0, "unit": "lbs|kg|in|cm|mmHg", "observed_at": "ISO-8601|null", "source": "photo|manual|assistant|null", "confidence": 0.0, "notes": "string|null"}],\n'
        '  "persistence_ops": [{"operation": "insert|update|upsert", "target": "users|body_metric_events|meal_events", "idempotency_key": "string|null"}]\n'
        "}\n"
        "If unknown, return empty arrays/nulls instead of guessing."
    )

    request_message = ChatMessage(
        role="user",
        contents=[
            TextContent(text=extraction_prompt),
            DataContent(data=image_bytes, media_type=mime_type),
        ],
    )

    provisional_key = None
    _status("Logging ingestion run...")
    run = repo.start_ingestion_run(
        user_id=user_id,
        source_type="meal_photo",
        idempotency_key=provisional_key,
        request_json={"image_path": str(image_file), "mime_type": mime_type},
    )

    try:
        _status("Analyzing image with LLM (streaming)...")
        result = await run_with_stream(
            agent,
            request_message,
            thread=thread,
            response_format=PhotoSubmissionStructuredOutput,
        )
        _status("Persisting extracted data to database...")
        payload, raw_output = _coerce_payload(result)
        idempotency_key = extract_idempotency_key(payload, image_bytes, user_id)
        persistence_result = repo.apply_photo_submission(
            user_id=user_id,
            image_path=str(image_file),
            payload=payload,
            raw_structured_output=raw_output,
            idempotency_key=idempotency_key,
        )
        repo.finish_ingestion_run(
            run_id=run.run_id,
            status="completed",
            response_json={"text": getattr(result, "text", "")},
            structured_output_json=raw_output,
            error=None,
        )
        _status("Photo analysis completed.")
        logger.info("Photo persisted: %s", persistence_result)
        if getattr(result, "usage_details", None):
            logger.info("Photo extraction tokens: %s", format_usage(result.usage_details))
    except Exception as exc:
        repo.finish_ingestion_run(
            run_id=run.run_id,
            status="failed",
            response_json=None,
            structured_output_json=None,
            error=str(exc),
        )
        logger.exception("Photo submission failed: %s", exc)


async def fitness_agent(image_path: str | None = None) -> None:
    """Run a multi-turn fitness and nutrition assistant with short-term thread memory and long-term database memory."""

    load_dotenv()
    cfg = get_config()
    startup_image = image_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    default_uid = cfg.ui.fitness.default_user_id
    user_id = os.getenv("FITNESS_USER_ID") or await _prompt_text(f"User ID [{default_uid}]: ") or default_uid
    session_key = os.getenv("FITNESS_SESSION_KEY") or f"fitness:{user_id}"
    agent_name = "fitness_agent"
    repo = get_fitness_repository()

    _status("Initializing fitness agent...")

    instructions = (
        "You are a fitness nutrition assistant with access to user profile, body metrics, and meal macro history. "
        "When users ask about goals or trends, use tracked data first and ask clarifying questions if missing data."
    )

    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    context_provider = DatabaseContextProvider(repo, user_id=user_id)
    agent = ChatAgent(
        chat_client=chat_client,
        instructions=instructions,
        name=agent_name,
        model=cfg.azure.openai.chat_deployment,
        context_providers=[context_provider],
        tools=[],
        max_completion_tokens=800,
        temperature=1.0,
    )

    saved_state = repo.load_thread_state(user_id=user_id, session_key=session_key, agent_name=agent_name)
    if saved_state:
        try:
            thread = await _resolve_maybe_awaitable(agent.deserialize_thread(saved_state))
        except Exception as exc:
            logger.error(
                "Could not restore prior conversation memory. Starting a new session thread. Error: %s",
                exc,
            )
            thread = await _resolve_maybe_awaitable(agent.get_new_thread())
    else:
        thread = await _resolve_maybe_awaitable(agent.get_new_thread())

    logger.info("Fitness assistant started for user_id=%s session_key=%s", user_id, session_key)
    _print_help()

    if startup_image:
        await _handle_photo_submission(
            agent=agent,
            thread=thread,
            repo=repo,
            user_id=user_id,
            image_path=startup_image,
        )
        await _persist_thread_state(
            repo,
            user_id=user_id,
            session_key=session_key,
            agent_name=agent_name,
            thread=thread,
            summary_text=f"Processed startup photo: {startup_image}",
        )

    while True:
        user_input = await _prompt_text("fitness> ")
        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"/exit", "exit", "quit"}:
            break
        if lowered in {"/help", "help"}:
            _print_help()
            continue
        if lowered == "/summary":
            snapshot = repo.get_read_model(user_id)
            print(json.dumps(snapshot, indent=2, ensure_ascii=False))
            continue
        if lowered.startswith("/photo"):
            command_path = _extract_photo_path(user_input)
            target_path = command_path or await _prompt_text("Image path: ")
            if not target_path:
                print("No image path provided.")
                continue
            await _handle_photo_submission(
                agent=agent,
                thread=thread,
                repo=repo,
                user_id=user_id,
                image_path=target_path,
            )
            await _persist_thread_state(
                repo,
                user_id=user_id,
                session_key=session_key,
                agent_name=agent_name,
                thread=thread,
                summary_text=f"Processed photo: {target_path}",
            )
            continue

        try:
            _status("Waiting for LLM response (streaming)...")
            thread = await _resolve_maybe_awaitable(thread)
            result = await run_with_stream(agent, user_input, thread=thread)
            assistant_text = getattr(result, "text", "") or ""
            await _persist_text_turn_memory(
                agent=agent,
                repo=repo,
                user_id=user_id,
                user_text=user_input,
                assistant_text=assistant_text,
            )
            if getattr(result, "usage_details", None):
                logger.info("Chat request tokens: %s", format_usage(result.usage_details))
            await _persist_thread_state(
                repo,
                user_id=user_id,
                session_key=session_key,
                agent_name=agent_name,
                thread=thread,
                summary_text=user_input,
            )
        except RateLimitError:
            logger.warning(
                "Rate limit reached. Please wait ~60 seconds and retry. "
                "Consider lowering request frequency or requesting a quota increase."
            )

    await _persist_thread_state(
        repo,
        user_id=user_id,
        session_key=session_key,
        agent_name=agent_name,
        thread=thread,
        summary_text="session-exit",
    )
    logger.info("Fitness assistant exited.")
