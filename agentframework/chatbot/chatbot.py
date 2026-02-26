import asyncio
import hashlib
import html
import io
import json
import logging
import mimetypes
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import yaml
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL_ENV = "LOG_LEVEL"
DEBUG_LOG_MAX_LINES_ENV = "DEBUG_LOG_MAX_LINES"


class _SessionLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            logs = st.session_state.setdefault("debug_logs", [])
            logs.append(message)
            max_lines = int(os.getenv(DEBUG_LOG_MAX_LINES_ENV, "200"))
            if len(logs) > max_lines:
                del logs[:-max_lines]
        except Exception:
            pass


def _ensure_debug_log_handler() -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if getattr(handler, "name", "") == "streamlit_debug":
            return
    handler = _SessionLogHandler()
    handler.name = "streamlit_debug"
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger.addHandler(handler)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agent_framework import ChatAgent, ChatMessage, DataContent, TextContent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from ai_chat_client import KaitoChatClient
from fitness_memory import (
    DatabaseContextProvider,
    PhotoSubmissionStructuredOutput,
    TextTurnStructuredOutput,
    extract_idempotency_key,
    get_fitness_repository,
)
from main import agent1, azure_foundry_general_agent, load_prompt_template, render_instructions
from run_utils import run_with_retry
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _clean_env(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip('"').strip("'")


def _split_models(value: str | list[str] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [_clean_env(str(item)) for item in value if _clean_env(str(item))]
    return [_clean_env(item) for item in value.split(",") if _clean_env(item)]


def _load_chatbot_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"providers": [], "agents": [], "ui": {}}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {"providers": [], "agents": [], "ui": {}}


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().strip('"').strip("'").rstrip("/")
    if endpoint.endswith("/v1/chat/completions"):
        return endpoint
    return f"{endpoint}/v1/chat/completions"


def _is_running_in_kubernetes() -> bool:
    return bool(os.getenv("KUBERNETES_SERVICE_HOST"))


def _is_cluster_local_endpoint(endpoint: str | None) -> bool:
    cleaned = _clean_env(endpoint)
    if not cleaned:
        return False
    parsed = urlparse(cleaned)
    return bool(parsed.hostname and parsed.hostname.endswith(".svc.cluster.local"))


def _post_chat_completion(
    *,
    endpoint: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None,
    max_tokens: int | None,
    top_p: float | None,
    verify_tls: bool,
) -> dict:
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        _normalize_endpoint(endpoint),
        data=data,
        headers=headers,
        method="POST",
    )

    context = None
    if not verify_tls:
        import ssl

        context = ssl._create_unverified_context()

    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def _extract_display_text(payload: object) -> str:
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return ""
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return text
        else:
            return text

    if isinstance(payload, dict):
        if "content" in payload and isinstance(payload["content"], str):
            return payload["content"].strip()
        if "answer" in payload and isinstance(payload["answer"], str):
            return payload["answer"].strip()
        if "text" in payload and isinstance(payload["text"], str):
            return payload["text"].strip()
        if "choices" in payload and isinstance(payload["choices"], list) and payload["choices"]:
            choice = payload["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message", {})
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"].strip()
                if isinstance(choice.get("text"), str):
                    return choice["text"].strip()

        return json.dumps(payload, indent=2)

    return str(payload)


def _format_agent1_output(raw_output: str) -> str:
    if not raw_output:
        return ""
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    step1 = ""
    step2 = ""
    step3 = ""
    current_step = None
    for line in lines:
        if line.lower().startswith("step 1 result"):
            current_step = "step1"
            inline = line.split(":", 1)[1].strip() if ":" in line else ""
            if inline:
                step1 = f"{step1} {inline}".strip()
            continue
        if line.lower().startswith("step 2 workflow"):
            current_step = "step2"
            inline = line.split(":", 1)[1].strip() if ":" in line else ""
            if inline:
                step2 = f"{step2} {inline}".strip()
            continue
        if line.lower().startswith("step 3 structured output"):
            current_step = "step3"
            inline = line.split(":", 1)[1].strip() if ":" in line else ""
            if inline:
                step3 = f"{step3} {inline}".strip()
            continue
        if line.lower().startswith("tokens:") or line.lower().startswith("hello from agentframework"):
            continue

        if current_step == "step1":
            step1 = f"{step1} {line}".strip()
        elif current_step == "step2":
            step2 = f"{step2} {line}".strip()
        elif current_step == "step3":
            step3 = f"{step3} {line}".strip()

    if not (step1 or step2 or step3):
        return raw_output

    def _pretty_json(text: str) -> str:
        text = text.strip()
        if not text:
            return text
        try:
            return json.dumps(json.loads(text), indent=2)
        except json.JSONDecodeError:
            return text

    parts = []
    if step1:
        parts.append(f"**Step 1:**\n\n    {step1}")
    if step2:
        step2_text = _pretty_json(_extract_display_text(step2))
        parts.append("**Step 2:**\n\n" + "\n".join(f"    {line}" for line in step2_text.splitlines()))
    if step3:
        step3_text = _pretty_json(_extract_display_text(step3))
        parts.append("**Step 3:**\n\n" + "\n".join(f"    {line}" for line in step3_text.splitlines()))
    return "\n\n".join(parts)


def _build_kaito_agent(model: str) -> ChatAgent:
    prompt_path = Path(
        os.getenv(
            "PROMPT_TEMPLATE_PATH",
            PROJECT_ROOT / "prompts" / "assistant_jinja.yaml",
        )
    )
    prompt = load_prompt_template(prompt_path)

    data_input = ""
    instructions = render_instructions(
        prompt.get("instructions", "You are a helpful assistant."),
        {"data_input": data_input},
    )

    model_block = prompt.get("model", {})
    model_id = model_block.get("id") if isinstance(model_block, dict) else model_block
    model_id = model or model_id or os.getenv("KAITO_MODEL", "phi-4-mini-instruct")
    if not model_id:
        raise ValueError("KAITO model is required. Set KAITO_MODEL or select a model.")

    chat_client = KaitoChatClient(
        endpoint=os.getenv("KAITO_INFERENCE_ENDPOINT"),
        api_key=os.getenv("KAITO_API_KEY"),
        default_model=model_id,
    )

    return ChatAgent(
        chat_client=chat_client,
        instructions=instructions,
        name=prompt.get("name", "KaitoAssistant"),
        model=model_id,
        tools=prompt.get("tools", []),
        max_tokens=prompt.get("max_tokens"),
    )


def _build_kaito_ragengine_agent(model: str, index_name: str | None = None) -> ChatAgent:
    """Build a ChatAgent targeting a KAITO RAGEngine deployment."""
    prompt_path = Path(
        os.getenv(
            "PROMPT_TEMPLATE_PATH",
            PROJECT_ROOT / "prompts" / "assistant_jinja.yaml",
        )
    )
    prompt = load_prompt_template(prompt_path)

    instructions = render_instructions(
        prompt.get("instructions", "You are a helpful assistant."),
        {"data_input": ""},
    )

    model_block = prompt.get("model", {})
    model_id = model_block.get("id") if isinstance(model_block, dict) else model_block
    model_id = model or model_id or os.getenv("KAITO_MODEL", "phi-4-mini-instruct")

    rag_index = index_name or os.getenv("KAITO_RAGENGINE_INDEX", "rag_index")

    chat_client = KaitoChatClient(
        endpoint=os.getenv("KAITO_RAGENGINE_ENDPOINT"),
        api_key=os.getenv("KAITO_RAGENGINE_API_KEY"),
        default_model=model_id,
        extra_payload={"index_name": rag_index},
    )

    return ChatAgent(
        chat_client=chat_client,
        instructions=instructions,
        name=prompt.get("name", "KaitoRAGEngineAssistant"),
        model=model_id,
        tools=prompt.get("tools", []),
        max_tokens=prompt.get("max_tokens"),
    )


def _coerce_photo_payload(result: object) -> tuple[PhotoSubmissionStructuredOutput, dict]:
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


def _coerce_text_turn_payload(result: object) -> tuple[TextTurnStructuredOutput, dict]:
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
    repo: object,
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
            _append_memory_debug_event("text_persist", "no explicit profile/body metric facts extracted")
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
        profile_debug = persist_result.get("profile_debug", {}) if isinstance(persist_result, dict) else {}
        applied_fields = profile_debug.get("applied_fields", []) if isinstance(profile_debug, dict) else []
        normalized_fields = profile_debug.get("normalized_fields", []) if isinstance(profile_debug, dict) else []
        skipped_fields = profile_debug.get("skipped_fields", []) if isinstance(profile_debug, dict) else []
        _append_memory_debug_event(
            "text_persist",
            (
                f"saved profile_updates={len(payload.profile_updates)} body_metrics={len(payload.body_metric_events_insert)} "
                f"applied={applied_fields or ['none']} normalized={normalized_fields or ['none']} skipped={skipped_fields or ['none']}"
            ),
        )
    except Exception as exc:
        logger.warning("Could not persist text-turn memory: %s", exc)
        _append_memory_debug_event("text_persist", f"failed: {exc}")


async def _resolve_maybe_awaitable(value: object) -> object:
    if hasattr(value, "__await__"):
        return await value
    return value


def _normalize_user_id(user_name: str) -> str:
    cleaned = (user_name or "").strip()
    return cleaned if cleaned else "default-user"


def _browser_timezone_name() -> str:
    try:
        context_tz = getattr(getattr(st, "context", None), "timezone", None)
        if isinstance(context_tz, str) and context_tz.strip():
            return context_tz.strip()
    except Exception:
        pass
    return "UTC"


def _short_local_datetime(value: str | None) -> str:
    if not value:
        return ""
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        tz = ZoneInfo(_browser_timezone_name())
        return dt.astimezone(tz).strftime("%m/%d/%Y %I:%M %p")
    except Exception:
        return str(value)


def _resolve_memory_user_id(user_name: str) -> tuple[str, bool]:
    normalized = _normalize_user_id(user_name)
    db_path = _fitness_db_path()
    if not db_path.exists():
        return normalized, False

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            by_id = conn.execute(
                "SELECT user_id FROM users WHERE lower(user_id) = lower(?) LIMIT 1",
                (normalized,),
            ).fetchone()
            if by_id:
                return str(by_id["user_id"]), False

            by_name = conn.execute(
                "SELECT user_id FROM users WHERE lower(name) = lower(?) ORDER BY updated_at DESC LIMIT 1",
                (normalized,),
            ).fetchone()
            if by_name:
                return str(by_name["user_id"]), True
    except Exception:
        logger.exception("Could not resolve memory user id by name")

    return normalized, False


def _sum_structured_macros(meal: dict) -> dict[str, float | None]:
    raw = meal.get("llm_structured_output_json")
    if not raw:
        return {"calories_kcal": None, "protein_g": None, "carbs_g": None, "fat_g": None}
    try:
        loaded = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {"calories_kcal": None, "protein_g": None, "carbs_g": None, "fat_g": None}

    events = loaded.get("macro_events_insert") if isinstance(loaded, dict) else None
    if not isinstance(events, list) or not events:
        return {"calories_kcal": None, "protein_g": None, "carbs_g": None, "fat_g": None}

    def _sum_field(field: str) -> float | None:
        vals = []
        for event in events:
            if isinstance(event, dict) and event.get(field) is not None:
                try:
                    vals.append(float(event.get(field)))
                except Exception:
                    pass
        if not vals:
            return None
        return round(sum(vals), 2)

    return {
        "calories_kcal": _sum_field("calories_kcal"),
        "protein_g": _sum_field("protein_g"),
        "carbs_g": _sum_field("carbs_g"),
        "fat_g": _sum_field("fat_g"),
    }


def _estimate_macros_from_labels(labels: list[str]) -> dict[str, float | None]:
    if not labels:
        return {"calories_kcal": None, "protein_g": None, "carbs_g": None, "fat_g": None}

    per_item_defaults = {
        "pizza dough": (280.0, 9.0, 56.0, 3.0),
        "tomato sauce": (40.0, 2.0, 8.0, 0.0),
        "mozzarella": (170.0, 12.0, 3.0, 12.0),
        "basil": (2.0, 0.2, 0.3, 0.0),
        "olive oil": (119.0, 0.0, 0.0, 14.0),
        "salmon": (233.0, 25.0, 0.0, 14.0),
        "romaine": (15.0, 1.0, 3.0, 0.0),
        "kimchi": (23.0, 1.0, 4.0, 0.0),
        "flour tortilla": (220.0, 6.0, 36.0, 5.0),
        "egg": (182.0, 12.0, 2.0, 14.0),
        "beef": (250.0, 26.0, 0.0, 15.0),
        "tomato": (22.0, 1.0, 5.0, 0.0),
        "lettuce": (15.0, 1.0, 3.0, 0.0),
        "cheese": (160.0, 10.0, 2.0, 12.0),
        "chicken": (240.0, 27.0, 0.0, 14.0),
        "rice": (205.0, 4.0, 45.0, 0.4),
        "noodle": (220.0, 7.0, 40.0, 3.0),
    }
    unknown_default = (120.0, 6.0, 12.0, 4.0)

    total_kcal = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0

    for label in labels:
        key = (label or "").strip().lower()
        matched = None
        for token, macro in per_item_defaults.items():
            if token in key:
                matched = macro
                break
        kcal, protein, carbs, fat = matched or unknown_default
        total_kcal += kcal
        total_protein += protein
        total_carbs += carbs
        total_fat += fat

    return {
        "calories_kcal": round(total_kcal, 2),
        "protein_g": round(total_protein, 2),
        "carbs_g": round(total_carbs, 2),
        "fat_g": round(total_fat, 2),
    }


def _display_meal_macros(meal: dict) -> tuple[str, str, str, str]:
    kcal = meal.get("calories_kcal")
    protein = meal.get("protein_g")
    carbs = meal.get("carbs_g")
    fat = meal.get("fat_g")

    if all(v in (None, 0, 0.0, "") for v in [kcal, protein, carbs, fat]):
        derived = _sum_structured_macros(meal)
        kcal = derived.get("calories_kcal")
        protein = derived.get("protein_g")
        carbs = derived.get("carbs_g")
        fat = derived.get("fat_g")

    if all(v in (None, 0, 0.0, "") for v in [kcal, protein, carbs, fat]):
        labels = _extract_detected_food_labels(meal)
        estimated = _estimate_macros_from_labels(labels)
        kcal = estimated.get("calories_kcal")
        protein = estimated.get("protein_g")
        carbs = estimated.get("carbs_g")
        fat = estimated.get("fat_g")

    def _fmt(value: float | int | str | None, suffix: str = "") -> str:
        if value is None or value == "":
            return "n/a"
        return f"{value}{suffix}"

    return _fmt(kcal), _fmt(protein, "g"), _fmt(carbs, "g"), _fmt(fat, "g")


def _extract_detected_food_labels(meal: dict) -> list[str]:
    raw = meal.get("llm_structured_output_json")
    if not raw:
        return []
    try:
        loaded = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    if not isinstance(loaded, dict):
        return []
    items = loaded.get("meal_items_upsert")
    if not isinstance(items, list):
        return []
    labels: list[str] = []
    for item in items:
        if isinstance(item, dict):
            label = str(item.get("food_label") or "").strip()
            if label:
                labels.append(label)
    return labels

def _meal_short_description(meal: dict) -> str:
    labels = _extract_detected_food_labels(meal)
    if labels:
        snippet = ", ".join(labels[:3])
        if len(labels) > 3:
            snippet = f"{snippet}, and more"
        return f"Detected meal: {snippet}."

    raw = meal.get("llm_structured_output_json")
    if raw:
        try:
            loaded = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(loaded, dict):
                meal_upsert = loaded.get("meal_upsert")
                if isinstance(meal_upsert, dict):
                    notes = str(meal_upsert.get("notes") or "").strip()
                    if notes:
                        return notes
        except Exception:
            pass
    return "Meal photo logged."


def _ensure_macro_events_in_output(raw_output: dict) -> dict:
    if not isinstance(raw_output, dict):
        return raw_output
    events = raw_output.get("macro_events_insert")
    if isinstance(events, list) and len(events) > 0:
        return raw_output
    items = raw_output.get("meal_items_upsert")
    if not isinstance(items, list) or not items:
        return raw_output
    labels = []
    for item in items:
        if isinstance(item, dict):
            label = str(item.get("food_label") or "").strip()
            if label:
                labels.append(label)
    estimated = _estimate_macros_from_labels(labels)
    if estimated.get("calories_kcal") is None:
        return raw_output

    raw_output["macro_events_insert"] = [
        {
            "calories_kcal": estimated.get("calories_kcal"),
            "protein_g": estimated.get("protein_g"),
            "carbs_g": estimated.get("carbs_g"),
            "fat_g": estimated.get("fat_g"),
            "fiber_g": None,
            "sugar_g": None,
            "sodium_mg": None,
            "confidence": 0.3,
            "model_name": "heuristic-food-label-estimator",
            "model_version": "1",
            "prompt_version": "fallback-v1",
            "notes": "Estimated from detected food labels because model returned no macro_events_insert.",
        }
    ]
    return raw_output


def _parse_birthday_mmddyyyy(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%m%d%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _age_from_birthday(value: str | None) -> int | None:
    dob = _parse_birthday_mmddyyyy(value)
    if dob is None:
        return None
    now_local = datetime.now(ZoneInfo(_browser_timezone_name()))
    years = now_local.year - dob.year
    if (now_local.month, now_local.day) < (dob.month, dob.day):
        years -= 1
    return max(years, 0)


def _latest_metric(metrics: list[dict], metric_type: str) -> dict | None:
    for item in metrics:
        if item.get("metric_type") == metric_type:
            return item
    return None


def _pretty_profile_key(key: str) -> str:
    return key.replace("_", " ").strip().title()


def _append_memory_debug_event(event: str, details: str) -> None:
    try:
        logs = st.session_state.setdefault("memory_debug_events", [])
        stamp = datetime.now(ZoneInfo(_browser_timezone_name())).strftime("%m/%d/%Y %I:%M:%S %p")
        logs.append(f"[{stamp}] {event}: {details}")
        if len(logs) > 400:
            del logs[:-400]
    except Exception:
        pass


def _estimate_tokens_from_text(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def _estimate_tokens_from_chat_messages(messages: list[ChatMessage]) -> int:
    total = 0
    for message in messages:
        total += _estimate_tokens_from_text(getattr(message, "text", "") or "")
        total += 8
    return total


def _parse_usage_summary(usage_summary: str) -> dict[str, int]:
    parsed = {"input": 0, "output": 0, "total": 0}
    for token in (usage_summary or "").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in parsed:
            try:
                parsed[key] = int(value)
            except ValueError:
                parsed[key] = 0
    return parsed


def _format_metrics_entry(entry: str) -> str:
    latency_match = re.search(r"latency_s=([0-9]+(?:\.[0-9]+)?)", entry)
    if not latency_match:
        return entry
    try:
        latency = float(latency_match.group(1))
    except ValueError:
        return entry
    if latency >= 20:
        return f"🟥 slow {entry}"
    if latency >= 10:
        return f"🟨 elevated {entry}"
    return f"🟩 ok {entry}"


def _render_longterm_meal_block(meal: dict) -> str:
    meal_title = meal.get("meal_type") or "meal"
    meal_time = _short_local_datetime(meal.get("occurred_at"))
    description = _meal_short_description(meal)
    kcal, protein, carbs, fat = _display_meal_macros(meal)

    lines = [
        f"<div class='meal-title'>{html.escape(str(meal_title))} {html.escape(str(meal_time))}</div>",
        f"<div class='meal-line'>{html.escape(str(description))}</div>",
        f"<div class='meal-line'>kcal={html.escape(str(kcal))} | protein={html.escape(str(protein))} | carbs={html.escape(str(carbs))} | fat={html.escape(str(fat))}</div>",
    ]

    if kcal == "n/a" and protein == "n/a" and carbs == "n/a" and fat == "n/a":
        labels = _extract_detected_food_labels(meal)
        if not labels:
            lines.append("<div class='meal-line'>No detected foods in structured output.</div>")

    return f"<div class='longterm-meal'>{''.join(lines)}</div>"


def _fitness_db_path() -> Path:
    env_path = _clean_env(os.getenv("FITNESS_DB_PATH"))
    if env_path:
        return Path(env_path)
    return PROJECT_ROOT / "agentframework.db"


def _fitness_chat_model(selected_model: str | None) -> str:
    return selected_model or _clean_env(os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-5.2-chat"))


def _build_fitness_runtime(user_id: str, selected_model: str | None) -> tuple[ChatAgent, object, str, str]:
    repo = get_fitness_repository(_fitness_db_path())
    session_key = f"fitness:{user_id}"
    agent_name = "fitness_agent"
    instructions = (
        "You are a fitness nutrition assistant with access to user profile, body metrics, and meal macro history. "
        "When users ask about goals or trends, use tracked data first and ask clarifying questions if missing data."
    )
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    context_provider = DatabaseContextProvider(repo, user_id=user_id, meal_limit=6)
    agent = ChatAgent(
        chat_client=chat_client,
        instructions=instructions,
        name=agent_name,
        model=_fitness_chat_model(selected_model),
        context_providers=[context_provider],
        tools=[],
        max_completion_tokens=800,
        temperature=1.0,
    )
    return agent, repo, session_key, agent_name


async def _run_fitness_turn(
    *,
    user_prompt: str,
    user_id: str,
    selected_model: str | None,
    image_bytes: bytes | None,
    image_name: str | None,
) -> tuple[str, str]:
    agent, repo, session_key, agent_name = _build_fitness_runtime(user_id, selected_model)
    _append_memory_debug_event(
        "run_start",
        f"user_id={user_id} session_key={session_key} model={_fitness_chat_model(selected_model)}",
    )
    saved_state = repo.load_thread_state(user_id=user_id, session_key=session_key, agent_name=agent_name)
    if saved_state:
        try:
            thread = await _resolve_maybe_awaitable(agent.deserialize_thread(saved_state))
            _append_memory_debug_event("thread_restore", "restored prior thread state")
        except Exception:
            thread = await _resolve_maybe_awaitable(agent.get_new_thread())
            _append_memory_debug_event("thread_restore", "restore failed, created new thread")
    else:
        thread = await _resolve_maybe_awaitable(agent.get_new_thread())
        _append_memory_debug_event("thread_restore", "no prior state, created new thread")

    usage_summary = ""
    if image_bytes is not None:
        mime_type, _ = mimetypes.guess_type(image_name or "")
        mime_type = mime_type or "application/octet-stream"
        request_message = ChatMessage(
            role="user",
            contents=[
                TextContent(text=user_prompt),
                DataContent(data=image_bytes, media_type=mime_type),
            ],
        )
        result = await run_with_retry(agent, request_message, thread=thread)
        content = _extract_display_text(getattr(result, "text", None) or str(result))
        usage = getattr(result, "usage_details", None)
        if usage:
            usage_summary = (
                f"input={usage.input_token_count or 0} "
                f"output={usage.output_token_count or 0} "
                f"total={usage.total_token_count or 0}"
            )

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
            "Rules:\n"
            "- If a meal is visible, provide best-effort numeric macro estimates in macro_events_insert (do not leave it empty).\n"
            "- Use conservative estimates with lower confidence when uncertain.\n"
            "- Return empty arrays/nulls only when the image is not food, unreadable, or insufficient for estimation."
        )
        extraction_message = ChatMessage(
            role="user",
            contents=[
                TextContent(text=extraction_prompt),
                DataContent(data=image_bytes, media_type=mime_type),
            ],
        )
        try:
            extraction_result = await run_with_retry(
                agent,
                extraction_message,
                response_format=PhotoSubmissionStructuredOutput,
            )
            payload, raw_output = _coerce_photo_payload(extraction_result)
            raw_output = _ensure_macro_events_in_output(raw_output)
            payload = PhotoSubmissionStructuredOutput.model_validate(raw_output)
            file_hint = image_name or "uploaded-image"
            idempotency_key = extract_idempotency_key(payload, image_bytes, user_id)
            repo.apply_photo_submission(
                user_id=user_id,
                image_path=file_hint,
                payload=payload,
                raw_structured_output=raw_output,
                idempotency_key=idempotency_key,
            )
            _append_memory_debug_event(
                "photo_persist",
                f"saved meal photo submission image={file_hint} profile_updates={len(payload.profile_updates)} body_metrics={len(payload.body_metric_events_insert)}",
            )
        except Exception:
            logger.exception("Could not persist photo extraction memory")
            _append_memory_debug_event("photo_persist", "failed to persist photo extraction")

        await _persist_text_turn_memory(
            agent=agent,
            repo=repo,
            user_id=user_id,
            user_text=user_prompt,
            assistant_text=content,
        )
    else:
        result = await run_with_retry(agent, user_prompt, thread=thread)
        content = _extract_display_text(getattr(result, "text", None) or str(result))
        usage = getattr(result, "usage_details", None)
        if usage:
            usage_summary = (
                f"input={usage.input_token_count or 0} "
                f"output={usage.output_token_count or 0} "
                f"total={usage.total_token_count or 0}"
            )

        await _persist_text_turn_memory(
            agent=agent,
            repo=repo,
            user_id=user_id,
            user_text=user_prompt,
            assistant_text=content,
        )

    try:
        resolved_thread = await _resolve_maybe_awaitable(thread)
        serialized_thread = await resolved_thread.serialize()
        repo.upsert_thread_state(
            user_id=user_id,
            session_key=session_key,
            agent_name=agent_name,
            session_state=serialized_thread,
            summary_text=user_prompt,
        )
        _append_memory_debug_event("thread_save", "thread state persisted")
    except Exception:
        logger.exception("Could not save fitness thread state")
        _append_memory_debug_event("thread_save", "failed to persist thread state")

    _append_memory_debug_event("run_end", "fitness turn completed")

    return content, usage_summary


st.set_page_config(page_title="AI Foundry Chatbot", page_icon="🤖", layout="wide")

st.title("AI Foundry Chatbot")

st.markdown(
        """
<style>
div[data-testid="stAppViewContainer"] div.block-container {
    max-width: 98%;
    padding-left: 1rem;
    padding-right: 1rem;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
    position: sticky;
    top: 0.75rem;
    align-self: flex-start;
    max-height: calc(100vh - 1.5rem);
    overflow-y: auto;
    padding-right: 0.25rem;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child > div[data-testid="stVerticalBlock"] {
    position: sticky;
    top: 0.75rem;
}

.longterm-meal {
    color: #ffffff !important;
    font-size: 0.86rem;
    line-height: 1.22;
    margin: 0 0 0.65rem 0;
    padding: 0.15rem 0;
}

.longterm-meal .meal-title {
    color: #ffffff !important;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
}

.longterm-meal .meal-line {
    color: #ffffff !important;
    margin: 0.15rem 0;
}
</style>
        """,
        unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics_log" not in st.session_state:
    st.session_state.metrics_log = []
if "_metrics_rerun" not in st.session_state:
    st.session_state._metrics_rerun = False
elif st.session_state._metrics_rerun:
    st.session_state._metrics_rerun = False

CHATBOT_CONFIG = _load_chatbot_config()
PROVIDERS = CHATBOT_CONFIG.get("providers", [])
AGENTS = CHATBOT_CONFIG.get("agents", [])
UI_CONFIG = CHATBOT_CONFIG.get("ui", {})
LOG_LEVEL_ENV = UI_CONFIG.get("log_level_env", LOG_LEVEL_ENV)
DEBUG_LOG_MAX_LINES_ENV = UI_CONFIG.get("debug_log_max_lines_env", DEBUG_LOG_MAX_LINES_ENV)

logging.basicConfig(
    level=os.getenv(LOG_LEVEL_ENV, "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("chatbot")
AGENT_OPTIONS = [agent.get("name") for agent in AGENTS if agent.get("name")] or [
    "Azure Foundry General",
    "Kaito Assistant",
    "Fitness Nutrition",
    "Agent1 Demo",
]
uploaded_food_image = None
uploaded_food_image_bytes = None
uploaded_food_image_name = ""
fitness_user_name = st.session_state.get("fitness_user_name", "roy")
chat_input_key = "chat_prompt_input"

with st.sidebar:
    st.header("Settings")
    default_agent_index = AGENT_OPTIONS.index("Fitness Nutrition") if "Fitness Nutrition" in AGENT_OPTIONS else 0
    agent_choice = st.selectbox("Agent", AGENT_OPTIONS, index=default_agent_index)
    agent_config = next((agent for agent in AGENTS if agent.get("name") == agent_choice), {})
    provider_name = agent_config.get("provider")
    provider_config = next((p for p in PROVIDERS if p.get("name") == provider_name), {})
    if not provider_config and PROVIDERS:
        provider_config = PROVIDERS[0]
        provider_name = provider_config.get("name")

    st.text_input("Provider", provider_name or "", disabled=True)

    endpoint_env = provider_config.get("endpoint_env")
    endpoint_default = provider_config.get("default_endpoint", "")
    endpoint = _clean_env(os.getenv(endpoint_env, endpoint_default)) if endpoint_env else endpoint_default

    api_key_env = provider_config.get("api_key_env")
    api_key = _clean_env(os.getenv(api_key_env)) if api_key_env else ""

    models_env = provider_config.get("models_env")
    if isinstance(models_env, list):
        models = _split_models(models_env)
    else:
        models = _split_models(os.getenv(models_env)) if models_env else []

    model_env = provider_config.get("model_env")
    provider_default_model = provider_config.get("default_model", "")
    if not models:
        model_fallback = _clean_env(os.getenv(model_env, provider_default_model)) if model_env else provider_default_model
        models = [model for model in [model_fallback] if model]

    agent_model = agent_config.get("model")
    if agent_model and agent_model not in models:
        models.append(agent_model)

    model_default = agent_model or (models[0] if models else "")

    endpoint = st.text_input("Endpoint", endpoint, help="Base endpoint or full /v1/chat/completions URL")
    if api_key_env:
        api_key = st.text_input("API Key", api_key, type="password")
    model_options = models or [model_default]
    model_key = "model_select"
    if model_key in st.session_state and st.session_state[model_key] in model_options:
        model_index = model_options.index(st.session_state[model_key])
    elif model_default in model_options:
        model_index = model_options.index(model_default)
    else:
        model_index = 0
    model = st.selectbox("Model", model_options, index=model_index, key=model_key)

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens", min_value=1, max_value=4096, value=512, step=1)
    top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.05)
    verify_tls = st.checkbox("Verify TLS", value=True)
    debug_enabled = st.checkbox("Debug mode", value=True)

    if debug_enabled:
        _ensure_debug_log_handler()

    if st.button("New chat"):
        st.session_state.messages = []

    if agent_choice in {"Kaito Assistant", "KAITO RAG Assistant"} and _is_cluster_local_endpoint(endpoint):
        if not _is_running_in_kubernetes():
            st.warning(
                "This endpoint uses Kubernetes cluster-local DNS (*.svc.cluster.local) and is not reachable from this runtime. "
                "Use `kubectl port-forward` and set Endpoint to a localhost URL such as "
                "`http://127.0.0.1:8000/v1/chat/completions`."
            )

    st.divider()
    st.subheader("Completion metrics")
    metrics_container = st.container(height=220)
    with metrics_container:
        if st.session_state.metrics_log:
            for entry in reversed(st.session_state.metrics_log[-50:]):
                st.caption(_format_metrics_entry(entry))
        else:
            st.caption("No completions yet.")

if agent_choice == "Fitness Nutrition":
    chat_col, right_col = st.columns([3.2, 1.2], gap="large")
else:
    chat_col, right_col = st.container(), None

if right_col is not None:
    with right_col:
        st.subheader("Fitness Context")
        fitness_user_name_input = st.text_input(
            "User name",
            value=fitness_user_name,
            key="fitness_user_name",
            help="Used as the user context key for long-term memory retrieval.",
        )
        memory_user_id, resolved_from_name = _resolve_memory_user_id(fitness_user_name_input)
        if resolved_from_name:
            st.caption(f"Retrieving profile for user: {fitness_user_name_input} (matched user_id: {memory_user_id})")
        else:
            st.caption(f"Retrieving profile for user: {memory_user_id}")

        profile = {}
        recent_body_metrics = []
        meals = []
        memory_error = None
        try:
            memory_repo = get_fitness_repository(_fitness_db_path())
            read_model = memory_repo.get_read_model(memory_user_id, meal_limit=6)
            profile = read_model.get("profile", {}) or {}
            recent_body_metrics = read_model.get("recent_body_metrics", []) or []
            meals = read_model.get("recent_meals", []) or []
        except Exception as exc:
            logger.exception("Could not load fitness memory panel")
            memory_error = str(exc)

        st.markdown("**User Profile**")
        if memory_error:
            st.caption(f"Could not load profile: {memory_error}")
        elif profile:
            profile_lines = []
            ordered_user_fields = [
                "name",
                "birthday_mmddyyyy",
                "height_value",
                "height_unit",
                "sex",
                "city",
                "country",
                "timezone",
                "external_user_key",
            ]

            def _display_or_na(value: object) -> str:
                if value is None:
                    return "n/a"
                text = str(value).strip()
                return text if text else "n/a"

            birthday = profile.get("birthday_mmddyyyy")
            age = _age_from_birthday(birthday)

            for key in ordered_user_fields:
                value = profile.get(key)
                if key == "birthday_mmddyyyy":
                    profile_lines.append(f"Birthday: {_display_or_na(value)}")
                    continue
                if key == "height_value":
                    h_val = _display_or_na(profile.get("height_value"))
                    h_unit = _display_or_na(profile.get("height_unit"))
                    if h_val == "n/a" and h_unit == "n/a":
                        profile_lines.append("Height: n/a")
                    elif h_unit == "n/a":
                        profile_lines.append(f"Height: {h_val}")
                    else:
                        profile_lines.append(f"Height: {h_val} {h_unit}")
                    continue
                if key == "height_unit":
                    continue
                profile_lines.append(f"{_pretty_profile_key(key)}: {_display_or_na(value)}")

            profile_lines.append(f"Current Age: {age if age is not None else 'n/a'}")

            latest_weight = _latest_metric(recent_body_metrics, "weight")
            latest_waist = _latest_metric(recent_body_metrics, "waist")
            latest_bp = _latest_metric(recent_body_metrics, "blood_pressure")

            if latest_weight:
                observed = _short_local_datetime(latest_weight.get("observed_at"))
                profile_lines.append(
                    f"Current Weight: {latest_weight.get('value_primary')} {latest_weight.get('unit')} ({observed})"
                )
            else:
                profile_lines.append("Current Weight: n/a")
            if latest_waist:
                observed = _short_local_datetime(latest_waist.get("observed_at"))
                profile_lines.append(
                    f"Current Waist: {latest_waist.get('value_primary')} {latest_waist.get('unit')} ({observed})"
                )
            else:
                profile_lines.append("Current Waist: n/a")
            if latest_bp:
                observed = _short_local_datetime(latest_bp.get("observed_at"))
                profile_lines.append(
                    f"Current Blood Pressure: {latest_bp.get('value_primary')}/{latest_bp.get('value_secondary')} {latest_bp.get('unit')} ({observed})"
                )
            else:
                profile_lines.append("Current Blood Pressure: n/a")

            if profile_lines:
                st.text("\n".join(profile_lines))
            else:
                st.caption("No profile fields populated yet.")
        else:
            st.caption("No profile found for this user.")

        st.divider()
        uploaded_food_image = st.file_uploader(
            "Upload food image (optional)",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=False,
            help="Attach an image; it will be included with your next chat message.",
            key="fitness_food_upload",
        )
        if uploaded_food_image is not None:
            uploaded_food_image_bytes = uploaded_food_image.getvalue()
            uploaded_food_image_name = uploaded_food_image.name or "uploaded-food-image"
            st.image(uploaded_food_image_bytes, caption=uploaded_food_image_name, use_container_width=True)
            upload_marker = f"{uploaded_food_image_name}:{len(uploaded_food_image_bytes)}"
            if st.session_state.get("fitness_last_upload_marker") != upload_marker:
                st.session_state["fitness_last_upload_marker"] = upload_marker
                st.session_state[chat_input_key] = "What are the macronutrients in this meal?"
        else:
            st.caption("No image attached.")

        st.divider()
        st.markdown("**Long-term memory (SQLite)**")
        try:
            if meals:
                st.caption("Last 6 meals and macros")
                for meal in meals[:6]:
                    st.markdown(_render_longterm_meal_block(meal), unsafe_allow_html=True)
            else:
                st.caption("No meal history found.")
        except Exception as exc:
            logger.exception("Could not render meal memory panel")
            st.caption(f"Could not render meals: {exc}")

        st.divider()
        st.markdown("**Short-term memory (chat)**")
        short_memory = [
            msg for msg in st.session_state.messages if msg.get("role") in {"user", "assistant"}
        ][-6:]
        if short_memory:
            for msg in short_memory:
                role = msg.get("role", "?")
                text = (msg.get("content") or "").strip().replace("\n", " ")
                st.caption(f"{role}: {text[:120]}{'...' if len(text) > 120 else ''}")
        else:
            st.caption("No short-term memory yet.")

        st.divider()
        with st.expander("Memory Debug", expanded=False):
            debug_user_id = _normalize_user_id(st.session_state.get("fitness_user_name", "roy"))
            debug_session_key = f"fitness:{debug_user_id}"
            st.caption(f"db_path={_fitness_db_path()}")
            st.caption(f"user_id={debug_user_id} | session_key={debug_session_key}")

            short_all = [msg for msg in st.session_state.messages if msg.get("role") in {"user", "assistant"}]
            user_turns = len([msg for msg in short_all if msg.get("role") == "user"])
            assistant_turns = len([msg for msg in short_all if msg.get("role") == "assistant"])
            st.caption(
                f"short_term: total_turns={len(short_all)} user_turns={user_turns} assistant_turns={assistant_turns}"
            )

            longterm_summary = {
                "profile_field_count": len(profile) if isinstance(profile, dict) else 0,
                "body_metric_count": len(recent_body_metrics) if isinstance(recent_body_metrics, list) else 0,
                "meal_count": len(meals) if isinstance(meals, list) else 0,
            }
            st.caption(
                "long_term: "
                f"profile_field_count={longterm_summary['profile_field_count']} "
                f"body_metric_count={longterm_summary['body_metric_count']} "
                f"meal_count={longterm_summary['meal_count']}"
            )

            st.markdown("**Long-term snapshot (read model)**")
            try:
                debug_read_model = {
                    "profile": profile,
                    "recent_body_metrics": recent_body_metrics,
                    "recent_meals": meals,
                }
                st.text_area(
                    "read_model_json",
                    value=json.dumps(debug_read_model, indent=2, ensure_ascii=False),
                    height=220,
                    disabled=True,
                    key="memory_debug_read_model_json",
                )
            except Exception as exc:
                st.caption(f"Could not render read model JSON: {exc}")

            st.markdown("**Operation Log**")
            debug_events = st.session_state.get("memory_debug_events", [])
            if debug_events:
                st.text_area(
                    "memory_events",
                    value="\n".join(debug_events[-120:]),
                    height=220,
                    disabled=True,
                    key="memory_debug_events_log",
                )
            else:
                st.caption("No memory operation events yet.")

with chat_col:
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if debug_enabled and message.get("role") == "assistant":
                tabs = st.tabs(["Response", "Debug Logs"])
                with tabs[0]:
                    st.markdown(message["content"])
                with tabs[1]:
                    debug_text = "\n".join(message.get("debug_logs", []))
                    st.text_area(
                        "Log output",
                        value=debug_text,
                        height=220,
                        disabled=True,
                        key=f"debug_logs_{idx}",
                    )
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask something...", key=chat_input_key)
    if prompt:
        logger.info("User prompt received. Agent=%s Endpoint=%s Model=%s", agent_choice, provider_name, model)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            started = time.perf_counter()
            status = "ok"
            usage_summary = ""
            agent_init_elapsed = 0.0
            model_call_elapsed = 0.0
            turn_debug_logs: list[str] = []

            def _add_turn_debug(line: str) -> None:
                stamped = f"[{datetime.now(ZoneInfo(_browser_timezone_name())).strftime('%H:%M:%S')}] {line}"
                turn_debug_logs.append(stamped)
                logger.info("TURN_DEBUG %s", line)

            _add_turn_debug(
                f"request_start agent={agent_choice} provider={provider_name or '-'} model={model or '-'}"
            )
            _add_turn_debug(
                f"params temperature={temperature:.2f} top_p={top_p:.2f} max_tokens={int(max_tokens)} verify_tls={verify_tls}"
            )
            try:
                if agent_choice == "General Chat Assistant":
                    if endpoint:
                        os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint.strip().strip('"').strip("'")
                        os.environ["AZURE_OPENAI_API_KEY"] = (api_key or "").strip().strip('"').strip("'")
                    if model:
                        os.environ["CHAT_MODEL"] = model
                        os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = model

                if agent_choice == "General Chat Assistant":
                    logger.info("Running General Chat Assistant agent")
                    agent_build_started = time.perf_counter()
                    agent = asyncio.run(azure_foundry_general_agent())
                    agent_build_elapsed = time.perf_counter() - agent_build_started
                    agent_init_elapsed = agent_build_elapsed
                    history_messages = [
                        ChatMessage(role=msg.get("role", "user"), text=msg.get("content", ""))
                        for msg in st.session_state.messages
                    ]
                    est_input_tokens = _estimate_tokens_from_chat_messages(history_messages)
                    _add_turn_debug(
                        f"context short_term_messages={len(history_messages)} est_input_tokens={est_input_tokens} configured_max_output_tokens={int(max_tokens)}"
                    )
                    _add_turn_debug(f"agent_init latency_s={agent_build_elapsed:.2f}")
                    llm_started = time.perf_counter()
                    result = asyncio.run(run_with_retry(agent, history_messages))
                    llm_elapsed = time.perf_counter() - llm_started
                    model_call_elapsed = llm_elapsed
                    content = _extract_display_text(getattr(result, "text", None) or str(result))
                    usage = getattr(result, "usage_details", None)
                    if usage:
                        usage_summary = (
                            f"input={usage.input_token_count or 0} "
                            f"output={usage.output_token_count or 0} "
                            f"total={usage.total_token_count or 0}"
                        )
                    usage_parsed = _parse_usage_summary(usage_summary)
                    _add_turn_debug(
                        f"llm_call latency_s={llm_elapsed:.2f} usage_input={usage_parsed['input']} usage_output={usage_parsed['output']} usage_total={usage_parsed['total']}"
                    )
                elif agent_choice == "Kaito Assistant":
                    logger.info("Running Kaito Assistant agent")
                    if endpoint:
                        os.environ["KAITO_INFERENCE_ENDPOINT"] = _clean_env(endpoint)
                    if api_key:
                        os.environ["KAITO_API_KEY"] = api_key
                    if model:
                        os.environ["KAITO_MODEL"] = model
                    _add_turn_debug(
                        f"context short_term_messages=1 est_input_tokens={_estimate_tokens_from_text(prompt)} configured_max_output_tokens={int(max_tokens)}"
                    )
                    agent = _build_kaito_agent(model)
                    llm_started = time.perf_counter()
                    result = asyncio.run(run_with_retry(agent, prompt))
                    llm_elapsed = time.perf_counter() - llm_started
                    model_call_elapsed = llm_elapsed
                    content = _extract_display_text(getattr(result, "text", None) or str(result))
                    usage = getattr(result, "usage_details", None)
                    if usage:
                        usage_summary = (
                            f"input={usage.input_token_count or 0} "
                            f"output={usage.output_token_count or 0} "
                            f"total={usage.total_token_count or 0}"
                        )
                    usage_parsed = _parse_usage_summary(usage_summary)
                    _add_turn_debug(
                        f"llm_call latency_s={llm_elapsed:.2f} usage_input={usage_parsed['input']} usage_output={usage_parsed['output']} usage_total={usage_parsed['total']}"
                    )
                elif agent_choice == "KAITO RAG Assistant":
                    logger.info("Running KAITO RAG Assistant agent")
                    if endpoint:
                        os.environ["KAITO_RAGENGINE_ENDPOINT"] = _clean_env(endpoint)
                    if api_key:
                        os.environ["KAITO_RAGENGINE_API_KEY"] = api_key
                    if model:
                        os.environ["KAITO_MODEL"] = model
                    _add_turn_debug(
                        f"context short_term_messages=1 est_input_tokens={_estimate_tokens_from_text(prompt)} configured_max_output_tokens={int(max_tokens)}"
                    )
                    agent = _build_kaito_ragengine_agent(model)
                    llm_started = time.perf_counter()
                    result = asyncio.run(run_with_retry(agent, prompt))
                    llm_elapsed = time.perf_counter() - llm_started
                    model_call_elapsed = llm_elapsed
                    content = _extract_display_text(getattr(result, "text", None) or str(result))
                    usage = getattr(result, "usage_details", None)
                    if usage:
                        usage_summary = (
                            f"input={usage.input_token_count or 0} "
                            f"output={usage.output_token_count or 0} "
                            f"total={usage.total_token_count or 0}"
                        )
                    usage_parsed = _parse_usage_summary(usage_summary)
                    _add_turn_debug(
                        f"llm_call latency_s={llm_elapsed:.2f} usage_input={usage_parsed['input']} usage_output={usage_parsed['output']} usage_total={usage_parsed['total']}"
                    )
                elif agent_choice == "Agent1 Demo":
                    logger.info("Running Agent1 Demo")
                    llm_started = time.perf_counter()
                    output = io.StringIO()
                    with redirect_stdout(output):
                        asyncio.run(agent1())
                    llm_elapsed = time.perf_counter() - llm_started
                    model_call_elapsed = llm_elapsed
                    raw_output = output.getvalue().strip()
                    content = _format_agent1_output(raw_output) or "No output from agent1."
                    _add_turn_debug(f"workflow latency_s={llm_elapsed:.2f}")
                else:
                    logger.info("Running Fitness Nutrition agent")
                    fitness_user_id = _normalize_user_id(st.session_state.get("fitness_user_name", "roy"))
                    fitness_started = time.perf_counter()
                    content, usage_summary = asyncio.run(
                        _run_fitness_turn(
                            user_prompt=prompt,
                            user_id=fitness_user_id,
                            selected_model=model,
                            image_bytes=uploaded_food_image_bytes,
                            image_name=uploaded_food_image_name,
                        )
                    )
                    fitness_elapsed = time.perf_counter() - fitness_started
                    model_call_elapsed = fitness_elapsed
                    usage_parsed = _parse_usage_summary(usage_summary)
                    _add_turn_debug(
                        f"fitness_run latency_s={fitness_elapsed:.2f} usage_input={usage_parsed['input']} usage_output={usage_parsed['output']} usage_total={usage_parsed['total']}"
                    )
                    try:
                        model_snapshot = get_fitness_repository(_fitness_db_path()).get_read_model(fitness_user_id, metric_limit=6, meal_limit=6)
                        short_count = len([m for m in st.session_state.messages if m.get("role") in {"user", "assistant"}])
                        long_profile = len(model_snapshot.get("profile", {})) if isinstance(model_snapshot.get("profile"), dict) else 0
                        long_metrics = len(model_snapshot.get("recent_body_metrics", [])) if isinstance(model_snapshot.get("recent_body_metrics"), list) else 0
                        long_meals = len(model_snapshot.get("recent_meals", [])) if isinstance(model_snapshot.get("recent_meals"), list) else 0
                        _add_turn_debug(
                            f"memory short_term_events={short_count} long_term_profile_fields={long_profile} long_term_body_metrics={long_metrics} long_term_meals={long_meals}"
                        )
                    except Exception as exc:
                        _add_turn_debug(f"memory_snapshot failed error={exc}")

                    memory_events = st.session_state.get("memory_debug_events", [])
                    if isinstance(memory_events, list):
                        _add_turn_debug(f"memory_events_count={len(memory_events)}")
                        for event_line in memory_events[-6:]:
                            _add_turn_debug(f"memory_event {event_line}")

                if usage_summary:
                    usage_parsed = _parse_usage_summary(usage_summary)
                    _add_turn_debug(
                        f"context_window usage_input={usage_parsed['input']} usage_output={usage_parsed['output']} usage_total={usage_parsed['total']} configured_max_output_tokens={int(max_tokens)}"
                    )
                else:
                    _add_turn_debug(
                        f"context_window usage_unavailable configured_max_output_tokens={int(max_tokens)} est_input_tokens={_estimate_tokens_from_text(prompt)}"
                    )

                st.markdown(content)
                debug_snapshot = turn_debug_logs if debug_enabled else []
                st.session_state.messages.append(
                    {"role": "assistant", "content": content, "debug_logs": debug_snapshot}
                )
            except urllib.error.HTTPError as exc:
                status = f"http-{exc.code}"
                error_body = exc.read().decode("utf-8") if exc.fp else ""
                logger.error("HTTP error: %s %s %s", exc.code, exc.reason, error_body)
                st.error(f"Request failed: {exc.code} {exc.reason}\n{error_body}")
            except urllib.error.URLError as exc:
                status = "url-error"
                logger.error("URL error: %s", exc.reason)
                st.error(f"Request failed: {exc.reason}")
            except Exception as exc:
                status = "error"
                logger.exception("Unhandled error: %s", exc)
                st.error(f"Request failed: {exc}")
            finally:
                elapsed_s = time.perf_counter() - started
                post_elapsed = max(0.0, elapsed_s - agent_init_elapsed - model_call_elapsed)
                if debug_enabled:
                    _add_turn_debug(f"request_end status={status} total_latency_s={elapsed_s:.2f}")
                metrics_line = (
                    f"agent={agent_choice} | endpoint={provider_name or '-'} | model={model or '-'} | "
                    f"status={status} | latency_s={elapsed_s:.2f} | "
                    f"breakdown_s=init:{agent_init_elapsed:.2f},llm:{model_call_elapsed:.2f},post:{post_elapsed:.2f}"
                )
                if usage_summary:
                    metrics_line = f"{metrics_line} | {usage_summary}"
                st.session_state.metrics_log.append(metrics_line)
                logger.info("Completion: %s", metrics_line)
                if not st.session_state._metrics_rerun:
                    st.session_state._metrics_rerun = True
                    st.rerun()

if debug_enabled:
    pass
