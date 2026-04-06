import asyncio
import copy
import hashlib
import html
import importlib
import inspect
import json
import logging
import mimetypes
import os
import re
import sys
import time
import threading
import urllib.error
import urllib.request
from urllib.parse import urlparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st
import yaml
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL_ENV = "LOG_LEVEL"
DEBUG_LOG_MAX_LINES_ENV = "DEBUG_LOG_MAX_LINES"

# Deferred import – config package is on sys.path after PROJECT_ROOT is added below.
_FITNESS_READ_MODEL_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}
_FITNESS_READ_MODEL_CACHE_LOCK = threading.Lock()


def _sensitive_values() -> list[str]:
    values: list[str] = []
    for env_name, env_value in os.environ.items():
        upper_name = env_name.upper()
        if not any(token in upper_name for token in ("ENDPOINT", "API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            continue
        cleaned = _clean_env(env_value)
        if cleaned:
            values.append(cleaned)

    providers = globals().get("PROVIDERS", [])
    if isinstance(providers, list):
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            default_endpoint = _clean_env(provider.get("default_endpoint"))
            if default_endpoint:
                values.append(default_endpoint)

    return sorted(set(values), key=len, reverse=True)


def _redact_sensitive_text(value: Any) -> str:
    text = str(value)
    for sensitive in _sensitive_values():
        text = text.replace(sensitive, "[redacted]")
    text = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+", r"\1[redacted]", text)
    return text


class _SessionLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = _redact_sensitive_text(self.format(record))
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
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config  # noqa: E402

from agent_framework import AgentRunResponse, ChatAgent, ChatMessage, DataContent, TextContent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from ai_chat_client import KaitoChatClient
from fitness_memory import (
    DatabaseContextProvider,
    PhotoSubmissionStructuredOutput,
    TextTurnStructuredOutput,
    build_fitness_context_instructions,
    extract_idempotency_key,
    get_fitness_repository,
)
from main import azure_foundry_general_agent, load_prompt_template, render_instructions
from openai import RateLimitError
from run_utils import get_backoff_seconds, run_with_retry
from diagnostics_store import (
    DiagnosticsTurn,
    PerformanceEvent,
    record_turn as _diag_record_turn,
    record_log as _diag_record_log,
    record_performance_event as _diag_record_performance,
    get_context_window_size as _diag_ctx_size,
    estimate_tokens as _diag_est_tokens,
)
from fitness_background_persistence import (
    FitnessPersistenceHooks,
    FitnessPersistenceRequest,
    schedule_fitness_persistence,
)
_CFG = get_config()
_FITNESS_AGENT_NAME = "Fitness Nutrition"


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
    """Return the chatbot section of the unified config as a plain dict.

    The returned dict has the same shape the old chatbot/config.yaml
    produced so that downstream code (PROVIDERS, AGENTS, UI_CONFIG)
    continues to work unchanged.
    """
    cfg = get_config()
    providers = [p.model_dump(by_alias=True) for p in cfg.ai.providers]
    agents = [a.model_dump(by_alias=True) for a in cfg.ai.agents]
    ui = {
        "log_level_env": "LOG_LEVEL",
        "debug_log_max_lines_env": "DEBUG_LOG_MAX_LINES",
    }
    return {"providers": providers, "agents": agents, "ui": ui}


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


def _record_perf_event(
    *,
    agent: str,
    request_id: str,
    category: str,
    name: str,
    started: float,
    status: str = "ok",
    **details: Any,
) -> None:
    try:
        _diag_record_performance(
            PerformanceEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent=agent,
                request_id=request_id,
                category=category,
                name=name,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                status=status,
                details={key: value for key, value in details.items() if value is not None},
            )
        )
    except Exception:
        logger.debug("Could not record performance event", exc_info=True)


def _format_agent1_output(raw_output: str) -> str:
    if not raw_output:
        return ""
    lines = raw_output.splitlines()
    section_lines: dict[str, list[str]] = {
        "step1": [],
        "step15": [],
        "step2": [],
        "step3": [],
        "step4": [],
    }
    current_step = None

    def _append_to_current(line_value: str) -> None:
        if current_step in section_lines:
            section_lines[current_step].append(line_value)

    for line in lines:
        line_lower = line.strip().lower()
        if line_lower.startswith("step 1 result"):
            current_step = "step1"
            inline = line.split(":", 1)[1].strip() if ":" in line else ""
            if inline:
                _append_to_current(inline)
            continue
        if line_lower.startswith("step 1.5 repo context"):
            current_step = "step15"
            inline = line.split(":", 1)[1].strip() if ":" in line else ""
            if inline:
                _append_to_current(inline)
            continue
        if line_lower.startswith("step 2 workflow"):
            current_step = "step2"
            inline = line.split(":", 1)[1].strip() if ":" in line else ""
            if inline:
                _append_to_current(inline)
            continue
        if line_lower.startswith("step 3 structured output"):
            current_step = "step3"
            inline = line.split(":", 1)[1].strip() if ":" in line else ""
            if inline:
                _append_to_current(inline)
            continue
        if line_lower.startswith("step 4 evidence block"):
            current_step = "step4"
            inline = line.split(":", 1)[1].strip() if ":" in line else ""
            if inline:
                _append_to_current(inline)
            continue
        if line_lower.startswith("tokens:") or line_lower.startswith("hello from agentframework"):
            continue

        _append_to_current(line)

    step1 = "\n".join(section_lines["step1"]).strip()
    step15 = "\n".join(section_lines["step15"]).strip()
    step2 = "\n".join(section_lines["step2"]).strip()
    step3 = "\n".join(section_lines["step3"]).strip()
    step4 = "\n".join(section_lines["step4"]).strip()

    if not (step1 or step15 or step2 or step3 or step4):
        return raw_output

    def _pretty_json(text: str) -> str:
        text = text.strip()
        if not text:
            return text
        try:
            parsed = json.loads(text)
            if isinstance(parsed, str):
                nested = parsed.strip()
                if nested.startswith("{") or nested.startswith("["):
                    try:
                        parsed = json.loads(nested)
                    except json.JSONDecodeError:
                        return parsed
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return text

    def _load_json_like(text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, str):
                nested = parsed.strip()
                if nested.startswith("{") or nested.startswith("["):
                    parsed = json.loads(nested)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _truncate(value: str, max_len: int = 120) -> str:
        compact = " ".join(value.split())
        return compact if len(compact) <= max_len else f"{compact[:max_len - 1]}…"

    def _format_step4_visual(step4_raw: str) -> str:
        payload = _load_json_like(_extract_display_text(step4_raw))
        if not isinstance(payload, dict):
            fallback = _pretty_json(_extract_display_text(step4_raw))
            return "\n".join(f"    {line}" for line in fallback.splitlines())

        grounding_path = str(payload.get("grounding_path") or "n/a")
        error = payload.get("error")
        matched_lines = payload.get("matched_lines") if isinstance(payload.get("matched_lines"), list) else []
        clean_lines = [str(line) for line in matched_lines if str(line).strip()]
        lines_to_show = clean_lines[:3]
        status_badge = "🟩 Evidence OK" if not error else "🟥 Evidence Error"

        parts: list[str] = [
            "### 🧾 Step 4 Evidence Summary",
            f"- {status_badge}",
            f"- 📄 **File:** `{grounding_path}`",
            f"- 🔎 **Matches:** {len(clean_lines)}",
        ]
        if error:
            parts.append(f"- ⚠️ **Error:** {_truncate(str(error), 140)}")

        if lines_to_show:
            parts.append("- ✨ **Top matched lines:**")
            for idx, line in enumerate(lines_to_show, start=1):
                parts.append(f"  - {idx}. {_truncate(line, 140)}")
        else:
            parts.append("- ℹ️ No matched evidence lines captured.")

        return "\n".join(parts)

    parts = []
    if step1:
        parts.append(f"**Step 1:**\n\n    {step1}")
    if step15:
        step15_text = _pretty_json(_extract_display_text(step15))
        parts.append("**Step 1.5:**\n\n" + "\n".join(f"    {line}" for line in step15_text.splitlines()))
    if step2:
        step2_text = _pretty_json(_extract_display_text(step2))
        parts.append("**Step 2:**\n\n" + "\n".join(f"    {line}" for line in step2_text.splitlines()))
    if step3:
        step3_text = _pretty_json(_extract_display_text(step3))
        parts.append("**Step 3:**\n\n" + "\n".join(f"    {line}" for line in step3_text.splitlines()))
    if step4:
        parts.append("**Step 4:**\n\n" + _format_step4_visual(step4))
    return "\n\n".join(parts)


def _agent1_trace_state() -> list[dict[str, Any]]:
    return st.session_state.setdefault("agent1_mcp_trace", [])


def _append_agent1_trace(direction: str, method: str, payload: dict[str, Any] | None = None) -> None:
    events = _agent1_trace_state()
    events.append(
        {
            "ts": datetime.now(ZoneInfo(_browser_timezone_name())).strftime("%m/%d/%Y"),
            "direction": direction,
            "method": method,
            "payload": payload or {},
        }
    )
    _perf_max = get_config().ui.performance.max_events
    if len(events) > _perf_max:
        del events[:-_perf_max]


def _render_agent1_trace(placeholder: Any | None = None) -> None:
    events = _agent1_trace_state()
    visible_events = events[-20:]

    def _event_kind(method: str) -> str:
        lowered = method.lower()
        if "initialize" in lowered:
            return "initialize"
        if "tools/call" in lowered:
            return "tools_call"
        if "tools/result" in lowered:
            return "tools_result"
        if "error" in lowered:
            return "error"
        if "complete" in lowered:
            return "complete"
        return "other"

    kind_badge = {
        "initialize": "🟦 INIT",
        "tools_call": "🟪 CALL",
        "tools_result": "🟩 RESULT",
        "error": "🟥 ERROR",
        "complete": "🟨 DONE",
        "other": "⬜ EVENT",
    }

    event_lines: list[str] = []
    sequence_lines: list[str] = []
    for event in visible_events:
        direction = event.get("direction") or "client->server"
        arrow = "➡️" if direction == "client->server" else "⬅️"
        method = event.get("method") or "unknown"
        ts = event.get("ts") or "--:--:--"
        kind = _event_kind(method)
        event_lines.append(f"- {ts} {kind_badge[kind]} {arrow} **{method}**")

        if direction == "client->server":
            sequence_lines.append(f"Client -> Server: {method}")
        else:
            sequence_lines.append(f"Server --> Client: {method}")

        payload = event.get("payload")
        if isinstance(payload, dict) and payload:
            event_lines.append("```json")
            event_lines.append(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
            event_lines.append("```")

    if not visible_events:
        markdown = "_No MCP trace events yet._"
    else:
        markdown = (
            "**Sequence View**\n\n"
            "```text\n"
            + "\n".join(sequence_lines)
            + "\n```\n\n"
            "**Event Log**\n\n"
            + "\n".join(event_lines)
        )

    target = placeholder if placeholder is not None else st
    target.markdown(markdown)


class _Agent1LiveCapture:
    def __init__(self, trace_placeholder: Any | None = None) -> None:
        self._trace_placeholder = trace_placeholder
        self._raw_chunks: list[str] = []
        self._line_buffer = ""
        self._json_label: str | None = None
        self._json_lines: list[str] = []
        self._brace_balance = 0

    def write(self, data: str) -> int:
        self._raw_chunks.append(data)
        self._line_buffer += data
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._handle_line(line)
        return len(data)

    def flush(self) -> None:
        return

    def getvalue(self) -> str:
        return "".join(self._raw_chunks)

    def _refresh(self) -> None:
        _render_agent1_trace(self._trace_placeholder)

    def _try_parse_json_block(self, label: str, text: str) -> None:
        try:
            payload = json.loads(text)
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        if label == "step15":
            query = payload.get("query")
            selected_path = payload.get("selected_path")
            candidates = payload.get("candidate_paths") or []
            _append_agent1_trace("client->server", "tools/call search_repo_paths", {"query": query})
            _append_agent1_trace(
                "server->client",
                "tools/result search_repo_paths",
                {"count": len(candidates), "selected_path": selected_path},
            )
            if selected_path:
                _append_agent1_trace(
                    "client->server",
                    "tools/call read_repo_file",
                    {"path": selected_path, "start_line": 1, "end_line": 80},
                )
                _append_agent1_trace(
                    "server->client",
                    "tools/result read_repo_file",
                    {"path": selected_path, "excerpt_available": True},
                )
            self._refresh()
        elif label == "step4":
            _append_agent1_trace("server->client", "trace/evidence", payload)
            self._refresh()

    def _handle_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return

        lowered = stripped.lower()
        if lowered.startswith("step 1 result"):
            _append_agent1_trace("client->server", "chat/request step1", {})
            self._refresh()
            return
        if lowered.startswith("step 2 workflow"):
            _append_agent1_trace("server->client", "chat/response workflow", {})
            self._refresh()
            return
        if lowered.startswith("step 3 structured output"):
            _append_agent1_trace("server->client", "chat/response structured_output", {})
            self._refresh()
            return
        if lowered.startswith("step 1.5 repo context"):
            self._json_label = "step15"
            self._json_lines = []
            self._brace_balance = 0
            self._refresh()
            return
        if lowered.startswith("step 4 evidence block"):
            self._json_label = "step4"
            self._json_lines = []
            self._brace_balance = 0
            self._refresh()
            return

        if self._json_label:
            if not self._json_lines and "{" not in stripped:
                return
            self._json_lines.append(line)
            self._brace_balance += line.count("{") - line.count("}")
            if self._brace_balance <= 0 and self._json_lines:
                label = self._json_label
                text = "\n".join(self._json_lines)
                self._json_label = None
                self._json_lines = []
                self._brace_balance = 0
                self._try_parse_json_block(label, text)



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


def _build_fitness_chat_client(backend_cfg: dict, providers: list[dict]):
    """Return (chat_client, model_id) for a Fitness Nutrition backend config entry."""
    provider_name = backend_cfg.get("provider", "AI Foundry")
    display_model = backend_cfg.get("model", "")
    provider_cfg = next((p for p in providers if p.get("name") == provider_name), {})
    if not display_model:
        display_model = provider_cfg.get("default_model", "")
    model_id = provider_cfg.get("request_model") or display_model

    endpoint_env = provider_cfg.get("endpoint_env")
    endpoint_default = provider_cfg.get("default_endpoint", "")
    resolved_endpoint = _clean_env(os.getenv(endpoint_env, endpoint_default)) if endpoint_env else endpoint_default

    api_key_env = provider_cfg.get("api_key_env")
    api_key = _clean_env(os.getenv(api_key_env)) if api_key_env else ""

    if provider_name == "AI Foundry":
        if resolved_endpoint:
            os.environ["AZURE_OPENAI_ENDPOINT"] = resolved_endpoint
        if api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
        return AzureOpenAIChatClient(credential=AzureCliCredential()), model_id
    return KaitoChatClient(
        endpoint=resolved_endpoint,
        api_key=api_key or None,
        default_model=model_id,
    ), model_id


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


def _heuristic_text_turn_payload(user_text: str) -> tuple[TextTurnStructuredOutput, dict] | None:
    text = (user_text or "").strip()
    if not text:
        return None

    profile_updates = []
    body_metric_events = []

    def _add_profile(field: str, value: object) -> None:
        if value is None:
            return
        profile_updates.append({"field": field, "value": value})

    birthday_match = re.search(
        r"\b(?:birthday|dob|date of birth)\s*(?:is|=|:|to)?\s*(\d{1,2}/\d{1,2}/\d{4}|\d{8}|\d{4}-\d{2}-\d{2})\b",
        text,
        re.IGNORECASE,
    )
    if birthday_match:
        _add_profile("birthday_mmddyyyy", birthday_match.group(1))

    name_match = re.search(r"\b(?:my name is|name\s*(?:is|=|:))\s*([A-Za-z][A-Za-z .'-]{0,40})", text, re.IGNORECASE)
    if name_match:
        _add_profile("name", name_match.group(1).strip())

    sex_match = re.search(r"\b(?:sex|gender)\s*(?:is|=|:)?\s*(male|female|other)\b", text, re.IGNORECASE)
    if sex_match:
        _add_profile("sex", sex_match.group(1).lower())

    city_match = re.search(r"\bcity\s*(?:is|=|:)?\s*([A-Za-z][A-Za-z .'-]{0,40})", text, re.IGNORECASE)
    if city_match:
        _add_profile("city", city_match.group(1).strip())

    country_match = re.search(r"\bcountry\s*(?:is|=|:)?\s*([A-Za-z][A-Za-z .'-]{0,40})", text, re.IGNORECASE)
    if country_match:
        _add_profile("country", country_match.group(1).strip())

    timezone_match = re.search(r"\btimezone\s*(?:is|=|:)?\s*([A-Za-z_]+/[A-Za-z_]+|[A-Za-z_+-]+)\b", text, re.IGNORECASE)
    if timezone_match:
        _add_profile("timezone", timezone_match.group(1).strip())

    height_ft_match = re.search(
        r"\b(?:height)\s*(?:is|=|:|to)?\s*(\d{1,2})\s*'\s*(\d{1,2})(?:\s*\"|\s+in(?:ches?)?)?",
        text,
        re.IGNORECASE,
    )
    if height_ft_match:
        feet = int(height_ft_match.group(1))
        inches = int(height_ft_match.group(2))
        _add_profile("height_value", float(feet * 12 + inches))
        _add_profile("height_unit", "in")
    else:
        height_match = re.search(
            r"\b(?:height)\s*(?:is|=|:|to)?\s*(\d+(?:\.\d+)?)\s*(in|inch|inches|cm)\b",
            text,
            re.IGNORECASE,
        )
        if height_match:
            _add_profile("height_value", float(height_match.group(1)))
            _add_profile("height_unit", height_match.group(2).lower())

    weight_match = re.search(
        r"\b(?:my\s+)?weight\s*(?:is|=|:|to)?\s*(\d+(?:\.\d+)?)\s*(lb|lbs|kg)\b",
        text,
        re.IGNORECASE,
    )
    if weight_match:
        unit = weight_match.group(2).lower()
        body_metric_events.append(
            {
                "metric_type": "weight",
                "value_primary": float(weight_match.group(1)),
                "value_secondary": None,
                "unit": "lbs" if unit in {"lb", "lbs"} else "kg",
                "observed_at": None,
                "source": "manual",
                "confidence": 1.0,
                "notes": "heuristic-text-fallback",
            }
        )

    waist_match = re.search(
        r"\b(?:my\s+)?waist\s*(?:is|=|:|to)?\s*(\d+(?:\.\d+)?)\s*(in|inch|inches|cm)\b",
        text,
        re.IGNORECASE,
    )
    if waist_match:
        unit = waist_match.group(2).lower()
        body_metric_events.append(
            {
                "metric_type": "waist",
                "value_primary": float(waist_match.group(1)),
                "value_secondary": None,
                "unit": "in" if unit in {"in", "inch", "inches"} else "cm",
                "observed_at": None,
                "source": "manual",
                "confidence": 1.0,
                "notes": "heuristic-text-fallback",
            }
        )

    bp_match = re.search(
        r"\b(?:blood pressure|bp)\s*(?:is|=|:|to)?\s*(\d+(?:\.\d+)?)\s*(?:/|over)\s*(\d+(?:\.\d+)?)\s*(mmhg)?\b",
        text,
        re.IGNORECASE,
    )
    if bp_match:
        body_metric_events.append(
            {
                "metric_type": "blood_pressure",
                "value_primary": float(bp_match.group(1)),
                "value_secondary": float(bp_match.group(2)),
                "unit": "mmHg",
                "observed_at": None,
                "source": "manual",
                "confidence": 1.0,
                "notes": "heuristic-text-fallback",
            }
        )

    if not profile_updates and not body_metric_events:
        return None

    payload_dict = {
        "profile_updates": profile_updates,
        "body_metric_events_insert": body_metric_events,
        "persistence_ops": [],
    }
    return TextTurnStructuredOutput.model_validate(payload_dict), payload_dict


async def _persist_text_turn_memory(
    *,
    agent: ChatAgent,
    repo: object,
    user_id: str,
    user_text: str,
    assistant_text: str,
    request_id: str = "",
) -> bool:
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
        extraction_started = time.perf_counter()
        result = await run_with_retry(
            agent,
            extraction_prompt,
            response_format=TextTurnStructuredOutput,
        )
        _record_perf_event(
            agent=_FITNESS_AGENT_NAME,
            request_id=request_id,
            category="llm",
            name="extract_text_turn_memory",
            started=extraction_started,
            output_tokens=getattr(getattr(result, "usage_details", None), "output_token_count", None),
        )
        payload, raw_output = _coerce_text_turn_payload(result)
        if not payload.profile_updates and not payload.body_metric_events_insert:
            heuristic_payload = _heuristic_text_turn_payload(user_text)
            if heuristic_payload is not None:
                payload, raw_output = heuristic_payload
                _append_memory_debug_event("text_persist", "used heuristic fallback for explicit profile/body metric facts")
                _diag_record_log(_FITNESS_AGENT_NAME, f"Heuristic text-turn persistence fallback used for {user_id}", "INFO")
            else:
                _append_memory_debug_event("text_persist", "no explicit profile/body metric facts extracted")
                return False

        idempotency_key = hashlib.sha256(
            f"{user_id}:{user_text}:{assistant_text}".encode("utf-8")
        ).hexdigest()
        persist_started = time.perf_counter()
        persist_result = repo.apply_text_turn_submission(
            user_id=user_id,
            payload=payload,
            raw_structured_output=raw_output,
            idempotency_key=idempotency_key,
        )
        _record_perf_event(
            agent=_FITNESS_AGENT_NAME,
            request_id=request_id,
            category="db",
            name="apply_text_turn_submission",
            started=persist_started,
            profile_updates=len(payload.profile_updates),
            body_metric_events=len(payload.body_metric_events_insert),
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
        return True
    except Exception as exc:
        heuristic_payload = _heuristic_text_turn_payload(user_text)
        if heuristic_payload is not None:
            payload, raw_output = heuristic_payload
            idempotency_key = hashlib.sha256(
                f"{user_id}:{user_text}:{assistant_text}:heuristic".encode("utf-8")
            ).hexdigest()
            persist_started = time.perf_counter()
            persist_result = repo.apply_text_turn_submission(
                user_id=user_id,
                payload=payload,
                raw_structured_output=raw_output,
                idempotency_key=idempotency_key,
            )
            _record_perf_event(
                agent=_FITNESS_AGENT_NAME,
                request_id=request_id,
                category="db",
                name="apply_text_turn_submission_heuristic_fallback",
                started=persist_started,
                profile_updates=len(payload.profile_updates),
                body_metric_events=len(payload.body_metric_events_insert),
            )
            logger.info("Text-turn memory persisted via heuristic fallback: %s", persist_result)
            _append_memory_debug_event("text_persist", "LLM extraction failed; used heuristic fallback")
            _diag_record_log(_FITNESS_AGENT_NAME, f"Heuristic fallback persisted text-turn memory for {user_id}", "WARNING")
            return True
        logger.warning("Could not persist text-turn memory: %s", exc)
        _append_memory_debug_event("text_persist", f"failed: {exc}")
        return False


async def _resolve_maybe_awaitable(value: object) -> object:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _run_with_streaming_placeholder(
    agent: ChatAgent,
    messages: object,
    *,
    placeholder: Any | None,
    max_retries: int = 5,
    **kwargs: Any,
) -> AgentRunResponse:
    for attempt in range(1, max_retries + 1):
        streamed_text = ""
        response_updates = []
        try:
            async for update in agent.run_stream(messages, **kwargs):
                response_updates.append(update)
                if update.text:
                    streamed_text = f"{streamed_text}{update.text}"
                    if placeholder is not None:
                        placeholder.markdown(streamed_text)

            response = AgentRunResponse.from_agent_run_response_updates(response_updates)
            final_text = _extract_display_text(getattr(response, "text", None) or streamed_text or str(response))
            if placeholder is not None and final_text != streamed_text:
                placeholder.markdown(final_text)
            return response
        except RateLimitError:
            if attempt == max_retries:
                raise
            if placeholder is not None:
                placeholder.empty()
            await asyncio.sleep(get_backoff_seconds(attempt))


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
        if isinstance(value, datetime):
            dt = value
        else:
            normalized = str(value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        tz = ZoneInfo(_browser_timezone_name())
        return dt.astimezone(tz).strftime("%m/%d/%Y")
    except Exception:
        return str(value)


def _short_local_date(value: str | None) -> str:
    if not value:
        return ""
    try:
        if isinstance(value, datetime):
            dt = value
        else:
            normalized = str(value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        tz = ZoneInfo(_browser_timezone_name())
        return dt.astimezone(tz).strftime("%m/%d/%Y")
    except Exception:
        parsed_birthday = _parse_birthday_mmddyyyy(value)
        if parsed_birthday is not None:
            return parsed_birthday.strftime("%m/%d/%Y")
        return str(value)


def _format_height_display(value: object, unit: object) -> str:
    if value is None:
        return "n/a"
    text_value = str(value).strip()
    text_unit = str(unit).strip().lower() if unit is not None else ""
    if not text_value:
        return "n/a"
    try:
        numeric_value = float(text_value)
    except (TypeError, ValueError):
        numeric_value = None

    if numeric_value is not None and text_unit in {"in", "inch", "inches"}:
        total_inches = int(round(numeric_value))
        feet = total_inches // 12
        inches = total_inches % 12
        return f"{feet}' {inches}\""

    if text_unit:
        return f"{text_value} {text_unit}"
    return text_value


def _format_metric_value(value: object, *, decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else "n/a"


def _resolve_memory_user_id(user_name: str) -> tuple[str, bool]:
    normalized = _normalize_user_id(user_name)
    try:
        started = time.perf_counter()
        repo = get_fitness_repository(_fitness_db_path())
        resolved_user_id, resolved_from_name = repo.resolve_user_id(normalized)
        _record_perf_event(
            agent=_FITNESS_AGENT_NAME,
            request_id="sidebar",
            category="db",
            name="resolve_user_id",
            started=started,
            resolved_from_name=resolved_from_name,
        )
        return resolved_user_id, resolved_from_name
    except Exception:
        logger.exception("Could not resolve memory user id")

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
        stamp = datetime.now(ZoneInfo(_browser_timezone_name())).strftime("%m/%d/%Y")
        logs.append(f"[{stamp}] {event}: {details}")
        if len(logs) > 400:
            del logs[:-400]
    except Exception:
        pass


def _sync_fitness_session_user(resolved_user_id: str) -> None:
    previous_user_id = str(st.session_state.get("fitness_active_user_id") or "").strip()
    current_user_id = str(resolved_user_id or "").strip()
    if not current_user_id:
        return
    if not previous_user_id:
        st.session_state["fitness_active_user_id"] = current_user_id
        return
    if previous_user_id == current_user_id:
        return

    st.session_state["fitness_active_user_id"] = current_user_id
    st.session_state.messages = []
    st.session_state["memory_debug_events"] = []
    st.session_state.pop("fitness_last_upload_marker", None)
    st.session_state.pop("chat_prompt_input", None)
    _invalidate_fitness_snapshot_cache(previous_user_id)
    _invalidate_fitness_snapshot_cache(current_user_id)
    st.rerun()


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


def _agent1_prompts_dir() -> Path:
    return PROJECT_ROOT / "prompts"


def _agent1_list_prompt_templates() -> list[str]:
    prompts_dir = _agent1_prompts_dir()
    prompts_dir.mkdir(parents=True, exist_ok=True)
    return sorted(path.name for path in prompts_dir.glob("*.yaml") if path.is_file())


def _agent1_normalize_template_name(raw_name: str) -> str:
    name = (raw_name or "").strip()
    if not name:
        raise ValueError("Template name is required.")
    if any(ch in name for ch in ["/", "\\"]):
        raise ValueError("Template name must not contain path separators.")
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    return name


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


def _is_small_context_model(selected_model: str | None) -> bool:
    return "phi" in (selected_model or "").lower()


def _supports_image_inputs(selected_model: str | None) -> bool:
    model_lower = (selected_model or "").lower()
    if "phi" in model_lower:
        return "vision" in model_lower
    return True


def _fitness_cache_namespace() -> tuple[str, ...]:
    backend = os.getenv("FITNESS_DB_BACKEND", "sqlite").strip().lower()
    if backend in {"azuresql", "azure_sql", "azure-sql"}:
        return (
            "azuresql",
            _clean_env(os.getenv("AZURE_SQL_SERVER")),
            _clean_env(os.getenv("AZURE_SQL_DATABASE")),
            _clean_env(os.getenv("AZURE_SQL_SCHEMA", "dbo")),
        )
    return ("sqlite", str(_fitness_db_path().resolve()))


def _fitness_snapshot_cache_key(user_id: str, metric_limit: int, meal_limit: int) -> tuple[str, ...]:
    return (*_fitness_cache_namespace(), user_id, str(metric_limit), str(meal_limit))


def _invalidate_fitness_snapshot_cache(user_id: str | None = None) -> None:
    with _FITNESS_READ_MODEL_CACHE_LOCK:
        keys_to_remove = [
            key for key in _FITNESS_READ_MODEL_CACHE
            if user_id is None or (len(key) >= 3 and key[-3] == user_id)
        ]
        for key in keys_to_remove:
            _FITNESS_READ_MODEL_CACHE.pop(key, None)


def _load_fitness_snapshot(
    user_id: str,
    *,
    metric_limit: int = 6,
    meal_limit: int = 6,
    force_refresh: bool = False,
    request_id: str = "",
) -> dict[str, Any]:
    cache_key = _fitness_snapshot_cache_key(user_id, metric_limit, meal_limit)
    if not force_refresh:
        with _FITNESS_READ_MODEL_CACHE_LOCK:
            cached = _FITNESS_READ_MODEL_CACHE.get(cache_key)
        if cached is not None:
            cached_profile = cached.get("profile", {}) if isinstance(cached, dict) else {}
            cached_profile_user_id = ""
            if isinstance(cached_profile, dict):
                cached_profile_user_id = str(cached_profile.get("user_id") or "").strip()
            if cached_profile_user_id and cached_profile_user_id != user_id:
                logger.warning(
                    "Discarding mismatched cached fitness snapshot for requested user_id=%s because profile.user_id=%s",
                    user_id,
                    cached_profile_user_id,
                )
                with _FITNESS_READ_MODEL_CACHE_LOCK:
                    _FITNESS_READ_MODEL_CACHE.pop(cache_key, None)
            else:
                snapshot_copy = copy.deepcopy(cached)
                _diag_record_performance(
                    PerformanceEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        agent=_FITNESS_AGENT_NAME,
                        request_id=request_id,
                        category="cache",
                        name="fitness_snapshot_cache_hit",
                        duration_ms=0.0,
                        status="ok",
                        details={"metric_limit": metric_limit, "meal_limit": meal_limit},
                    )
                )
                return snapshot_copy

    repo = get_fitness_repository(_fitness_db_path())
    started = time.perf_counter()
    read_model = repo.get_read_model(user_id, metric_limit=metric_limit, meal_limit=meal_limit)
    _record_perf_event(
        agent=_FITNESS_AGENT_NAME,
        request_id=request_id,
        category="db",
        name="get_read_model",
        started=started,
        metric_limit=metric_limit,
        meal_limit=meal_limit,
        force_refresh=force_refresh,
    )
    snapshot = {
        "profile": read_model.get("profile", {}) or {},
        "recent_body_metrics": read_model.get("recent_body_metrics", []) or [],
        "recent_meals": read_model.get("recent_meals", []) or [],
    }
    profile_user_id = ""
    if isinstance(snapshot.get("profile"), dict):
        profile_user_id = str(snapshot["profile"].get("user_id") or "").strip()
    if profile_user_id and profile_user_id != user_id:
        logger.warning(
            "Fitness read model mismatch for requested user_id=%s because profile.user_id=%s",
            user_id,
            profile_user_id,
        )
    snapshot["context_instructions"] = build_fitness_context_instructions(
        snapshot,
        default_context_prompt=getattr(DatabaseContextProvider, "DEFAULT_CONTEXT_PROMPT", ""),
    )
    with _FITNESS_READ_MODEL_CACHE_LOCK:
        _FITNESS_READ_MODEL_CACHE[cache_key] = copy.deepcopy(snapshot)
    return copy.deepcopy(snapshot)


async def _persist_photo_turn_memory(
    *,
    agent: ChatAgent,
    repo: object,
    user_id: str,
    image_bytes: bytes,
    image_name: str | None,
    request_id: str = "",
) -> bool:
    mime_type, _ = mimetypes.guess_type(image_name or "")
    mime_type = mime_type or "application/octet-stream"
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
        extraction_started = time.perf_counter()
        extraction_result = await run_with_retry(
            agent,
            extraction_message,
            response_format=PhotoSubmissionStructuredOutput,
        )
        _record_perf_event(
            agent=_FITNESS_AGENT_NAME,
            request_id=request_id,
            category="llm",
            name="extract_photo_memory",
            started=extraction_started,
            image_name=image_name,
            output_tokens=getattr(getattr(extraction_result, "usage_details", None), "output_token_count", None),
        )
        payload, raw_output = _coerce_photo_payload(extraction_result)
        raw_output = _ensure_macro_events_in_output(raw_output)
        payload = PhotoSubmissionStructuredOutput.model_validate(raw_output)
        file_hint = image_name or "uploaded-image"
        idempotency_key = extract_idempotency_key(payload, image_bytes, user_id)
        persist_started = time.perf_counter()
        repo.apply_photo_submission(
            user_id=user_id,
            image_path=file_hint,
            payload=payload,
            raw_structured_output=raw_output,
            idempotency_key=idempotency_key,
        )
        _record_perf_event(
            agent=_FITNESS_AGENT_NAME,
            request_id=request_id,
            category="db",
            name="apply_photo_submission",
            started=persist_started,
            image_name=file_hint,
            profile_updates=len(payload.profile_updates),
            body_metric_events=len(payload.body_metric_events_insert),
        )
        logger.info(
            "Photo memory persisted image=%s profile_updates=%s body_metrics=%s",
            file_hint,
            len(payload.profile_updates),
            len(payload.body_metric_events_insert),
        )
        return True
    except Exception:
        logger.exception("Could not persist photo extraction memory")
        return False


async def _persist_fitness_turn_background_async(request: FitnessPersistenceRequest) -> None:
    repo = get_fitness_repository(_fitness_db_path())
    fitness_chat_client = None
    fitness_model = request.selected_model
    if request.backend_cfg:
        fitness_chat_client, fitness_model = _build_fitness_chat_client(request.backend_cfg, PROVIDERS)

    small_context = _is_small_context_model(fitness_model)
    metric_limit = 2 if small_context else 5
    meal_limit = 2 if small_context else 6
    snapshot = _load_fitness_snapshot(request.user_id, metric_limit=metric_limit, meal_limit=meal_limit)
    agent, _, _, _ = _build_fitness_runtime(
        request.user_id,
        fitness_model,
        chat_client=fitness_chat_client,
        repo=repo,
        cached_snapshot=snapshot,
    )

    cache_dirty = False
    if request.image_bytes is not None and not small_context:
        cache_dirty = await _persist_photo_turn_memory(
            agent=agent,
            repo=repo,
            user_id=request.user_id,
            image_bytes=request.image_bytes,
            image_name=request.image_name,
            request_id=request.request_id,
        ) or cache_dirty

    if not small_context:
        cache_dirty = await _persist_text_turn_memory(
            agent=agent,
            repo=repo,
            user_id=request.user_id,
            user_text=request.user_prompt,
            assistant_text=request.assistant_text,
            request_id=request.request_id,
        ) or cache_dirty

    if request.serialized_thread is not None:
        thread_save_started = time.perf_counter()
        repo.upsert_thread_state(
            user_id=request.user_id,
            session_key=request.session_key,
            agent_name=request.agent_name,
            session_state=request.serialized_thread,
            summary_text=request.user_prompt,
        )
        _record_perf_event(
            agent=_FITNESS_AGENT_NAME,
            request_id=request.request_id,
            category="db",
            name="upsert_thread_state",
            started=thread_save_started,
        )

    if cache_dirty:
        _invalidate_fitness_snapshot_cache(request.user_id)


def _build_fitness_runtime(
    user_id: str,
    selected_model: str | None,
    chat_client=None,
    repo: object | None = None,
    cached_snapshot: dict[str, Any] | None = None,
) -> tuple[ChatAgent, object, str, str]:
    repo = repo or get_fitness_repository(_fitness_db_path())
    session_key = f"fitness:{user_id}"
    agent_name = "fitness_agent"
    instructions = (
        "You are a fitness nutrition assistant with access to user profile, body metrics, and meal macro history. "
        "When users ask about goals or trends, use tracked data first and ask clarifying questions if missing data."
    )
    if chat_client is None:
        chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    # Small-context models (phi family) need tighter limits to stay under their token ceiling.
    _small_context = _is_small_context_model(selected_model)
    meal_limit = 2 if _small_context else 6
    metric_limit = 2 if _small_context else 5
    snapshot = cached_snapshot or _load_fitness_snapshot(user_id, metric_limit=metric_limit, meal_limit=meal_limit)
    context_provider = DatabaseContextProvider(
        repo,
        user_id=user_id,
        meal_limit=meal_limit,
        metric_limit=metric_limit,
        read_model=snapshot,
        context_instructions=snapshot.get("context_instructions"),
    )
    agent = ChatAgent(
        chat_client=chat_client,
        instructions=instructions,
        name=agent_name,
        model=_fitness_chat_model(selected_model),
        context_providers=[context_provider],
        tools=[],
        max_completion_tokens=800,
        temperature=get_config().ai.defaults.extraction_temperature,
    )
    return agent, repo, session_key, agent_name


async def _run_fitness_turn(
    *,
    user_prompt: str,
    user_id: str,
    selected_model: str | None,
    image_bytes: bytes | None,
    image_name: str | None,
    assistant_placeholder: Any | None = None,
    request_id: str = "",
    chat_client=None,
    backend_cfg: dict[str, Any] | None = None,
) -> tuple[str, str]:
    small_context = _is_small_context_model(selected_model)
    metric_limit = 2 if small_context else 5
    meal_limit = 2 if small_context else 6
    snapshot = _load_fitness_snapshot(user_id, metric_limit=metric_limit, meal_limit=meal_limit, request_id=request_id)
    agent, repo, session_key, agent_name = _build_fitness_runtime(
        user_id,
        selected_model,
        chat_client=chat_client,
        cached_snapshot=snapshot,
    )
    _append_memory_debug_event(
        "run_start",
        f"user_id={user_id} session_key={session_key} model={_fitness_chat_model(selected_model)}",
    )
    # Small-context models start a fresh thread each turn to avoid accumulated history
    # overflowing their token limit. Durable fitness memory still comes via DatabaseContextProvider.
    thread_load_started = time.perf_counter()
    saved_state = None if small_context else repo.load_thread_state(user_id=user_id, session_key=session_key, agent_name=agent_name)
    if not small_context:
        _record_perf_event(
            agent=_FITNESS_AGENT_NAME,
            request_id=request_id,
            category="db",
            name="load_thread_state",
            started=thread_load_started,
            restored=bool(saved_state),
        )
    if saved_state:
        try:
            deserialize_started = time.perf_counter()
            thread = await _resolve_maybe_awaitable(agent.deserialize_thread(saved_state))
            _record_perf_event(
                agent=_FITNESS_AGENT_NAME,
                request_id=request_id,
                category="thread",
                name="deserialize_thread",
                started=deserialize_started,
            )
            _append_memory_debug_event("thread_restore", "restored prior thread state")
        except Exception:
            new_thread_started = time.perf_counter()
            thread = await _resolve_maybe_awaitable(agent.get_new_thread())
            _record_perf_event(
                agent=_FITNESS_AGENT_NAME,
                request_id=request_id,
                category="thread",
                name="get_new_thread_after_restore_failure",
                started=new_thread_started,
                status="error",
            )
            _append_memory_debug_event("thread_restore", "restore failed, created new thread")
    else:
        new_thread_started = time.perf_counter()
        thread = await _resolve_maybe_awaitable(agent.get_new_thread())
        _record_perf_event(
            agent=_FITNESS_AGENT_NAME,
            request_id=request_id,
            category="thread",
            name="get_new_thread",
            started=new_thread_started,
        )
        _append_memory_debug_event("thread_restore", "no prior state, created new thread")

    usage_summary = ""
    if image_bytes is not None and not _supports_image_inputs(selected_model):
        return (
            f"⚠️ The selected model (**{_fitness_chat_model(selected_model)}**) does not support image inputs. "
            "Please switch to an AI Foundry model (e.g. gpt-5.2-chat) to analyse meal photos.",
            "",
        )
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
        llm_started = time.perf_counter()
        result = await _run_with_streaming_placeholder(
            agent,
            request_message,
            thread=thread,
            placeholder=assistant_placeholder,
        )
        _record_perf_event(
            agent=_FITNESS_AGENT_NAME,
            request_id=request_id,
            category="llm",
            name="fitness_turn_with_image",
            started=llm_started,
            model=_fitness_chat_model(selected_model),
        )
        content = _extract_display_text(getattr(result, "text", None) or str(result))
        usage = getattr(result, "usage_details", None)
        if usage:
            usage_summary = (
                f"input={usage.input_token_count or 0} "
                f"output={usage.output_token_count or 0} "
                f"total={usage.total_token_count or 0}"
            )
        if small_context:
            _append_memory_debug_event("photo_persist", "skipped: model does not support structured output")
            _append_memory_debug_event("text_persist", "skipped: model does not support structured output")
    else:
        llm_started = time.perf_counter()
        result = await _run_with_streaming_placeholder(
            agent,
            user_prompt,
            thread=thread,
            placeholder=assistant_placeholder,
        )
        _record_perf_event(
            agent=_FITNESS_AGENT_NAME,
            request_id=request_id,
            category="llm",
            name="fitness_turn_text_only",
            started=llm_started,
            model=_fitness_chat_model(selected_model),
        )
        content = _extract_display_text(getattr(result, "text", None) or str(result))
        usage = getattr(result, "usage_details", None)
        if usage:
            usage_summary = (
                f"input={usage.input_token_count or 0} "
                f"output={usage.output_token_count or 0} "
                f"total={usage.total_token_count or 0}"
            )

        if small_context:
            _append_memory_debug_event("text_persist", "skipped: model does not support structured output")

    serialized_thread: dict[str, Any] | None = None
    try:
        resolved_thread = await _resolve_maybe_awaitable(thread)
        thread_serialize_started = time.perf_counter()
        serialized_thread = await resolved_thread.serialize()
        _record_perf_event(
            agent=_FITNESS_AGENT_NAME,
            request_id=request_id,
            category="thread",
            name="serialize_thread",
            started=thread_serialize_started,
        )
        _append_memory_debug_event("thread_save", "queued background thread persistence")
    except Exception:
        logger.exception("Could not save fitness thread state")
        _append_memory_debug_event("thread_save", "failed to serialize thread state")

    schedule_fitness_persistence(
        FitnessPersistenceRequest(
            user_id=user_id,
            request_id=request_id,
            selected_model=selected_model,
            backend_cfg=backend_cfg,
            user_prompt=user_prompt,
            assistant_text=content,
            image_bytes=image_bytes,
            image_name=image_name,
            session_key=session_key,
            agent_name=agent_name,
            serialized_thread=serialized_thread,
        ),
        hooks=FitnessPersistenceHooks(
            persist_async=_persist_fitness_turn_background_async,
            record_log=_diag_record_log,
            logger=logger,
        ),
    )
    _append_memory_debug_event("background_persist", "scheduled memory persistence worker")

    _append_memory_debug_event("run_end", "fitness turn completed")

    return content, usage_summary


_ui = get_config().ui
st.set_page_config(page_title=_ui.theme.page_title, page_icon=_ui.theme.page_icon, layout=_ui.theme.layout)

st.markdown(
        """
<style>
[data-testid="stSidebarNav"] {display: none !important;}
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
PREFS_PATH = Path(__file__).resolve().parents[1] / ".chatbot_ui_prefs.json"


def _load_ui_prefs() -> dict[str, Any]:
    if not PREFS_PATH.exists():
        return {}
    try:
        payload = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_ui_prefs(prefs: dict[str, Any]) -> None:
    try:
        PREFS_PATH.write_text(json.dumps(prefs, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    except Exception:
        logger.debug("Could not persist UI prefs", exc_info=True)


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
agent1_prompt_template_path = ""
agent1_system_prompt_override = ""
agent1_trace_placeholder: Any | None = None
ui_prefs = _load_ui_prefs()

with st.sidebar:
    st.header("Settings")
    saved_agent_choice = ui_prefs.get("agent_choice")
    if isinstance(saved_agent_choice, str) and saved_agent_choice in AGENT_OPTIONS:
        default_agent_index = AGENT_OPTIONS.index(saved_agent_choice)
    else:
        default_agent_index = AGENT_OPTIONS.index("Fitness Nutrition") if "Fitness Nutrition" in AGENT_OPTIONS else 0
    agent_choice = st.selectbox("Agent", AGENT_OPTIONS, index=default_agent_index)
    if ui_prefs.get("agent_choice") != agent_choice:
        ui_prefs["agent_choice"] = agent_choice
        _save_ui_prefs(ui_prefs)

    agent_config = next((agent for agent in AGENTS if agent.get("name") == agent_choice), {})
    available_providers = agent_config.get("available_providers", [])
    agent_model = agent_config.get("model")

    # --- AI Provider selector (hierarchical) or simple display ---
    if available_providers:
        _ap_names = [ap.get("name", "") for ap in available_providers if ap.get("name")]
        _saved_ap = ui_prefs.get("ai_provider_by_agent", {}).get(agent_choice)
        _ap_default_idx = _ap_names.index(_saved_ap) if isinstance(_saved_ap, str) and _saved_ap in _ap_names else 0
        selected_ai_provider = st.selectbox("AI Provider", _ap_names, index=_ap_default_idx, key="ai_provider_select")
        # persist choice
        _ap_prefs = ui_prefs.get("ai_provider_by_agent")
        if not isinstance(_ap_prefs, dict):
            _ap_prefs = {}
        if _ap_prefs.get(agent_choice) != selected_ai_provider:
            _ap_prefs[agent_choice] = selected_ai_provider
            ui_prefs["ai_provider_by_agent"] = _ap_prefs
            _save_ui_prefs(ui_prefs)

        _ap_entry = next((ap for ap in available_providers if ap.get("name") == selected_ai_provider), {})
        _ap_models = _ap_entry.get("models")

        # Resolve to the config-level provider for endpoint/key lookup
        provider_name = selected_ai_provider if selected_ai_provider != "AKS KAITO" else agent_config.get("provider", "AI Foundry")
        provider_config = next((p for p in PROVIDERS if p.get("name") == provider_name), {})
        if not provider_config and PROVIDERS:
            provider_config = PROVIDERS[0]
            provider_name = provider_config.get("name")

        # When the available_providers entry lists explicit models, use them;
        # otherwise fall back to the provider-level models list.
        if isinstance(_ap_models, list) and _ap_models:
            models = _split_models(_ap_models)
        else:
            # Try the direct models list first, then models_env, then model_env fallback
            _direct_models = provider_config.get("models")
            if isinstance(_direct_models, list) and _direct_models:
                models = _split_models(_direct_models)
            else:
                models_env = provider_config.get("models_env")
                if isinstance(models_env, list):
                    models = _split_models(models_env)
                else:
                    models = _split_models(os.getenv(models_env)) if models_env else []
            model_env = provider_config.get("model_env")
            provider_default_model = provider_config.get("default_model", "")
            if not models:
                model_fallback = _clean_env(os.getenv(model_env, provider_default_model)) if model_env else provider_default_model
                models = [m for m in [model_fallback] if m]
    else:
        provider_name = agent_config.get("provider")
        provider_config = next((p for p in PROVIDERS if p.get("name") == provider_name), {})
        if not provider_config and PROVIDERS:
            provider_config = PROVIDERS[0]
            provider_name = provider_config.get("name")

        st.text_input("Provider", provider_name or "", disabled=True)

        # Try the direct models list first, then models_env, then model_env fallback
        _direct_models = provider_config.get("models")
        if isinstance(_direct_models, list) and _direct_models:
            models = _split_models(_direct_models)
        else:
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

        if agent_model and agent_model not in models:
            models.append(agent_model)

        # Merge agent-level extra_models into the dropdown
        for _em in agent_config.get("extra_models", []):
            _em = _clean_env(str(_em))
            if _em and _em not in models:
                models.append(_em)

    endpoint_env = provider_config.get("endpoint_env")
    endpoint_default = provider_config.get("default_endpoint", "")
    endpoint = _clean_env(os.getenv(endpoint_env, endpoint_default)) if endpoint_env else endpoint_default

    api_key_env = provider_config.get("api_key_env")
    api_key = _clean_env(os.getenv(api_key_env)) if api_key_env else ""

    model_default = agent_model or (models[0] if models else "")
    model_options = models or [model_default]
    model_key = "model_select"
    saved_models = ui_prefs.get("model_by_agent")
    saved_model_for_agent = saved_models.get(agent_choice) if isinstance(saved_models, dict) else None
    if model_key in st.session_state and st.session_state[model_key] in model_options:
        model_index = model_options.index(st.session_state[model_key])
    elif isinstance(saved_model_for_agent, str) and saved_model_for_agent in model_options:
        model_index = model_options.index(saved_model_for_agent)
    elif model_default in model_options:
        model_index = model_options.index(model_default)
    else:
        model_index = 0
    model = st.selectbox("Model", model_options, index=model_index, key=model_key)
    model_by_agent = ui_prefs.get("model_by_agent")
    if not isinstance(model_by_agent, dict):
        model_by_agent = {}
    if model_by_agent.get(agent_choice) != model:
        model_by_agent[agent_choice] = model
        ui_prefs["model_by_agent"] = model_by_agent
        _save_ui_prefs(ui_prefs)

    _ai_defs = get_config().ai.defaults
    _labels = get_config().ui.labels
    temperature = st.slider(_labels.temperature, _ai_defs.temperature_min, _ai_defs.temperature_max, _ai_defs.temperature, _ai_defs.temperature_step)
    max_tokens = st.number_input(_labels.max_tokens, min_value=_ai_defs.max_tokens_min, max_value=_ai_defs.max_tokens_max, value=_ai_defs.max_tokens, step=1)
    top_p = st.slider(_labels.top_p, _ai_defs.top_p_min, _ai_defs.top_p_max, _ai_defs.top_p, _ai_defs.top_p_step)
    verify_tls = st.checkbox(_labels.verify_tls, value=_ai_defs.verify_tls)
    debug_enabled = st.checkbox(_labels.debug_mode, value=_ai_defs.debug_mode)

    if debug_enabled:
        _ensure_debug_log_handler()

    if st.button("New chat"):
        st.session_state.messages = []

    if agent_choice in {"Kaito Assistant", "KAITO RAG Assistant"} and _is_cluster_local_endpoint(endpoint):
        if not _is_running_in_kubernetes():
            st.warning(
                "This endpoint uses Kubernetes cluster-local DNS (*.svc.cluster.local) and is not reachable from this runtime. "
                "Use `kubectl port-forward` and configure a localhost URL such as "
                "`http://127.0.0.1:8000/v1/chat/completions`."
            )

    st.divider()
    st.subheader("Completion metrics")
    metrics_container = st.container(height=220)
    with metrics_container:
        if st.session_state.metrics_log:
            for entry in reversed(st.session_state.metrics_log[-get_config().ui.sidebar.metrics_max_entries:]):
                st.caption(_format_metrics_entry(entry))
        else:
            st.caption("No completions yet.")

    st.divider()
    _diag_url = f"/diagnostics?agent={urllib.request.pathname2url(agent_choice)}"
    st.markdown(
        f'<a href="{_diag_url}" target="_blank" '
        f'style="display:inline-flex;align-items:center;gap:0.35rem;'
        f'font-size:0.85rem;color:#4a9eff;text-decoration:none;">'
        f'{get_config().ui.labels.diagnostics_link}</a>',
        unsafe_allow_html=True,
    )

if agent_choice in {"Fitness Nutrition", "Agent1 Demo"}:
    chat_col, right_col = st.columns([3.2, 1.2], gap="large")
else:
    chat_col, right_col = st.container(), None

if right_col is not None and agent_choice == "Agent1 Demo":
    with right_col:
        st.subheader("Agent1 Config")

        templates = _agent1_list_prompt_templates()
        if not templates:
            default_template = "assistant_jinja.yaml"
            default_path = _agent1_prompts_dir() / default_template
            if not default_path.exists():
                default_path.write_text(
                    "name: Assistant\nmodel:\n  id: =Env.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME\ninstructions: You are a helpful assistant.\n",
                    encoding="utf-8",
                )
            templates = _agent1_list_prompt_templates()

        selected_template = st.selectbox("Prompt library", templates, key="agent1_template_select")
        selected_template_path = _agent1_prompts_dir() / selected_template
        agent1_prompt_template_path = str(selected_template_path)

        selected_marker_key = "agent1_template_selected_marker"
        if st.session_state.get(selected_marker_key) != selected_template:
            try:
                current_content = selected_template_path.read_text(encoding="utf-8")
            except Exception:
                current_content = ""
            st.session_state["agent1_template_editor"] = current_content
            try:
                loaded = yaml.safe_load(current_content) or {}
                default_instructions = str(loaded.get("instructions") or "")
            except Exception:
                default_instructions = ""
            st.session_state["agent1_system_prompt_override"] = default_instructions
            st.session_state[selected_marker_key] = selected_template

        st.markdown("**System prompt**")
        st.text_area(
            "Agent1 system prompt",
            key="agent1_system_prompt_override",
            height=180,
            label_visibility="collapsed",
            help="Override the system prompt used by Agent1 for this session.",
        )
        agent1_system_prompt_override = (st.session_state.get("agent1_system_prompt_override") or "").strip()

        st.markdown("**Template editor**")
        template_content = st.text_area(
            "template_content",
            key="agent1_template_editor",
            height=260,
            label_visibility="collapsed",
        )

        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button("Save template", key="agent1_save_template_btn"):
                try:
                    selected_template_path.write_text(template_content, encoding="utf-8")
                    st.success(f"Saved {selected_template}")
                except Exception as exc:
                    st.error(f"Save failed: {exc}")
        with delete_col:
            delete_disabled = len(templates) <= 1
            if st.button("Delete template", key="agent1_delete_template_btn", disabled=delete_disabled):
                try:
                    selected_template_path.unlink(missing_ok=False)
                    st.success(f"Deleted {selected_template}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")

        st.divider()
        st.markdown("**Add new template**")
        new_template_name = st.text_input("New template name", key="agent1_new_template_name")
        new_template_content = st.text_area(
            "New template content",
            key="agent1_new_template_content",
            height=140,
        )
        if st.button("Add template", key="agent1_add_template_btn"):
            try:
                normalized_name = _agent1_normalize_template_name(new_template_name)
                new_path = _agent1_prompts_dir() / normalized_name
                if new_path.exists():
                    raise ValueError(f"Template already exists: {normalized_name}")
                write_content = (
                    new_template_content.strip()
                    if new_template_content.strip()
                    else "name: Assistant\nmodel:\n  id: =Env.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME\ninstructions: You are a helpful assistant.\n"
                )
                new_path.write_text(write_content, encoding="utf-8")
                st.success(f"Added {normalized_name}")
                st.session_state["agent1_template_select"] = normalized_name
                st.session_state["agent1_new_template_name"] = ""
                st.session_state["agent1_new_template_content"] = ""
                st.rerun()
            except Exception as exc:
                st.error(f"Add failed: {exc}")

        st.divider()
        st.markdown("**Context window info**")
        short_messages = [
            msg for msg in st.session_state.messages if msg.get("role") in {"user", "assistant"}
        ]
        short_text = "\n".join((msg.get("content") or "") for msg in short_messages)
        est_input_tokens = _estimate_tokens_from_text(short_text)
        st.caption(
            f"short_term_messages={len(short_messages)} est_input_tokens={est_input_tokens} configured_max_output_tokens={int(max_tokens)}"
        )
        if st.session_state.metrics_log:
            st.caption(f"last_metrics={st.session_state.metrics_log[-1]}")

        st.divider()
        st.markdown("**MCP Wire Trace (Live)**")
        clear_trace_col, _ = st.columns([1, 1])
        with clear_trace_col:
            if st.button("Clear trace", key="agent1_clear_trace_btn"):
                st.session_state["agent1_mcp_trace"] = []
        agent1_trace_placeholder = st.empty()
        _render_agent1_trace(agent1_trace_placeholder)

if right_col is not None and agent_choice == "Fitness Nutrition":
    with right_col:
        st.subheader("Fitness Context")
        fitness_user_name_input = st.text_input(
            "User name",
            value=fitness_user_name,
            key="fitness_user_name",
            help="Used as the user context key for long-term memory retrieval.",
        )
        memory_user_id, resolved_from_name = _resolve_memory_user_id(fitness_user_name_input)
        _sync_fitness_session_user(memory_user_id)
        if resolved_from_name:
            st.caption(f"Retrieving profile for user: {fitness_user_name_input} (matched user_id: {memory_user_id})")
        else:
            st.caption(f"Retrieving profile for user: {memory_user_id}")

        force_snapshot_refresh = st.button(
            "Refresh memory snapshot",
            key=f"fitness_refresh_snapshot_{memory_user_id}",
            help="Reload the cached profile, body metrics, and meal history for the selected user.",
        )

        profile = {}
        recent_body_metrics = []
        meals = []
        read_model: dict[str, Any] = {}
        memory_error = None
        try:
            read_model = _load_fitness_snapshot(
                memory_user_id,
                metric_limit=get_config().ui.fitness.recent_meals_count,
                meal_limit=get_config().ui.fitness.recent_meals_count,
                force_refresh=force_snapshot_refresh,
            )
            profile = read_model.get("profile", {}) or {}
            recent_body_metrics = read_model.get("recent_body_metrics", []) or []
            meals = read_model.get("recent_meals", []) or []
        except Exception as exc:
            logger.exception("Could not load fitness memory panel")
            memory_error = str(exc)

        with st.expander("User Profile", expanded=False):
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
                        profile_lines.append(f"Birthday: {_display_or_na(_short_local_date(value))}")
                        continue
                    if key == "height_value":
                        profile_lines.append(
                            f"Height: {_format_height_display(profile.get('height_value'), profile.get('height_unit'))}"
                        )
                        continue
                    if key == "height_unit":
                        continue
                    profile_lines.append(f"{_pretty_profile_key(key)}: {_display_or_na(value)}")

                profile_lines.append(f"Current Age: {age if age is not None else 'n/a'}")

                latest_weight = _latest_metric(recent_body_metrics, "weight")
                latest_waist = _latest_metric(recent_body_metrics, "waist")
                latest_bp = _latest_metric(recent_body_metrics, "blood_pressure")

                if latest_weight:
                    observed = _short_local_date(latest_weight.get("observed_at"))
                    profile_lines.append(
                        f"Current Weight: {_format_metric_value(latest_weight.get('value_primary'))} {latest_weight.get('unit')} ({observed})"
                    )
                else:
                    profile_lines.append("Current Weight: n/a")
                if latest_waist:
                    observed = _short_local_date(latest_waist.get("observed_at"))
                    profile_lines.append(
                        f"Current Waist: {_format_metric_value(latest_waist.get('value_primary'))} {latest_waist.get('unit')} ({observed})"
                    )
                else:
                    profile_lines.append("Current Waist: n/a")
                if latest_bp:
                    observed = _short_local_date(latest_bp.get("observed_at"))
                    profile_lines.append(
                        f"Current Blood Pressure: {_format_metric_value(latest_bp.get('value_primary'), decimals=0)}/{_format_metric_value(latest_bp.get('value_secondary'), decimals=0)} {latest_bp.get('unit')} ({observed})"
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
        _fit_cfg = get_config().ui.fitness
        uploaded_food_image = st.file_uploader(
            get_config().ui.labels.food_upload,
            type=_fit_cfg.accepted_image_types,
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
                st.session_state[chat_input_key] = get_config().ui.fitness.default_food_prompt
        else:
            st.caption("No image attached.")

        st.divider()
        memory_backend = get_config().database.default_backend
        memory_backend_label = "Azure SQL" if memory_backend in {"azuresql", "azure_sql", "azure-sql"} else "SQLite"
        st.markdown(f"**Long-term memory ({memory_backend_label})**")
        try:
            if meals:
                st.caption(f"Last {get_config().ui.fitness.recent_meals_count} meals and macros")
                for meal in meals[:get_config().ui.fitness.recent_meals_count]:
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
        ][-get_config().ui.fitness.recent_messages_count:]
        if short_memory:
            _trunc = get_config().ui.fitness.message_truncate_length
            for msg in short_memory:
                role = msg.get("role", "?")
                text = (msg.get("content") or "").strip().replace("\n", " ")
                st.caption(f"{role}: {text[:_trunc]}{'...' if len(text) > _trunc else ''}")
        else:
            st.caption("No short-term memory yet.")

        st.divider()
        with st.expander("Memory Debug", expanded=False):
            debug_user_id, debug_resolved = _resolve_memory_user_id(st.session_state.get("fitness_user_name", "roy"))
            debug_session_key = f"fitness:{debug_user_id}"
            debug_backend = get_config().database.default_backend
            debug_backend_label = "Azure SQL" if debug_backend in {"azuresql", "azure_sql", "azure-sql"} else "SQLite"
            if st.button("Force refresh debug snapshot", key=f"fitness_refresh_debug_{debug_user_id}"):
                _invalidate_fitness_snapshot_cache(debug_user_id)
                st.rerun()
            st.caption(f"backend={debug_backend_label}")
            st.caption(f"user_id={debug_user_id} | session_key={debug_session_key}")
            if debug_resolved:
                st.caption("user name resolved via external key or profile name match")

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
                    value=json.dumps(debug_read_model, indent=2, ensure_ascii=False, default=str),
                    height=220,
                    disabled=True,
                    key=f"memory_debug_read_model_json_{debug_backend_label}_{debug_user_id}",
                )
            except Exception as exc:
                st.caption(f"Could not render read model JSON: {exc}")

            st.markdown("**Operation Log**")
            debug_events = st.session_state.get("memory_debug_events", [])
            if debug_events:
                st.text_area(
                    "memory_events",
                    value="\n".join(debug_events[-get_config().ui.performance.max_events:]),
                    height=220,
                    disabled=True,
                    key=f"memory_debug_events_log_{debug_user_id}",
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
                    if not debug_text.strip():
                        debug_text = "No debug logs captured for this response."
                    st.markdown(
                        (
                            "<div style='background-color:#111111;padding:0.75rem;border-radius:0.5rem;'>"
                            "<pre style='white-space:pre-wrap;color:#ffffff;margin:0;font-size:0.8rem;'>"
                            f"{html.escape(debug_text)}"
                            "</pre></div>"
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask something...", key=chat_input_key)
    if prompt:
        logger.info("User prompt received. Agent=%s Provider=%s Model=%s", agent_choice, provider_name, model)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            started = time.perf_counter()
            status = "ok"
            usage_summary = ""
            content_rendered = False
            request_id = f"{agent_choice.lower().replace(' ', '-')}-{int(time.time() * 1000)}"
            agent_init_elapsed = 0.0
            model_call_elapsed = 0.0
            turn_debug_logs: list[str] = []
            _diag_system_prompt = ""
            _diag_context_data: dict | None = None
            _diag_errors: list[str] = []

            def _add_turn_debug(line: str) -> None:
                safe_line = _redact_sensitive_text(line)
                stamped = f"[{datetime.now(ZoneInfo(_browser_timezone_name())).strftime('%m/%d/%Y')}] {safe_line}"
                turn_debug_logs.append(stamped)
                logger.info("TURN_DEBUG %s", safe_line)

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
                    _diag_system_prompt = "You are a helpful AI assistant."
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
                    _diag_system_prompt = "You are a helpful Kaito assistant."
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
                    _diag_system_prompt = "You are a KAITO RAG assistant."
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
                    if agent1_prompt_template_path:
                        os.environ["AGENT1_PROMPT_TEMPLATE_PATH"] = agent1_prompt_template_path
                    else:
                        os.environ.pop("AGENT1_PROMPT_TEMPLATE_PATH", None)

                    if agent1_system_prompt_override:
                        os.environ["AGENT1_SYSTEM_PROMPT_OVERRIDE"] = agent1_system_prompt_override
                    else:
                        os.environ.pop("AGENT1_SYSTEM_PROMPT_OVERRIDE", None)

                    llm_started = time.perf_counter()
                    _append_agent1_trace("client->server", "initialize", {"transport": "stdio"})
                    _render_agent1_trace(agent1_trace_placeholder)
                    output = _Agent1LiveCapture(trace_placeholder=agent1_trace_placeholder)
                    with redirect_stdout(output):
                        import agent1_demo as agent1_demo_module

                        importlib.reload(agent1_demo_module)
                        runtime_agent1 = agent1_demo_module.agent1
                        params = inspect.signature(runtime_agent1).parameters
                        if len(params) >= 1:
                            asyncio.run(runtime_agent1(prompt))
                        else:
                            asyncio.run(runtime_agent1())
                    llm_elapsed = time.perf_counter() - llm_started
                    model_call_elapsed = llm_elapsed
                    raw_output = output.getvalue().strip()
                    _append_agent1_trace("server->client", "run/complete", {"ok": bool(raw_output)})
                    _render_agent1_trace(agent1_trace_placeholder)
                    content = _format_agent1_output(raw_output) or "No output from agent1."
                    _add_turn_debug(f"workflow latency_s={llm_elapsed:.2f}")
                else:
                    logger.info("Running Fitness Nutrition agent")
                    assistant_placeholder = st.empty()
                    _diag_system_prompt = (
                        "You are a fitness nutrition assistant with access to user profile, body metrics, and meal macro history. "
                        "When users ask about goals or trends, use tracked data first and ask clarifying questions if missing data."
                    )
                    fitness_user_id, _ = _resolve_memory_user_id(st.session_state.get("fitness_user_name", "roy"))
                    # Route to the right backend based on the model chosen in the sidebar dropdown.
                    # model_backends in config.yaml maps model names to provider names.
                    _fit_agent_cfg = next((a for a in AGENTS if a.get("name") == "Fitness Nutrition"), {})
                    _model_backends = _fit_agent_cfg.get("model_backends", {})
                    _backend_provider_name = _model_backends.get(model) if isinstance(_model_backends, dict) else None
                    if _backend_provider_name:
                        _backend_cfg = {"provider": _backend_provider_name, "model": model}
                        fitness_chat_client, fitness_model = _build_fitness_chat_client(_backend_cfg, PROVIDERS)
                    else:
                        fitness_chat_client, fitness_model = None, model
                    fitness_started = time.perf_counter()
                    content, usage_summary = asyncio.run(
                        _run_fitness_turn(
                            user_prompt=prompt,
                            user_id=fitness_user_id,
                            selected_model=fitness_model or model,
                            image_bytes=uploaded_food_image_bytes,
                            image_name=uploaded_food_image_name,
                            assistant_placeholder=assistant_placeholder,
                            request_id=request_id,
                            chat_client=fitness_chat_client,
                            backend_cfg=_backend_cfg if _backend_provider_name else None,
                        )
                    )
                    content_rendered = True
                    fitness_elapsed = time.perf_counter() - fitness_started
                    model_call_elapsed = fitness_elapsed
                    usage_parsed = _parse_usage_summary(usage_summary)
                    _add_turn_debug(
                        f"fitness_run latency_s={fitness_elapsed:.2f} usage_input={usage_parsed['input']} usage_output={usage_parsed['output']} usage_total={usage_parsed['total']}"
                    )
                    try:
                        model_snapshot = _load_fitness_snapshot(fitness_user_id, metric_limit=get_config().ui.fitness.recent_meals_count, meal_limit=get_config().ui.fitness.recent_meals_count, request_id=request_id)
                        _diag_context_data = model_snapshot
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

                if not content_rendered:
                    st.markdown(content)
                debug_snapshot = turn_debug_logs if debug_enabled else []
                st.session_state.messages.append(
                    {"role": "assistant", "content": content, "debug_logs": debug_snapshot}
                )
            except urllib.error.HTTPError as exc:
                status = f"http-{exc.code}"
                error_body = exc.read().decode("utf-8") if exc.fp else ""
                safe_reason = _redact_sensitive_text(exc.reason)
                safe_body = _redact_sensitive_text(error_body)
                logger.error("HTTP error: %s %s %s", exc.code, safe_reason, safe_body)
                error_message = f"Request failed: {exc.code} {safe_reason}\n{safe_body}"
                _add_turn_debug(f"error type=http code={exc.code} reason={exc.reason}")
                _diag_errors.append(f"HTTP {exc.code}: {safe_reason}")
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"❌ {error_message}",
                        "debug_logs": turn_debug_logs if debug_enabled else [],
                    }
                )
            except urllib.error.URLError as exc:
                status = "url-error"
                safe_reason = _redact_sensitive_text(exc.reason)
                logger.error("URL error: %s", safe_reason)
                error_message = f"Request failed: {safe_reason}"
                _add_turn_debug(f"error type=url reason={exc.reason}")
                _diag_errors.append(f"URL error: {safe_reason}")
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"❌ {error_message}",
                        "debug_logs": turn_debug_logs if debug_enabled else [],
                    }
                )
            except Exception as exc:
                status = "error"
                safe_exc = _redact_sensitive_text(exc)
                logger.exception("Unhandled error: %s", safe_exc)
                error_message = f"Request failed: {safe_exc}"
                _add_turn_debug(f"error type=exception class={type(exc).__name__} detail={exc}")
                _diag_errors.append(f"{type(exc).__name__}: {safe_exc}")
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"❌ {error_message}",
                        "debug_logs": turn_debug_logs if debug_enabled else [],
                    }
                )
            finally:
                elapsed_s = time.perf_counter() - started
                post_elapsed = max(0.0, elapsed_s - agent_init_elapsed - model_call_elapsed)
                if debug_enabled:
                    _add_turn_debug(f"request_end status={status} total_latency_s={elapsed_s:.2f}")
                metrics_line = (
                    f"agent={agent_choice} | provider={provider_name or '-'} | model={model or '-'} | "
                    f"status={status} | latency_s={elapsed_s:.2f} | "
                    f"breakdown_s=init:{agent_init_elapsed:.2f},llm:{model_call_elapsed:.2f},post:{post_elapsed:.2f}"
                )
                if usage_summary:
                    metrics_line = f"{metrics_line} | {usage_summary}"
                st.session_state.metrics_log.append(metrics_line)
                logger.info("Completion: %s", _redact_sensitive_text(metrics_line))

                # ── Record diagnostics turn ──────────────────────────
                try:
                    _du = _parse_usage_summary(usage_summary)
                    _sys_est = _diag_est_tokens(_diag_system_prompt)
                    _ctx_est = _diag_est_tokens(json.dumps(_diag_context_data, default=str)) if _diag_context_data else 0
                    _input_tok = _du["input"] or _estimate_tokens_from_text(prompt)
                    _hist_est = max(0, _input_tok - _sys_est - _ctx_est)
                    _diag_record_turn(DiagnosticsTurn(
                        request_id=request_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        agent=agent_choice,
                        model=model or "",
                        provider=provider_name or "",
                        status=status,
                        latency_s=round(elapsed_s, 3),
                        input_tokens=_du["input"],
                        output_tokens=_du["output"],
                        total_tokens=_du["total"],
                        context_window_max=_diag_ctx_size(model or ""),
                        system_prompt_est_tokens=_sys_est,
                        context_provider_est_tokens=_ctx_est,
                        chat_history_est_tokens=_hist_est,
                        output_reserved_tokens=int(max_tokens),
                        messages_count=len([m for m in st.session_state.messages if m.get("role") in {"user", "assistant"}]),
                        debug_logs=turn_debug_logs[:],
                        errors=_diag_errors[:],
                    ))
                    if _diag_errors:
                        for _de in _diag_errors:
                            _diag_record_log(agent_choice, _de, "ERROR")
                except Exception:
                    logger.debug("Could not record diagnostics turn", exc_info=True)

                if not st.session_state._metrics_rerun:
                    st.session_state._metrics_rerun = True
                    st.rerun()

if debug_enabled:
    pass
