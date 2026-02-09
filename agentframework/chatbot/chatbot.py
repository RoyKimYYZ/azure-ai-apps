import asyncio
import io
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

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

from agent_framework import ChatAgent, ChatMessage
from ai_chat_client import KaitoChatClient
from main import (
    agent1,
    azure_foundry_general_agent,
    fitness_agent,
    load_prompt_template,
    render_instructions,
    run_with_retry,
)
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


st.set_page_config(page_title="AI Foundry Chatbot", page_icon="🤖", layout="centered")

st.title("AI Foundry Chatbot")

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

with st.sidebar:
    st.header("Settings")
    agent_choice = st.selectbox("Agent", AGENT_OPTIONS, index=0)
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

    fitness_image_path = ""
    if agent_choice == "Fitness Nutrition":
        fitness_image_path = st.text_input("Image path", "", help="Path to a local image for nutrition estimate")

    if st.button("New chat"):
        st.session_state.messages = []

    st.divider()
    st.subheader("Completion metrics")
    metrics_container = st.container(height=220)
    with metrics_container:
        if st.session_state.metrics_log:
            for entry in reversed(st.session_state.metrics_log[-50:]):
                st.caption(entry)
        else:
            st.caption("No completions yet.")

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

prompt = st.chat_input("Ask something...")
if prompt:
    logger.info("User prompt received. Agent=%s Endpoint=%s Model=%s", agent_choice, provider_name, model)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        started = time.perf_counter()
        status = "ok"
        usage_summary = ""
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
                agent = asyncio.run(azure_foundry_general_agent())
                history_messages = [
                    ChatMessage(role=msg.get("role", "user"), text=msg.get("content", ""))
                    for msg in st.session_state.messages
                ]
                result = asyncio.run(run_with_retry(agent, history_messages))
                content = _extract_display_text(getattr(result, "text", None) or str(result))
                usage = getattr(result, "usage_details", None)
                if usage:
                    usage_summary = (
                        f"input={usage.input_token_count or 0} "
                        f"output={usage.output_token_count or 0} "
                        f"total={usage.total_token_count or 0}"
                    )
            elif agent_choice == "Kaito Assistant":
                logger.info("Running Kaito Assistant agent")
                if endpoint:
                    os.environ["KAITO_INFERENCE_ENDPOINT"] = endpoint
                if api_key:
                    os.environ["KAITO_API_KEY"] = api_key
                if model:
                    os.environ["KAITO_MODEL"] = model
                agent = _build_kaito_agent(model)
                result = asyncio.run(run_with_retry(agent, prompt))
                content = _extract_display_text(getattr(result, "text", None) or str(result))
                usage = getattr(result, "usage_details", None)
                if usage:
                    usage_summary = (
                        f"input={usage.input_token_count or 0} "
                        f"output={usage.output_token_count or 0} "
                        f"total={usage.total_token_count or 0}"
                    )
            elif agent_choice == "KAITO RAG Assistant":
                logger.info("Running KAITO RAG Assistant agent")
                if endpoint:
                    os.environ["KAITO_RAGENGINE_ENDPOINT"] = endpoint
                if api_key:
                    os.environ["KAITO_RAGENGINE_API_KEY"] = api_key
                if model:
                    os.environ["KAITO_MODEL"] = model
                agent = _build_kaito_ragengine_agent(model)
                result = asyncio.run(run_with_retry(agent, prompt))
                content = _extract_display_text(getattr(result, "text", None) or str(result))
                usage = getattr(result, "usage_details", None)
                if usage:
                    usage_summary = (
                        f"input={usage.input_token_count or 0} "
                        f"output={usage.output_token_count or 0} "
                        f"total={usage.total_token_count or 0}"
                    )
            elif agent_choice == "Agent1 Demo":
                logger.info("Running Agent1 Demo")
                output = io.StringIO()
                with redirect_stdout(output):
                    asyncio.run(agent1())
                raw_output = output.getvalue().strip()
                content = _format_agent1_output(raw_output) or "No output from agent1."
            else:
                logger.info("Running Fitness Nutrition agent")
                if not fitness_image_path:
                    st.warning("Provide an image path to run the fitness agent.")
                    st.stop()
                output = io.StringIO()
                with redirect_stdout(output):
                    asyncio.run(fitness_agent(fitness_image_path))
                raw_output = output.getvalue().strip()
                content = _extract_display_text(raw_output) or "No output from fitness agent."

            st.markdown(content)
            debug_snapshot = list(st.session_state.get("debug_logs", [])) if debug_enabled else []
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
            metrics_line = (
                f"agent={agent_choice} | endpoint={provider_name or '-'} | model={model or '-'} | "
                f"status={status} | latency_s={elapsed_s:.2f}"
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
