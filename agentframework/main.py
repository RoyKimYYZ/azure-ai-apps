import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml
from agent_framework import Agent, ChatOptions
from agent_framework.openai import OpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from jinja2 import Template

from agent1_demo import agent1
from ai_chat_client import KaitoChatClient
from config import get_config, resolve_env, resolve_provider_secrets
from logging_colors import ColorLogFormatter, strip_ansi, supports_color
from run_utils import run_with_retry


def configure_logging() -> None:
    log_cfg = get_config().logging
    log_level = log_cfg.level.upper()
    log_to_console = log_cfg.to_console
    log_to_file = log_cfg.to_file
    log_file_path = log_cfg.file_path
    log_color = log_cfg.color and supports_color(sys.stdout)

    handlers: list[logging.Handler] = []
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    console_formatter = ColorLogFormatter(fmt) if log_color else logging.Formatter(fmt)

    file_formatter = logging.Formatter(fmt)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    if log_to_file:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(file_formatter)
        # Keep file logs plain even if console logs use ANSI colors.
        class _StripAnsiFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    record.msg = strip_ansi(str(record.getMessage()))
                    record.args = ()
                except Exception:
                    pass
                return True

        file_handler.addFilter(_StripAnsiFilter())
        handlers.append(file_handler)

    if not handlers:
        handlers.append(logging.NullHandler())

    # If any library configured logging before we run (common in CLIs), `basicConfig`
    # becomes a no-op unless `force=True`.
    try:
        logging.basicConfig(level=log_level, handlers=handlers, force=True)
    except TypeError:
        logging.basicConfig(level=log_level, handlers=handlers)


logger = logging.getLogger(__name__)

# Utility functions
def load_prompt_template(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    model_id = data.get("model", {}).get("id")
    if isinstance(model_id, str) and model_id.startswith("=Env."):
        env_key = model_id.removeprefix("=Env.")
        data["model"]["id"] = os.getenv(env_key, model_id)

    return data


def render_instructions(template: str, context: dict[str, str]) -> str:
    """Render instructions from a template and context.

    Args:
        template (str): The template string containing placeholders.
        context (dict[str, str]): A dictionary mapping placeholders to their values.

    Returns:
        str: The rendered instructions with placeholders replaced by context values.
    """
    if "{{" in template and "}}" in template:
        return Template(template).render(**context)
    return template.format(**context)

async def azure_foundry_general_agent() -> Agent:
    load_dotenv()
    prompt_path = Path(
        os.getenv(
            "PROMPT_TEMPLATE_PATH",
            Path(__file__).parent / "prompts" / "assistant_jinja.yaml",
        )
    )
    prompt = load_prompt_template(prompt_path)

    data_input = sys.argv[1] if len(sys.argv) > 1 else ""
    instructions = render_instructions(
        prompt.get("instructions", "You are a helpful assistant."),
        {"data_input": data_input},
    )

    model_block = prompt.get("model", {})
    prompt_model_id = model_block.get("id") if isinstance(model_block, dict) else model_block
    cfg = get_config()
    model_id = os.getenv("CHAT_MODEL") or prompt_model_id or cfg.azure.openai.chat_deployment
    if not model_id:
        raise ValueError("Chat model is required. Set CHAT_MODEL or provide model in the prompt.")

    endpoint = resolve_env(cfg.azure.openai.endpoint_env)
    if isinstance(endpoint, str):
        endpoint = endpoint.strip().strip('"').strip("'").strip()
    chat_client = OpenAIChatClient(
        credential=AzureCliCredential(),
        azure_endpoint=endpoint or None,
    )
    agent = Agent(
        client=chat_client,
        instructions=instructions,
        name=prompt.get("name", "GeneralChatAssistant"),
        default_options=ChatOptions(model=model_id, max_tokens=prompt.get("max_tokens")),  # type: ignore[typeddict-item]
        tools=prompt.get("tools", []),
    )
    return agent

async def kaito_agent() -> None:
    logger.info("Starting KAITO agent")
    load_dotenv()
    prompt_path = Path(
        os.getenv(
            "PROMPT_TEMPLATE_PATH",
            Path(__file__).parent / "prompts" / "assistant_jinja.yaml",
        )
    )
    logger.debug("KAITO prompt path: %s", prompt_path)
    prompt = load_prompt_template(prompt_path)

    data_input = sys.argv[1] if len(sys.argv) > 1 else ""
    instructions = render_instructions(
        prompt.get("instructions", "You are a helpful assistant."),
        {"data_input": data_input},
    )

    model_block = prompt.get("model", {})
    model_id = model_block.get("id") if isinstance(model_block, dict) else model_block
    kaito_prov = next((p for p in get_config().ai.providers if p.name == "KAITO"), None)
    if kaito_prov:
        _resolved = resolve_provider_secrets(kaito_prov)
        model_id = model_id or _resolved.default_model or "phi-4-mini-instruct"
        endpoint = _resolved.endpoint
        api_key = _resolved.api_key or None
    else:
        model_id = model_id or "phi-4-mini-instruct"
        endpoint = "http://workspace-phi-4-mini.default.svc.cluster.local:80/v1/chat/completions"
        api_key = None
    if not model_id:
        raise ValueError("KAITO model is required. Set KAITO_MODEL or provide model in the prompt.")
    logger.info("KAITO model selected: %s", model_id)
    logger.debug("KAITO endpoint configured: %s", bool(endpoint))
    logger.debug("KAITO api key configured: %s", bool(api_key))


    chat_client = KaitoChatClient(
        endpoint=endpoint,
        api_key=api_key,
        default_model=model_id,
    )
    agent = Agent(
        client=chat_client,
        instructions=instructions,
        name=prompt.get("name", "KaitoAssistant"),
        default_options=ChatOptions(
            model=model_id,
            temperature=prompt.get("temperature"),  # type: ignore[typeddict-item]
            top_p=prompt.get("top_p"),  # type: ignore[typeddict-item]
            max_tokens=prompt.get("max_tokens"),  # type: ignore[typeddict-item]
        ),
        tools=prompt.get("tools", []),
    )

    logger.info("Sending KAITO greeting prompt")
    result = await run_with_retry(agent, "Hello from KAITO. Give a one-sentence reply.")
    logger.info("KAITO response received")
    print("KAITO result:\n", result.text)


async def kaito_ragengine_bge_small_agent(
    index_name: str | None = None,
) -> Agent:
    """
    Build a Agent that targets a KAITO RAGEngine deployment.

    The RAGEngine sits in front of the inference model (phi-4-mini) and adds
    retrieval-augmented generation using a local bge-small-en-v1.5 embedding
    model.  Its ``/v1/chat/completions`` endpoint is OpenAI-compatible with an
    extra ``index_name`` field that selects which document index to ground
    against.

    Architecture:
        User prompt ──► RAGEngine ──► vector search (bge-small) ──►
        retrieved context + prompt ──► phi-4-mini LLM ──► grounded response

    Configuration:
        Settings are read from ``appconfig.yaml`` via the ``config`` package.
        The "KAITO RAGEngine" provider entry supplies the endpoint, API key,
        index name, and default model.  Environment variables referenced in
        the provider (``KAITO_RAGENGINE_ENDPOINT``, ``KAITO_RAGENGINE_API_KEY``,
        ``KAITO_RAGENGINE_INDEX``) are resolved automatically.
    """
    logger.info("Starting KAITO RAGEngine agent")
    load_dotenv()

    prompt_path = Path(
        os.getenv(
            "PROMPT_TEMPLATE_PATH",
            Path(__file__).parent / "prompts" / "assistant_jinja.yaml",
        )
    )
    logger.debug("RAGEngine prompt path: %s", prompt_path)
    prompt = load_prompt_template(prompt_path)

    data_input = sys.argv[1] if len(sys.argv) > 1 else ""
    instructions = render_instructions(
        prompt.get("instructions", "You are a helpful assistant."),
        {"data_input": data_input},
    )

    # Model & provider – resolved from config
    model_block = prompt.get("model", {})
    model_id = model_block.get("id") if isinstance(model_block, dict) else model_block
    rag_prov = next(
        (p for p in get_config().ai.providers if p.name == "KAITO RAGEngine"), None
    )
    if rag_prov:
        _resolved = resolve_provider_secrets(rag_prov)
        model_id = model_id or _resolved.default_model or "phi-4-mini-instruct"
        endpoint = _resolved.endpoint
        api_key = _resolved.api_key or None
        rag_index = index_name or _resolved.index_name or "rag_index"
    else:
        model_id = model_id or "phi-4-mini-instruct"
        endpoint = "http://<ragengine-service-name>.default.svc.cluster.local:80"
        api_key = None
        rag_index = index_name or "rag_index"
    logger.info("RAGEngine inference model: %s", model_id)
    logger.debug("RAGEngine endpoint: %s", endpoint)
    logger.debug("RAGEngine api key configured: %s", bool(api_key))
    logger.info("RAGEngine index_name: %s", rag_index)

    chat_client = KaitoChatClient(
        endpoint=endpoint,
        api_key=api_key,
        default_model=model_id,
        extra_payload={"index_name": rag_index},
    )

    agent = Agent(
        client=chat_client,
        instructions=instructions,
        name=prompt.get("name", "KaitoRAGEngineAssistant"),
        default_options=ChatOptions(model=model_id, max_tokens=prompt.get("max_tokens")),  # type: ignore[typeddict-item]
        tools=prompt.get("tools", []),
    )
    logger.info("KAITO RAGEngine agent built successfully")
    return agent





if __name__ == "__main__":
    configure_logging()
    logger.info("Starting agentframework main")
    asyncio.run(agent1())
