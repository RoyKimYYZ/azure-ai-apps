import asyncio
import os
import sys
import logging
from pathlib import Path
import yaml
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from jinja2 import Template

from app_settings import Settings
from ai_chat_client import KaitoChatClient
from run_utils import run_with_retry
from logging_colors import ColorLogFormatter, strip_ansi, supports_color
from agent1_demo import agent1


def _str_to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def configure_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_to_console = _str_to_bool(os.getenv("LOG_TO_CONSOLE"), True)
    log_to_file = _str_to_bool(os.getenv("LOG_TO_FILE"), False)
    log_file_path = os.getenv("LOG_FILE", "agentframework.log")
    log_color = _str_to_bool(os.getenv("LOG_COLOR"), True) and supports_color(sys.stdout)

    handlers: list[logging.Handler] = []
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    console_formatter: logging.Formatter
    if log_color:
        console_formatter = ColorLogFormatter(fmt)
    else:
        console_formatter = logging.Formatter(fmt)

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

async def azure_foundry_general_agent() -> None:
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
    model_id = os.getenv("CHAT_MODEL") or prompt_model_id or Settings().azure_openai_chat_deployment
    if not model_id:
        raise ValueError("Chat model is required. Set CHAT_MODEL or provide model in the prompt.")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or getattr(Settings(), "azure_openai_endpoint", None)
    if isinstance(endpoint, str):
        endpoint = endpoint.strip().strip('"').strip("'").strip()
    client_kwargs = {"endpoint": endpoint} if endpoint else {}

    chat_client = AzureOpenAIChatClient(
        credential=AzureCliCredential(),
        **client_kwargs,
    )
    agent = ChatAgent(
        chat_client=chat_client,
        instructions=instructions,
        name=prompt.get("name", "GeneralChatAssistant"),
        model=model_id,
        tools=prompt.get("tools", []),
        max_tokens=prompt.get("max_tokens"),
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
    model_id = model_id or os.getenv("KAITO_MODEL", "phi-4-mini-instruct")
    if not model_id:
        raise ValueError("KAITO model is required. Set KAITO_MODEL or provide model in the prompt.")
    logger.info("KAITO model selected: %s", model_id)

    endpoint = os.getenv("KAITO_INFERENCE_ENDPOINT")
    if not endpoint:
        endpoint = "http://workspace-phi-4-mini.default.svc.cluster.local:80/v1/chat/completions"
    api_key = os.getenv("KAITO_API_KEY") or None
    logger.debug("KAITO endpoint configured: %s", bool(endpoint))
    logger.debug("KAITO api key configured: %s", bool(api_key))


    chat_client = KaitoChatClient(
        endpoint=endpoint,
        api_key=api_key,
        default_model=model_id,
    )
    agent = ChatAgent(
        max_iterations=prompt.get("max_iterations"),
        temperature=prompt.get("temperature"),
        top_p=prompt.get("top_p"),
        verbose=prompt.get("verbose"),
        chat_client=chat_client,
        instructions=instructions,
        name=prompt.get("name", "KaitoAssistant"),
        model=model_id,
        tools=prompt.get("tools", []),
        max_tokens=prompt.get("max_tokens"),
    )

    logger.info("Sending KAITO greeting prompt")
    result = await run_with_retry(agent, "Hello from KAITO. Give a one-sentence reply.")
    logger.info("KAITO response received")
    print("KAITO result:\n", result.text)


async def kaito_ragengine_bge_small_agent(
    index_name: str | None = None,
) -> ChatAgent:
    """
    Build a ChatAgent that targets a KAITO RAGEngine deployment.

    The RAGEngine sits in front of the inference model (phi-4-mini) and adds
    retrieval-augmented generation using a local bge-small-en-v1.5 embedding
    model.  Its ``/v1/chat/completions`` endpoint is OpenAI-compatible with an
    extra ``index_name`` field that selects which document index to ground
    against.

    Architecture:
        User prompt ──► RAGEngine ──► vector search (bge-small) ──►
        retrieved context + prompt ──► phi-4-mini LLM ──► grounded response

    Environment variables:
        KAITO_RAGENGINE_ENDPOINT   Base URL of the RAGEngine K8s service.
        KAITO_RAGENGINE_API_KEY    Optional bearer token (cluster-internal
                                   deployments typically need none).
        KAITO_RAGENGINE_INDEX      Default index name for RAG queries.
        KAITO_MODEL                Model name reported by the backend
                                   (default: phi-4-mini-instruct).
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

    # Model – the inference model behind the RAGEngine
    model_block = prompt.get("model", {})
    model_id = model_block.get("id") if isinstance(model_block, dict) else model_block
    model_id = model_id or os.getenv("KAITO_MODEL", "phi-4-mini-instruct")
    logger.info("RAGEngine inference model: %s", model_id)

    # Endpoint – the RAGEngine service (NOT the raw inference workspace)
    endpoint = os.getenv("KAITO_RAGENGINE_ENDPOINT")
    if not endpoint:
        # TODO: Replace with your actual RAGEngine service URL after deployment
        endpoint = "http://<ragengine-service-name>.default.svc.cluster.local:80"
    api_key = os.getenv("KAITO_RAGENGINE_API_KEY") or None
    logger.debug("RAGEngine endpoint: %s", endpoint)
    logger.debug("RAGEngine api key configured: %s", bool(api_key))

    # Index name – which document index to ground against
    rag_index = index_name or os.getenv("KAITO_RAGENGINE_INDEX", "rag_index")
    logger.info("RAGEngine index_name: %s", rag_index)

    chat_client = KaitoChatClient(
        endpoint=endpoint,
        api_key=api_key,
        default_model=model_id,
        extra_payload={"index_name": rag_index},
    )

    agent = ChatAgent(
        chat_client=chat_client,
        instructions=instructions,
        name=prompt.get("name", "KaitoRAGEngineAssistant"),
        model=model_id,
        tools=prompt.get("tools", []),
        max_tokens=prompt.get("max_tokens"),
    )
    logger.info("KAITO RAGEngine agent built successfully")
    return agent



    

if __name__ == "__main__":
    configure_logging()
    logger.info("Starting agentframework main")
    asyncio.run(agent1())
