import asyncio
import os
import sys
import logging
from pathlib import Path
import json
import mimetypes
import random
from prompt_toolkit import prompt
import yaml
from agent_framework import ChatAgent, AgentRunResponse, ChatMessage, DataContent, TextContent, UsageContent, UsageDetails
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from jinja2 import Template
from openai import RateLimitError
from pydantic import BaseModel

from config import Settings
from db import DEFAULT_DB_PATH, StructuredOutputStore
from ai_chat_client import KaitoChatClient


def _str_to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def configure_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_to_console = _str_to_bool(os.getenv("LOG_TO_CONSOLE"), True)
    log_to_file = _str_to_bool(os.getenv("LOG_TO_FILE"), False)
    log_file_path = os.getenv("LOG_FILE", "agentframework.log")

    handlers: list[logging.Handler] = []
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_to_file:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    if not handlers:
        handlers.append(logging.NullHandler())

    logging.basicConfig(level=log_level, handlers=handlers)


logger = logging.getLogger(__name__)


class WorkflowPlan(BaseModel):
    """
    Represents a structured plan for a workflow, validated and serialized via Pydantic.

    This model captures:
    - ``steps``: An ordered list of step descriptions (strings) that define what to do.
    - ``rationale``: A human-readable explanation for why these steps were chosen.

    About Pydantic ``BaseModel``:
    Pydantic's ``BaseModel`` is a base class that enables *runtime data parsing and
    validation* based on Python type hints. By inheriting from ``BaseModel``, this
    class gains features such as:

    - Type validation and coercion when creating instances (e.g., ensuring ``steps``
        is a list of strings).
    - Helpful error messages when input data does not match the declared schema.
    - Easy serialization/deserialization (e.g., to/from dict/JSON).
    - Generated schema/metadata useful for docs and tooling.

    Why ``BaseModel`` appears in the class definition:
    In Python, ``class WorkflowPlan(BaseModel):`` means "define ``WorkflowPlan`` as a
    subclass of ``BaseModel``." This inheritance is how ``WorkflowPlan`` obtains
    Pydantic's validation/serialization behavior; without it, the annotations would
    be ordinary type hints and no automatic validation would occur.
    """
    steps: list[str]
    rationale: str


class StructuredOutput(BaseModel):
    steps: list[str]
    rationale: str
    type: str


class MacroNutrients(BaseModel):
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    confidence: str
    notes: str


def load_prompt_template(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    model_id = data.get("model", {}).get("id")
    if isinstance(model_id, str) and model_id.startswith("=Env."):
        env_key = model_id.removeprefix("=Env.")
        data["model"]["id"] = os.getenv(env_key, model_id)

    return data


def render_instructions(template: str, context: dict[str, str]) -> str:
    if "{{" in template and "}}" in template:
        return Template(template).render(**context)
    return template.format(**context)


def format_usage(usage: UsageDetails) -> str:
    return (
        f"input={usage.input_token_count or 0} "
        f"output={usage.output_token_count or 0} "
        f"total={usage.total_token_count or 0}"
    )


def get_backoff_seconds(attempt: int) -> float:
    base = float(os.getenv("RATE_LIMIT_BASE_DELAY", "60"))
    max_delay = float(os.getenv("RATE_LIMIT_MAX_DELAY", "300"))
    exp = min(max_delay, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, base * 0.1)
    return exp + jitter


async def run_with_retry(agent: ChatAgent, *args, max_retries: int = 5, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            response = await agent.run(*args, **kwargs)
            if os.getenv("STREAM_TOKENS", "1") == "1" and response.usage_details:
                print(f"Tokens: {format_usage(response.usage_details)}")
            return response
        except RateLimitError as exc:
            if attempt == max_retries:
                raise
            wait_seconds = get_backoff_seconds(attempt)
            print(f"Rate limit hit. Retrying in {wait_seconds:.1f}s (attempt {attempt}/{max_retries})")
            await asyncio.sleep(wait_seconds)


async def run_with_stream(agent: ChatAgent, messages: str, *, max_retries: int = 5) -> AgentRunResponse:
    for attempt in range(1, max_retries + 1):
        try:
            response_updates = []
            async for update in agent.run_stream(messages):
                if update.text:
                    print(update.text, end="", flush=True)
                if os.getenv("STREAM_TOKENS", "1") == "1":
                    usage_chunks = [c for c in update.contents if isinstance(c, UsageContent)]
                    for usage_content in usage_chunks:
                        print(f"\nTokens: {format_usage(usage_content.details)}")
                response_updates.append(update)
            print()
            response = AgentRunResponse.from_agent_run_response_updates(response_updates)
            if os.getenv("STREAM_TOKENS", "1") == "1" and response.usage_details:
                print(f"Tokens: {format_usage(response.usage_details)}")
            return response
        except RateLimitError:
            if attempt == max_retries:
                raise
            wait_seconds = get_backoff_seconds(attempt)
            print(f"\nRate limit hit. Retrying in {wait_seconds:.1f}s (attempt {attempt}/{max_retries})")
            await asyncio.sleep(wait_seconds)



async def agent1() -> None:
    """
    Run a small two-step demo using an Azure OpenAI chat agent configured from a YAML
    prompt template.
    Overview:
    - Loads environment variables from a `.env` file.
    - Loads `assistant.yaml` from the local `prompts/` directory to obtain:
        - `instructions` (formatted with optional CLI input),
        - agent metadata such as `name`, `model`, `tools`,
        - runtime settings like `max_iterations`, `temperature`, `top_p`, and `verbose`.
    - Builds an `AzureOpenAIChatClient` using `AzureCliCredential()` and converts it to an agent.
    - Executes two agent calls:
        1) Ask for a pirate joke and print the response.
        2) Ask the agent to produce a 3-step workflow in JSON using the previous response as context.
    Inputs:
    - Optional command-line argument `sys.argv[1]` is treated as `data_input` and is injected
        into the `instructions` template via `.format(data_input=data_input)`.
    Notes on the string-building syntax used for `workflow_prompt`:
    - Parentheses `( ... )` around multiple string parts allow Python to treat them as a
        single expression that spans multiple lines without using backslashes.
    - When multiple string literals appear next to each other inside parentheses, Python
        performs *implicit string literal concatenation*, e.g.:
            ("a" "b")  -> "ab"
    - This also works when mixing a normal string literal with an adjacent f-string, e.g.:
            ("prefix " f"{value}") -> "prefix " + str(value)
    - In this function, the `workflow_prompt` is formed by several adjacent string literals
        plus a final f-string containing `result.text`, producing one combined prompt string.
    """
    load_dotenv()
    print("Hello from agentframework!")
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
    model_id = model_block.get("id") if isinstance(model_block, dict) else model_block

    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = ChatAgent(
        chat_client=chat_client,
        instructions=instructions,
        name=prompt.get("name", "Assistant"),
        model=model_id,
        tools=prompt.get("tools", []),
        max_tokens=prompt.get("max_tokens")
    )
    
    if os.getenv("STREAM_OUTPUT", "1") == "1":
        print("Step 1 result (streaming):")
        result = await run_with_stream(agent, "Tell me a joke about a pirate.")
    else:
        result = await run_with_retry(agent, "Tell me a joke about a pirate.")
        print("Step 1 result:\n", result.text)

    workflow_prompt = (
        "Using the previous response, create a simple 3-step workflow that shows how "
        "an agent would proceed. Return a JSON object with keys: steps (array of strings) "
        "and rationale (string).\n\nPrevious response:\n"
        f"{result.text}"
    )
    workflow_plan = await run_with_retry(agent, workflow_prompt, response_format=WorkflowPlan)
    print("Step 2 workflow:\n", workflow_plan)

    if hasattr(workflow_plan, "model_dump"):
        workflow_payload = workflow_plan.model_dump()
    else:
        workflow_payload = workflow_plan

    structured_prompt = (
        "Using this workflow plan, produce the final structured output as JSON with keys: "
        "steps (array of strings), rationale (string), and type (string, must be 'Chat').\n\n"
        f"Workflow plan:\n{workflow_payload}"
    )
    final_result = await run_with_retry(agent, structured_prompt, response_format=StructuredOutput)
    print("Step 3 structured output:\n", final_result)

    if hasattr(final_result, "model_dump"):
        final_payload = final_result.model_dump()
    elif hasattr(final_result, "parsed"):
        final_payload = final_result.parsed
    elif hasattr(final_result, "text"):
        try:
            final_payload = json.loads(final_result.text)
        except json.JSONDecodeError:
            final_payload = {
                "steps": [final_result.text],
                "rationale": "",
                "type": "Chat",
            }
    else:
        final_payload = final_result

    store = StructuredOutputStore(DEFAULT_DB_PATH)
    output_id = store.insert(
        steps=final_payload.get("steps", []) if isinstance(final_payload, dict) else [],
        rationale=final_payload.get("rationale", "") if isinstance(final_payload, dict) else "",
        output_type=final_payload.get("type", "Chat") if isinstance(final_payload, dict) else "Chat",
    )
    print(f"Saved structured output with id={output_id}")

    all_structured_outputs = store.list_all()
    print("All structured outputs in the database:")
    for output in all_structured_outputs:
        # Best-effort conversion to a JSON-serializable dict
        if hasattr(output, "model_dump"):
            payload = output.model_dump()
        elif hasattr(output, "_asdict"):
            payload = output._asdict()
        elif isinstance(output, dict):
            payload = output
        else:
            try:
                payload = dict(output)  # e.g., sqlite3.Row / mapping-like
            except Exception:
                payload = {"value": output}

        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))

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


async def fitness_agent(image_path: str | None = None) -> None:
    """
    Estimate macronutrients for a food image.
    Notes:
    - Requires a vision-capable model deployment.
    - The model will provide estimates; verify with nutrition labels when possible.
    """
    load_dotenv()
    image_path = image_path or (sys.argv[1] if len(sys.argv) > 1 else prompt("Image path: "))
    if not image_path:
        print("No image file path provided.")
        return

    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Image not found: {image_file}")
        return

    mime_type, _ = mimetypes.guess_type(image_file.name)
    mime_type = mime_type or "application/octet-stream"
    image_bytes = image_file.read_bytes()

    instructions = (
        "You are a fitness nutrition assistant. Use the provided food image to estimate "
        "macronutrients. Provide best-effort estimates with clear uncertainty."
    )

    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = ChatAgent(
        chat_client=chat_client,
        instructions=instructions,
        name="fitness_agent",
        model=Settings().azure_openai_chat_deployment,
        tools=[],
        max_completion_tokens=800,
        temperature=1.0,
    )

    # Build a multipart ChatMessage with text + image content parts.
    # This sends the image as a proper vision content block (image_url)
    # instead of embedding the base64 string in the text prompt.
    prompt_text = (
        "Estimate macronutrients for the food in this image. "
        "Return estimates in grams for protein, carbs, fat, and calories. "
        "Include a confidence level (low/medium/high) and short notes."
    )
    message = ChatMessage(
        role="user",
        contents=[
            TextContent(text=prompt_text),
            DataContent(data=image_bytes, media_type=mime_type),
        ],
    )

    try:
        result = await run_with_retry(agent, message, response_format=MacroNutrients)
        print("Macronutrient estimates:\n", result)
        if result.usage_details:
            print(f"Image request tokens: {format_usage(result.usage_details)}")
    except RateLimitError:
        print(
            "Rate limit reached. Please wait ~60 seconds and retry. "
            "Consider lowering request frequency or requesting a quota increase."
        )
    

if __name__ == "__main__":
    configure_logging()
    logger.info("Starting agentframework main")
    asyncio.run(agent1())
