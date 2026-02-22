import logging
import mimetypes
import os
import sys
import json
from pathlib import Path

from agent_framework import ChatAgent, ChatMessage, DataContent, TextContent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from openai import RateLimitError
from prompt_toolkit import prompt
from pydantic import BaseModel

from config import Settings
from logging_colors import colorize_columns, supports_color
from run_utils import format_usage, run_with_stream


logger = logging.getLogger(__name__)


def _log_color_enabled() -> bool:
    return os.getenv("LOG_COLOR", "1") == "1" and supports_color(sys.stdout)


class MacroNutrients(BaseModel):
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    confidence: str
    notes: str


async def fitness_agent(image_path: str | None = None) -> None:
    """Estimate macronutrients for a food image.

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
        result = await run_with_stream(agent, message, response_format=MacroNutrients)
        logger.debug("Macronutrient estimates: %s", result)

        macros: dict[str, object] | None = None
        parsed = getattr(result, "parsed", None)
        if isinstance(parsed, MacroNutrients):
            macros = parsed.model_dump()
        elif isinstance(parsed, dict):
            macros = parsed
        elif hasattr(result, "text") and isinstance(getattr(result, "text"), str):
            try:
                macros = json.loads(result.text)
            except json.JSONDecodeError:
                macros = None

        macros = macros or {}
        logger.info(
            "Macros: %s",
            colorize_columns(
                [
                    f"cal={macros.get('calories', '?')}",
                    f"p_g={macros.get('protein_g', '?')}",
                    f"c_g={macros.get('carbs_g', '?')}",
                    f"f_g={macros.get('fat_g', '?')}",
                    f"conf={macros.get('confidence', '?')}",
                ],
                enabled=_log_color_enabled(),
            ),
        )
        notes = macros.get("notes", "") if isinstance(macros, dict) else ""
        if notes:
            logger.info("Notes: %s", notes)
        if getattr(result, "usage_details", None):
            logger.info("Image request tokens: %s", format_usage(result.usage_details))
    except RateLimitError:
        logger.warning(
            "Rate limit reached. Please wait ~60 seconds and retry. "
            "Consider lowering request frequency or requesting a quota increase."
        )
