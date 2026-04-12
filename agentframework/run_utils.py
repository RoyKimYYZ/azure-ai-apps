import asyncio
import random
from typing import Any

from agent_framework import Agent, AgentResponse, UsageDetails
from openai import RateLimitError

from config import get_config


def format_usage(usage: UsageDetails) -> str:
    return (
        f"input={usage.get('input_token_count') or 0} "
        f"output={usage.get('output_token_count') or 0} "
        f"total={usage.get('total_token_count') or 0}"
    )


def get_backoff_seconds(attempt: int) -> float:
    cfg = get_config()
    base = cfg.runtime.retry.rate_limit_base_delay
    max_delay = cfg.runtime.retry.rate_limit_max_delay
    exp = min(max_delay, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, base * 0.1)
    return exp + jitter


async def run_with_retry(agent: Agent, *args: Any, max_retries: int = 5, **kwargs: Any) -> AgentResponse:
    for attempt in range(1, max_retries + 1):
        try:
            response = await agent.run(*args, **kwargs)
            if get_config().runtime.stream_tokens and getattr(response, "usage_details", None):
                print(f"Tokens: {format_usage(response.usage_details)}")
            return response
        except RateLimitError:
            if attempt == max_retries:
                raise
            wait_seconds = get_backoff_seconds(attempt)
            print(
                f"Rate limit hit. Retrying in {wait_seconds:.1f}s (attempt {attempt}/{max_retries})"
            )
            await asyncio.sleep(wait_seconds)
    raise RuntimeError("Exhausted retries")


async def run_with_stream(agent: Agent, messages: Any, *, max_retries: int = 5, **kwargs: Any) -> AgentResponse:
    for attempt in range(1, max_retries + 1):
        try:
            response_updates = []
            async for update in agent.run(messages, stream=True, **kwargs):
                if getattr(update, "text", None):
                    print(update.text, end="", flush=True)
                if get_config().runtime.stream_tokens:
                    usage_details = getattr(update, "usage_details", None)
                    if usage_details:
                        print(f"\nTokens: {format_usage(usage_details)}")
                response_updates.append(update)
            print()
            response = AgentResponse.from_updates(response_updates)
            _stream_usage = getattr(response, "usage_details", None)
            if get_config().runtime.stream_tokens and _stream_usage is not None:
                print(f"Tokens: {format_usage(_stream_usage)}")
            return response
        except RateLimitError:
            if attempt == max_retries:
                raise
            wait_seconds = get_backoff_seconds(attempt)
            print(f"\nRate limit hit. Retrying in {wait_seconds:.1f}s (attempt {attempt}/{max_retries})")
            await asyncio.sleep(wait_seconds)
    raise RuntimeError("Exhausted retries")
