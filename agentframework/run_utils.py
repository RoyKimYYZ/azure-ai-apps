import asyncio
import random

from agent_framework import ChatAgent, AgentRunResponse, UsageContent, UsageDetails
from config import get_config
from openai import RateLimitError


def format_usage(usage: UsageDetails) -> str:
    return (
        f"input={usage.input_token_count or 0} "
        f"output={usage.output_token_count or 0} "
        f"total={usage.total_token_count or 0}"
    )


def get_backoff_seconds(attempt: int) -> float:
    cfg = get_config()
    base = cfg.runtime.retry.rate_limit_base_delay
    max_delay = cfg.runtime.retry.rate_limit_max_delay
    exp = min(max_delay, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, base * 0.1)
    return exp + jitter


async def run_with_retry(agent: ChatAgent, *args, max_retries: int = 5, **kwargs):
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


async def run_with_stream(agent: ChatAgent, messages, *, max_retries: int = 5, **kwargs) -> AgentRunResponse:
    for attempt in range(1, max_retries + 1):
        try:
            response_updates = []
            async for update in agent.run_stream(messages, **kwargs):
                if getattr(update, "text", None):
                    print(update.text, end="", flush=True)
                if get_config().runtime.stream_tokens:
                    usage_chunks = [c for c in getattr(update, "contents", []) if isinstance(c, UsageContent)]
                    for usage_content in usage_chunks:
                        print(f"\nTokens: {format_usage(usage_content.details)}")
                response_updates.append(update)
            print()
            response = AgentRunResponse.from_agent_run_response_updates(response_updates)
            if get_config().runtime.stream_tokens and getattr(response, "usage_details", None):
                print(f"Tokens: {format_usage(response.usage_details)}")
            return response
        except RateLimitError:
            if attempt == max_retries:
                raise
            wait_seconds = get_backoff_seconds(attempt)
            print(f"\nRate limit hit. Retrying in {wait_seconds:.1f}s (attempt {attempt}/{max_retries})")
            await asyncio.sleep(wait_seconds)
