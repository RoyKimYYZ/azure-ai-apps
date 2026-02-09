from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable
from collections.abc import AsyncIterable

from agent_framework import (
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    FinishReason,
    TextContent,
    UsageDetails,
)
from agent_framework._clients import BaseChatClient


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1/chat/completions"):
        return endpoint
    return f"{endpoint}/v1/chat/completions"


@dataclass(frozen=True)
class AIChatclientConfig:
    endpoint: str
    api_key: str | None = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    timeout_seconds: float = 30.0
    verify_tls: bool = True
    default_model: str | None = None


class AIChatclient:
    """
    Minimal client for a KAITO-hosted, OpenAI-compatible chat completions endpoint.

    Environment variables (optional):
      - KAITO_INFERENCE_ENDPOINT: base URL for the KAITO service (no path),
        e.g. https://kaito.example.com or http://kaito.kube.local:8000
      - KAITO_API_KEY: optional bearer token
      - KAITO_MODEL: default model name
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        api_key: str | None = None,
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer",
        timeout_seconds: float = 30.0,
        verify_tls: bool = True,
        default_model: str | None = None,
    ) -> None:
        endpoint = endpoint or os.getenv("KAITO_INFERENCE_ENDPOINT", "")
        if not endpoint:
            raise ValueError("KAITO inference endpoint is required.")
        api_key = api_key or os.getenv("KAITO_API_KEY")
        default_model = default_model or os.getenv("KAITO_MODEL")
        self._config = AIChatclientConfig(
            endpoint=_normalize_endpoint(endpoint),
            api_key=api_key,
            api_key_header=api_key_header,
            api_key_prefix=api_key_prefix,
            timeout_seconds=timeout_seconds,
            verify_tls=verify_tls,
            default_model=default_model,
        )

    def complete(
        self,
        *,
        messages: Iterable[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": list(messages),
            "model": model or self._config.default_model,
        }
        if payload["model"] is None:
            raise ValueError("Model name is required. Set KAITO_MODEL or pass model=.")
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if extra_body:
            payload.update(extra_body)

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._config.api_key:
            token = self._config.api_key
            if self._config.api_key_prefix:
                token = f"{self._config.api_key_prefix} {token}"
            headers[self._config.api_key_header] = token

        request = urllib.request.Request(
            self._config.endpoint,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            context = None
            if not self._config.verify_tls:
                import ssl

                context = ssl._create_unverified_context()

            with urllib.request.urlopen(
                request,
                timeout=self._config.timeout_seconds,
                context=context,
            ) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8") if exc.fp else ""
            raise RuntimeError(
                f"KAITO chat request failed: {exc.code} {exc.reason} - {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"KAITO chat request failed: {exc.reason}") from exc


class KaitoChatClient(BaseChatClient):
    """Async adapter compatible with the agent framework's chat client expectations."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        verify_tls: bool = True,
        default_model: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._extra_payload = extra_payload or {}
        self._client = AIChatclient(
            endpoint=endpoint,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            verify_tls=verify_tls,
            default_model=default_model,
        )

    async def _inner_get_response(
        self,
        *,
        messages: list[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> ChatResponse:
        payload_messages: list[dict[str, Any]] = []
        for message in messages:
            payload_messages.append(
                {
                    "role": message.role.value if hasattr(message.role, "value") else str(message.role),
                    "content": message.text,
                }
            )

        extra_body: dict[str, Any] = dict(self._extra_payload)
        if chat_options.temperature is not None:
            extra_body["temperature"] = chat_options.temperature
        if chat_options.max_tokens is not None:
            extra_body["max_tokens"] = chat_options.max_tokens
        if chat_options.top_p is not None:
            extra_body["top_p"] = chat_options.top_p

        response = await asyncio.to_thread(
            self._client.complete,
            messages=payload_messages,
            model=chat_options.model_id or self._client._config.default_model,
            extra_body=extra_body,
        )

        choice = None
        choices = response.get("choices") if isinstance(response, dict) else None
        if isinstance(choices, list) and choices:
            choice = choices[0]

        text = ""
        finish_reason = None
        if isinstance(choice, dict):
            finish_reason = choice.get("finish_reason")
            message = choice.get("message") or {}
            if isinstance(message, dict):
                text = message.get("content") or ""
            else:
                text = choice.get("text") or ""

        response_id = response.get("id") if isinstance(response, dict) else None
        created_at = None
        if isinstance(response, dict):
            created_raw = response.get("created")
            if isinstance(created_raw, (int, float)):
                created_at = datetime.fromtimestamp(created_raw, tz=timezone.utc).isoformat()
            elif isinstance(created_raw, str):
                created_at = created_raw

        usage_details = None
        if isinstance(response, dict) and isinstance(response.get("usage"), dict):
            usage = response.get("usage") or {}
            usage_details = UsageDetails(
                input_token_count=usage.get("prompt_tokens"),
                output_token_count=usage.get("completion_tokens"),
                total_token_count=usage.get("total_tokens"),
            )

        chat_message = ChatMessage(role="assistant", text=text)
        return ChatResponse(
            messages=[chat_message],
            response_id=response_id,
            created_at=created_at,
            model_id=response.get("model") if isinstance(response, dict) else chat_options.model_id,
            finish_reason=FinishReason(value=finish_reason) if finish_reason else None,
            usage_details=usage_details,
            response_format=chat_options.response_format,
            raw_representation=response,
        )

    async def _inner_get_streaming_response(
        self,
        *,
        messages: list[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        response = await self._inner_get_response(messages=messages, chat_options=chat_options, **kwargs)
        text = response.messages[0].text if response.messages else ""
        yield ChatResponseUpdate(
            text=TextContent(text=text),
            role="assistant",
            response_id=response.response_id,
            created_at=response.created_at,
            model_id=response.model_id,
            finish_reason=response.finish_reason,
        )
