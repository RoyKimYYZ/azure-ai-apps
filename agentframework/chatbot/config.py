from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelEndpoint:
    name: str
    endpoint: str
    api_key: str | None
    models: list[str]


@dataclass(frozen=True)
class Settings:
    """
    Encapsulates configuration settings for chatbot model endpoints.
    """
    ai_foundry_endpoint: str
    ai_foundry_api_key: str | None
    ai_foundry_models: list[str]

    @classmethod
    def from_env(cls) -> "Settings":
        ai_foundry_models = _split_models(os.getenv("AI_FOUNDRY_MODELS"))
        if not ai_foundry_models:
            ai_foundry_models = [model for model in [_clean_env(os.getenv("AI_FOUNDRY_MODEL", ""))] if model]

        return cls(
            ai_foundry_endpoint=_clean_env(os.getenv("AI_FOUNDRY_ENDPOINT", "")),
            ai_foundry_api_key=_clean_env(os.getenv("AI_FOUNDRY_API_KEY")),
            ai_foundry_models=ai_foundry_models,
        )

    def to_model_endpoints(self) -> list[ModelEndpoint]:
        return [
            ModelEndpoint(
                name="AI Foundry",
                endpoint=self.ai_foundry_endpoint,
                api_key=self.ai_foundry_api_key,
                models=self.ai_foundry_models,
            ),
        ]


def _split_models(value: str | None) -> list[str]:
    if not value:
        return []
    return [_clean_env(item) for item in value.split(",") if _clean_env(item)]


def _clean_env(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip('"').strip("'")


def get_model_endpoints() -> list[ModelEndpoint]:
    """
    Returns configured model endpoints and their supported models.

    Env overrides:
      - AI_FOUNDRY_ENDPOINT
      - AI_FOUNDRY_API_KEY
      - AI_FOUNDRY_MODELS (comma-separated)
    """
    settings = Settings.from_env()
    return settings.to_model_endpoints()


def get_endpoint_by_name(name: str, endpoints: Iterable[ModelEndpoint]) -> ModelEndpoint | None:
    for endpoint in endpoints:
        if endpoint.name == name:
            return endpoint
    return None
