"""Resolve *_env fields to actual values from environment variables.

This is the ONLY module that materialises secrets.  The rest of the app
works with the resolved values – never raw env-var names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from config.models import ProviderConfig


def resolve_env(env_key: str, default: str = "") -> str:
    """Look up *env_key* in the process environment."""
    if not env_key:
        return default
    return os.getenv(env_key, default)


@dataclass(frozen=True)
class ResolvedProvider:
    """A provider with endpoint and api_key resolved from env vars."""

    name: str
    endpoint: str
    api_key: str
    models: list[str]
    default_model: str
    default_endpoint: str
    index_name: str
    default_index_name: str
    embedding_model: str
    request_model: str


def resolve_provider_secrets(provider: ProviderConfig) -> ResolvedProvider:
    """Return a *ResolvedProvider* with env vars materialised."""
    return ResolvedProvider(
        name=provider.name,
        endpoint=resolve_env(provider.endpoint_env, provider.default_endpoint),
        api_key=resolve_env(provider.api_key_env),
        models=list(provider.models),
        default_model=provider.default_model,
        default_endpoint=provider.default_endpoint,
        index_name=resolve_env(provider.index_name_env, provider.default_index_name),
        default_index_name=provider.default_index_name,
        embedding_model=provider.embedding_model,
        request_model=provider.request_model,
    )


def redact_secret(value: str, visible_chars: int = 4) -> str:
    """Return a redacted version of *value* for display purposes."""
    if not value or len(value) <= visible_chars:
        return "***"
    return value[:visible_chars] + "***"
