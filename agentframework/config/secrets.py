"""Resolve *_env fields to actual values from environment variables.

This is the ONLY module that materialises secrets.  The rest of the app
works with the resolved values – never raw env-var names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from config.models import ExternalIdentitiesConfig, OAuthProviderConfig, ProviderConfig


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


@dataclass(frozen=True)
class ResolvedOAuthProvider:
    """An OAuth provider with credentials resolved from env vars."""

    provider_name: str
    client_id: str
    client_secret: str
    authority_url: str
    scope: str
    enabled: bool


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


def resolve_oauth_provider_secrets(provider: OAuthProviderConfig) -> ResolvedOAuthProvider:
    """Return a *ResolvedOAuthProvider* with client_id and client_secret materialized from env."""
    return ResolvedOAuthProvider(
        provider_name=provider.provider_name,
        client_id=resolve_env(provider.client_id_env),
        client_secret=resolve_env(provider.client_secret_env),
        authority_url=provider.authority_url,
        scope=provider.scope,
        enabled=provider.enabled,
    )


def resolve_external_identities_secrets(
    identities_config: ExternalIdentitiesConfig,
) -> dict[str, ResolvedOAuthProvider]:
    """Return a dict mapping provider_name to ResolvedOAuthProvider with secrets materialised.
    
    Only includes providers that are enabled. Returns empty dict if external_identities is disabled.
    """
    if not identities_config.enabled:
        return {}

    resolved = {}
    for provider in identities_config.providers:
        if provider.enabled:
            resolved_provider = resolve_oauth_provider_secrets(provider)
            resolved[provider.provider_name] = resolved_provider

    return resolved


def redact_secret(value: str, visible_chars: int = 4) -> str:
    """Return a redacted version of *value* for display purposes."""
    if not value or len(value) <= visible_chars:
        return "***"
    return value[:visible_chars] + "***"
