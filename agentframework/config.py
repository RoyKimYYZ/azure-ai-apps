from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).parent
DEFAULT_DB_PATH = BASE_DIR / "agentframework.db"
DEFAULT_PROMPT_PATH = BASE_DIR / "prompts" / "assistant.yaml"


def _get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value is not None else default


@dataclass(frozen=True)
class Settings:
    # Azure OpenAI (required for agent to run)
    azure_openai_endpoint: str = _get_env("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = _get_env("AZURE_OPENAI_API_KEY")
    azure_openai_chat_deployment: str = _get_env("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-5.2-chat")

    # Optional model settings
    azure_openai_api_version: str = _get_env("AZURE_OPENAI_API_VERSION", "")
    azure_openai_embedding_deployment: str = _get_env(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
    )
    azure_openai_responses_deployment: str = _get_env(
        "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME", "gpt-5.2-responses"
    )

    # Local project settings
    db_path: Path = DEFAULT_DB_PATH
    prompt_path: Path = DEFAULT_PROMPT_PATH
    log_level: str = _get_env("LOG_LEVEL", "INFO")

    # AKS / deployment runtime settings
    environment: str = _get_env("ENVIRONMENT", "dev")
    port: str = _get_env("PORT", "8000")
    workers: str = _get_env("WORKERS", "1")
    timeout: str = _get_env("TIMEOUT", "60")

    # Observability (OpenTelemetry / App Insights)
    appinsights_connection_string: str = _get_env("APPINSIGHTS_CONNECTION_STRING", "")
    otel_service_name: str = _get_env("OTEL_SERVICE_NAME", "agentframework")
    otel_exporter_otlp_endpoint: str = _get_env("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    # Identity (Workload identity or service principal)
    azure_client_id: str = _get_env("AZURE_CLIENT_ID", "")
    azure_tenant_id: str = _get_env("AZURE_TENANT_ID", "")
    azure_client_secret: str = _get_env("AZURE_CLIENT_SECRET", "")


settings = Settings()
