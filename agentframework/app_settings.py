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


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


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
    db_path: Path = Path(_get_env("FITNESS_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()
    fitness_db_backend: str = _get_env("FITNESS_DB_BACKEND", "sqlite").strip().lower()
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

    # Azure SQL database settings
    azure_sql_server: str = _get_env("AZURE_SQL_SERVER", "")
    azure_sql_database: str = _get_env("AZURE_SQL_DATABASE", "")
    azure_sql_schema: str = _get_env("AZURE_SQL_SCHEMA", "dbo")
    azure_sql_driver: str = _get_env("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server")
    azure_sql_auth_mode: str = _get_env("AZURE_SQL_AUTH_MODE", "defaultazurecredential").strip().lower()
    azure_sql_admin_user: str = _get_env("AZURE_SQL_ADMIN_USER", "")
    azure_sql_admin_password: str = _get_env("AZURE_SQL_ADMIN_PASSWORD", "")
    azure_sql_encrypt: bool = _get_env_bool("AZURE_SQL_ENCRYPT", True)
    azure_sql_trust_server_certificate: bool = _get_env_bool("AZURE_SQL_TRUST_SERVER_CERTIFICATE", False)
    azure_sql_connection_timeout: int = _get_env_int("AZURE_SQL_CONNECTION_TIMEOUT", 30)


settings = Settings()