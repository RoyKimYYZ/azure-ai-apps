"""config – unified application configuration for AgentFramework.

Public API
----------
- ``get_config()``  – return the cached :class:`AppConfig` singleton
- ``reload_config()`` – re-read from disk and return the new config
- ``save_config(cfg)`` – write an AppConfig back to ``appconfig.yaml``
- ``resolve_env()`` / ``resolve_provider_secrets()`` – secret helpers
- ``redact_secret()`` – mask a value for safe display
"""

from config.loader import get_config, reload_config
from config.models import AppConfig
from config.secrets import redact_secret, resolve_env, resolve_provider_secrets
from config.writer import save_config

__all__ = [
    "AppConfig",
    "get_config",
    "redact_secret",
    "reload_config",
    "resolve_env",
    "resolve_provider_secrets",
    "save_config",
]
