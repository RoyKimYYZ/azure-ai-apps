"""Minimal tests for the config package."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the agentframework root is on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig, get_config, reload_config, save_config  # noqa: E402
from config.models import (  # noqa: E402
    AIConfig,
    AIDefaultsConfig,
    AppMeta,
    DatabaseConfig,
    LoggingConfig,
    MCPConfig,
    RuntimeConfig,
    UIConfig,
)


def test_pydantic_defaults():
    """Every section model should instantiate with defaults."""
    for model_cls in (AppMeta, RuntimeConfig, LoggingConfig, DatabaseConfig, AIConfig, AIDefaultsConfig, UIConfig, MCPConfig):
        instance = model_cls()
        assert instance is not None


def test_appconfig_defaults():
    """AppConfig instantiates with all-defaults when no YAML is provided."""
    cfg = AppConfig()
    assert cfg.app.name == "AgentFramework Chatbot"
    assert cfg.runtime.port == 8000
    assert cfg.logging.level == "INFO"
    assert cfg.database.default_backend == "sqlite"
    assert cfg.ai.defaults.temperature == 0.2
    assert cfg.ui.theme.page_title == "AI Foundry Chatbot"
    assert cfg.mcp.github.api_base == "https://api.github.com"
    assert cfg.admin.enabled is True


def test_get_config_loads_yaml():
    """get_config() should return a valid AppConfig from appconfig.yaml."""
    cfg = get_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.app.name == "AgentFramework Chatbot"
    assert len(cfg.ai.providers) >= 1
    assert len(cfg.ai.agents) >= 1


def test_reload_config():
    """reload_config() should return a fresh AppConfig."""
    cfg1 = get_config()
    cfg2 = reload_config()
    assert isinstance(cfg2, AppConfig)
    assert cfg2.app.name == cfg1.app.name


def test_save_round_trip(tmp_path: Path):
    """save_config + load should round-trip without data loss."""
    cfg = get_config()
    out_path = tmp_path / "appconfig_test.yaml"
    save_config(cfg, config_path=out_path)

    # Reload from the written file.
    cfg2 = reload_config(config_path=out_path)
    assert cfg2.app.name == cfg.app.name
    assert cfg2.runtime.port == cfg.runtime.port
    assert len(cfg2.ai.providers) == len(cfg.ai.providers)
    assert len(cfg2.ai.agents) == len(cfg.ai.agents)

    # Restore original config for other tests.
    reload_config()
