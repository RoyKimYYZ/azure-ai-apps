"""Load appconfig.yaml, validate with Pydantic, cache as a singleton."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from config.models import AppConfig

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "appconfig.yaml"
_config: AppConfig | None = None


def get_config(*, config_path: Path | None = None) -> AppConfig:
    """Return the cached AppConfig.  Loads from disk on first call."""
    global _config
    if _config is None:
        _config = _load_from_disk(config_path or _CONFIG_PATH)
    return _config


def reload_config(*, config_path: Path | None = None) -> AppConfig:
    """Force re-read from disk and return the new AppConfig."""
    global _config
    _config = _load_from_disk(config_path or _CONFIG_PATH)
    logger.info("Configuration reloaded from %s", config_path or _CONFIG_PATH)
    return _config


def _load_from_disk(path: Path) -> AppConfig:
    """Read YAML, validate, return AppConfig."""
    if not path.exists():
        logger.warning("Config file %s not found – using Pydantic defaults", path)
        return AppConfig()

    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    return AppConfig(**raw)
