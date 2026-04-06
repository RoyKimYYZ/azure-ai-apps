"""Serialise AppConfig back to YAML and write to disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from config.models import AppConfig

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "appconfig.yaml"


def save_config(config: AppConfig, *, config_path: Path | None = None) -> None:
    """Dump *config* as YAML to *config_path* (defaults to appconfig.yaml)."""
    path = config_path or _CONFIG_PATH
    data = _serialise(config)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info("Configuration saved to %s", path)


def _serialise(config: AppConfig) -> dict[str, Any]:
    """Convert AppConfig to a plain dict suitable for yaml.dump.

    Uses by_alias=True so that aliased fields (e.g. schema -> schema_name)
    are written under their YAML-facing key.
    """
    return config.model_dump(by_alias=True)
