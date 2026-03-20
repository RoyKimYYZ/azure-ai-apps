"""Shared diagnostics data store for the chatbot.

Uses a ``st.cache_resource`` singleton so the main chatbot and the
diagnostics page (which may run in a separate browser tab) share the
same in-memory store without any disk I/O or file-locking.  A
``threading.Lock`` guards mutations for safety.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import streamlit as st

MAX_TURNS_PER_AGENT = 200
MAX_LOGS_PER_AGENT = 500

# ── Context-window sizes (tokens) per model ──────────────────────────
CONTEXT_WINDOW_SIZES: dict[str, int] = {
    "gpt-5.2-chat": 128_000,
    "gpt-5-mini": 128_000,
    "deepseek-v3.2": 128_000,
    "phi-4": 16_384,
    "phi-4-mini-instruct": 128_000,
    "phi3.5 vision": 128_000,
    "phi3.5-vision": 128_000,
}
DEFAULT_CONTEXT_WINDOW = 128_000


def get_context_window_size(model: str) -> int:
    model_lower = (model or "").lower().strip()
    for key, size in CONTEXT_WINDOW_SIZES.items():
        if key in model_lower or model_lower in key:
            return size
    if "phi-4" in model_lower and "mini" not in model_lower:
        return 16_384
    return DEFAULT_CONTEXT_WINDOW


def estimate_tokens(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


# ── Turn dataclass ───────────────────────────────────────────────────

@dataclass
class DiagnosticsTurn:
    timestamp: str
    agent: str
    model: str
    provider: str
    status: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    context_window_max: int
    system_prompt_est_tokens: int
    context_provider_est_tokens: int
    chat_history_est_tokens: int
    output_reserved_tokens: int
    messages_count: int
    debug_logs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ── Thread-safe in-memory store via st.cache_resource ────────────────

class _DiagnosticsStore:
    """Process-wide singleton holding all diagnostics data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: dict[str, dict[str, list]] = {}

    def _ensure_agent(self, agent: str) -> dict[str, list]:
        return self._agents.setdefault(
            agent, {"turns": [], "app_logs": [], "errors": []}
        )

    def record_turn(self, turn: DiagnosticsTurn) -> None:
        with self._lock:
            agent_data = self._ensure_agent(turn.agent)
            agent_data["turns"].append(asdict(turn))
            if len(agent_data["turns"]) > MAX_TURNS_PER_AGENT:
                agent_data["turns"] = agent_data["turns"][-MAX_TURNS_PER_AGENT:]

    def record_log(self, agent: str, message: str, level: str = "INFO") -> None:
        with self._lock:
            agent_data = self._ensure_agent(agent)
            entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}"
            agent_data["app_logs"].append(entry)
            if level in {"ERROR", "WARNING"}:
                agent_data["errors"].append(entry)
            if len(agent_data["app_logs"]) > MAX_LOGS_PER_AGENT:
                agent_data["app_logs"] = agent_data["app_logs"][-MAX_LOGS_PER_AGENT:]
            if len(agent_data["errors"]) > MAX_LOGS_PER_AGENT:
                agent_data["errors"] = agent_data["errors"][-MAX_LOGS_PER_AGENT:]

    def get_agent_diagnostics(self, agent: str) -> dict[str, Any]:
        with self._lock:
            data = self._agents.get(agent)
            if data is None:
                return {"turns": [], "app_logs": [], "errors": []}
            # Return a shallow copy so readers don't hold the lock.
            return {k: list(v) for k, v in data.items()}

    def get_all_agents(self) -> list[str]:
        with self._lock:
            return list(self._agents.keys())

    def clear_agent(self, agent: str) -> None:
        with self._lock:
            if agent in self._agents:
                self._agents[agent] = {"turns": [], "app_logs": [], "errors": []}


@st.cache_resource
def _get_store() -> _DiagnosticsStore:
    return _DiagnosticsStore()


# ── Public API (unchanged signatures) ────────────────────────────────

def record_turn(turn: DiagnosticsTurn) -> None:
    _get_store().record_turn(turn)


def record_log(agent: str, message: str, level: str = "INFO") -> None:
    _get_store().record_log(agent, message, level)


def get_agent_diagnostics(agent: str) -> dict[str, Any]:
    return _get_store().get_agent_diagnostics(agent)


def get_all_agents() -> list[str]:
    return _get_store().get_all_agents()


def clear_agent_diagnostics(agent: str) -> None:
    _get_store().clear_agent(agent)
