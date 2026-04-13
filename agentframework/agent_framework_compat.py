from __future__ import annotations

from importlib import metadata

import agent_framework as _agent_framework


def _ensure_agent_framework_version() -> None:
    if hasattr(_agent_framework, "__version__"):
        return
    try:
        _agent_framework.__version__ = metadata.version("agent-framework")
    except metadata.PackageNotFoundError:
        _agent_framework.__version__ = "0"


_ensure_agent_framework_version()

try:
    from agent_framework import (
        Agent,
        AgentResponse,
        ChatOptions,
        ChatResponse,
        ChatResponseUpdate,
        Content,
        ContextProvider,
        FinishReason,
        Message,
        UsageDetails,
    )
except ImportError:
    from agent_framework._agents import Agent
    from agent_framework._sessions import ContextProvider
    from agent_framework._types import (
        AgentResponse,
        ChatOptions,
        ChatResponse,
        ChatResponseUpdate,
        Content,
        FinishReason,
        Message,
        UsageDetails,
    )


__all__ = [
    "Agent",
    "AgentResponse",
    "ChatOptions",
    "ChatResponse",
    "ChatResponseUpdate",
    "Content",
    "ContextProvider",
    "FinishReason",
    "Message",
    "UsageDetails",
]