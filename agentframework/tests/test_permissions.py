from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure the agentframework root is on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chatbot.permissions import (  # noqa: E402
    can_use_agent,
    has_admin_access,
    has_diagnostics_access,
    list_allowed_agents,
)
from config.models import AppConfig  # noqa: E402


def _cfg() -> AppConfig:
    return AppConfig(
        rbac={
            "enabled": True,
            "super_admin_emails": ["rsx79@hotmail.com"],
            "default_authenticated_agents": ["Fitness Nutrition"],
            "user_permissions": [
                {
                    "email": "alice@example.com",
                    "admin": False,
                    "diagnostics": False,
                    "allowed_agents": ["General Chat Assistant"],
                },
                {
                    "email": "diag@example.com",
                    "admin": False,
                    "diagnostics": True,
                    "allowed_agents": [],
                },
            ],
        }
    )


def _session(email: str | None, *, user_id: str = "u-1") -> SimpleNamespace | None:
    if email is None:
        return None
    return SimpleNamespace(user_id=user_id, email=email, is_demo=False)


def test_super_admin_gets_full_access() -> None:
    cfg = _cfg()
    session = _session("RSX79@HOTMAIL.COM")
    all_agents = ["General Chat Assistant", "Fitness Nutrition", "Agent1 Demo"]

    assert has_admin_access(session, cfg=cfg)
    assert has_diagnostics_access(session, cfg=cfg)
    assert list_allowed_agents(session, all_agents, cfg=cfg) == all_agents


def test_authenticated_user_gets_fitness_default() -> None:
    cfg = _cfg()
    session = _session("bob@example.com")
    all_agents = ["General Chat Assistant", "Fitness Nutrition", "Agent1 Demo"]

    assert not has_admin_access(session, cfg=cfg)
    assert list_allowed_agents(session, all_agents, cfg=cfg) == ["Fitness Nutrition"]


def test_user_allowed_agents_are_added() -> None:
    cfg = _cfg()
    session = _session("alice@example.com")
    all_agents = ["General Chat Assistant", "Fitness Nutrition", "Agent1 Demo"]

    assert list_allowed_agents(session, all_agents, cfg=cfg) == ["General Chat Assistant", "Fitness Nutrition"]
    assert can_use_agent(session, "General Chat Assistant", all_agents, cfg=cfg)
    assert not can_use_agent(session, "Agent1 Demo", all_agents, cfg=cfg)


def test_diagnostics_permission_can_be_granted_without_admin() -> None:
    cfg = _cfg()
    session = _session("diag@example.com")

    assert not has_admin_access(session, cfg=cfg)
    assert has_diagnostics_access(session, cfg=cfg)


def test_unauthenticated_is_limited_to_fitness() -> None:
    cfg = _cfg()
    all_agents = ["General Chat Assistant", "Fitness Nutrition", "Agent1 Demo"]

    assert list_allowed_agents(None, all_agents, cfg=cfg) == ["Fitness Nutrition"]
