from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import get_config
from config.models import AppConfig


FITNESS_AGENT_NAME = "Fitness Nutrition"


@dataclass(frozen=True)
class EffectivePermissions:
    is_authenticated: bool
    is_admin: bool
    can_access_diagnostics: bool
    allowed_agents: list[str]


def _normalize_email(value: str | None) -> str:
    return str(value or "").strip().lower()


def _rbac_enabled(cfg: AppConfig) -> bool:
    return bool(getattr(cfg, "rbac", None) and cfg.rbac.enabled)


def _session_email(session: Any) -> str:
    return _normalize_email(getattr(session, "email", None))


def _is_authenticated_session(session: Any) -> bool:
    return session is not None and bool(getattr(session, "user_id", None))


def _find_user_permission(cfg: AppConfig, email: str) -> Any | None:
    if not email:
        return None
    for entry in cfg.rbac.user_permissions:
        if _normalize_email(entry.email) == email:
            return entry
    return None


def is_super_admin(session: Any, cfg: AppConfig | None = None) -> bool:
    cfg = cfg or get_config()
    if not _rbac_enabled(cfg):
        return False
    email = _session_email(session)
    if not email:
        return False
    return email in {_normalize_email(item) for item in cfg.rbac.super_admin_emails}


def has_admin_access(session: Any, cfg: AppConfig | None = None) -> bool:
    cfg = cfg or get_config()
    if not _rbac_enabled(cfg):
        return True
    if not _is_authenticated_session(session):
        return False
    if is_super_admin(session, cfg):
        return True
    email = _session_email(session)
    user_permission = _find_user_permission(cfg, email)
    return bool(user_permission and user_permission.admin)


def has_diagnostics_access(session: Any, cfg: AppConfig | None = None) -> bool:
    cfg = cfg or get_config()
    if not _rbac_enabled(cfg):
        return True
    if not _is_authenticated_session(session):
        return False
    if has_admin_access(session, cfg):
        return True
    email = _session_email(session)
    user_permission = _find_user_permission(cfg, email)
    return bool(user_permission and user_permission.diagnostics)


def list_allowed_agents(session: Any, all_agent_names: list[str], cfg: AppConfig | None = None) -> list[str]:
    cfg = cfg or get_config()
    unique_agents = [name for name in all_agent_names if isinstance(name, str) and name.strip()]
    if not _rbac_enabled(cfg):
        return unique_agents

    if not _is_authenticated_session(session):
        return [name for name in unique_agents if name == FITNESS_AGENT_NAME]

    if has_admin_access(session, cfg):
        return unique_agents

    allowed = set(cfg.rbac.default_authenticated_agents)
    email = _session_email(session)
    user_permission = _find_user_permission(cfg, email)
    if user_permission is not None:
        allowed.update(user_permission.allowed_agents)

    return [name for name in unique_agents if name in allowed]


def can_use_agent(session: Any, agent_name: str, all_agent_names: list[str], cfg: AppConfig | None = None) -> bool:
    return agent_name in list_allowed_agents(session, all_agent_names, cfg=cfg)


def resolve_effective_permissions(session: Any, all_agent_names: list[str], cfg: AppConfig | None = None) -> EffectivePermissions:
    cfg = cfg or get_config()
    is_authenticated = _is_authenticated_session(session)
    admin = has_admin_access(session, cfg)
    diagnostics = has_diagnostics_access(session, cfg)
    allowed_agents = list_allowed_agents(session, all_agent_names, cfg)
    return EffectivePermissions(
        is_authenticated=is_authenticated,
        is_admin=admin,
        can_access_diagnostics=diagnostics,
        allowed_agents=allowed_agents,
    )
