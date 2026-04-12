"""Streamlit authentication UI and session management.

Handles:
- OAuth/OIDC login flow (redirect URL, token exchange)
- Demo mode gate (anonymous access)
- Session state initialization with authenticated user
- Authorization guards on data operations
- UI messaging for current auth state
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import Any

import streamlit as st

from chatbot.auth_identity import IdentityClaimNormalizer, generate_app_user_id
from config import get_config
from config.secrets import ResolvedOAuthProvider, resolve_env, resolve_external_identities_secrets
from fitness_memory import get_fitness_repository, utc_now_iso

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthenticatedSession:
    """An authenticated user session."""

    user_id: str
    display_name: str | None
    email: str | None
    auth_provider: str  # 'oauth' or 'demo'
    is_demo: bool


def _init_auth_session_state() -> None:
    """Initialize session state keys for authentication.

    Called once per Streamlit app startup.
    """
    if "auth_session" not in st.session_state:
        st.session_state["auth_session"] = None

    if "auth_state_code" not in st.session_state:
        st.session_state.auth_state_code = None  # OAuth state parameter for CSRF protection

    if "fitness_active_user_id" not in st.session_state:
        st.session_state.fitness_active_user_id = None


def get_current_auth_session() -> AuthenticatedSession | None:
    """Get the current authenticated session from Streamlit session_state.

    For demo sessions, auto-refresh if the configured demo_user_id has changed.
    """
    _init_auth_session_state()
    session = st.session_state.auth_session
    if session and session.is_demo:
        from config import get_config
        cfg_demo = get_config().demo_mode
        if session.user_id != cfg_demo.demo_user_id or session.display_name != cfg_demo.demo_user_name:
            session = AuthenticatedSession(
                user_id=cfg_demo.demo_user_id,
                display_name=cfg_demo.demo_user_name,
                email=None,
                auth_provider="demo",
                is_demo=True,
            )
            set_auth_session(session)
    return session


def set_auth_session(session: AuthenticatedSession | None) -> None:
    """Set the current authenticated session in Streamlit session_state.

    Also updates fitness_active_user_id to maintain backward compatibility.
    """
    _init_auth_session_state()
    st.session_state.auth_session = session
    if session:
        st.session_state.fitness_active_user_id = session.user_id
    else:
        st.session_state.fitness_active_user_id = None


def require_auth() -> str:
    """Assert user is authenticated or in demo mode; return user_id.

    This is a guard function to ensure authorized data access.
    Call this before any operation that needs a user context.

    Returns:
        user_id of the current authenticated user

    Raises:
        RuntimeError: If no authenticated session exists
    """
    session = get_current_auth_session()
    if not session:
        raise RuntimeError("No authenticated session. User must log in first.")
    return session.user_id


def render_auth_state_banner() -> None:
    """Render a banner showing current authentication state.

    For authenticated users: shows provider and email
    For demo users: shows demo banner with clear messaging
    """
    session = get_current_auth_session()
    if not session:
        return

    if session.is_demo:
        st.warning(
            f"🎭 **Demo Mode** – You are using the app anonymously as '{session.display_name}'. "
            f"Data is isolated to this demo user only. "
            f"[Log in](/) to access your personal fitness data."
        )
    else:
        provider_label = session.auth_provider.replace("_", " ").title()
        email_text = f" ({session.email})" if session.email else ""
        st.info(
            f"✅ **Logged in** via {provider_label}{email_text} as {session.display_name}. "
            f"Your data is private to you."
        )


def render_login_gate() -> None:
    """Render the OAuth/demo mode login gate.

    Displays:
    - OAuth provider buttons (Microsoft, Google, Twitter) if enabled
    - Demo mode option (if enabled and user not logged in)
    - Or: message if auth disabled and demo disabled
    """
    st.markdown("---")
    st.markdown("### Sign in or Continue as Demo")

    config = get_config()
    oauth_enabled = config.external_identities.enabled
    demo_enabled = config.demo_mode.enabled

    if oauth_enabled:
        providers_by_name = {p.provider_name: p for p in config.external_identities.providers if p.enabled}
        resolved_secrets = resolve_external_identities_secrets(config.external_identities)

        _process_oauth_callback(resolved_secrets)

        # Render OAuth provider buttons
        col1, col2, col3 = st.columns(3)

        if "microsoft" in providers_by_name:
            with col1:
                if st.button("🔵 Login with Microsoft"):
                    _initiate_oauth_flow("microsoft", resolved_secrets.get("microsoft"))

        if "google" in providers_by_name:
            with col2:
                if st.button("🔴 Login with Google"):
                    _initiate_oauth_flow("google", resolved_secrets.get("google"))

        if "twitter" in providers_by_name:
            with col3:
                if st.button("𝕏 Login with Twitter"):
                    _initiate_oauth_flow("twitter", resolved_secrets.get("twitter"))

    if demo_enabled:
        st.markdown("---")
        if st.button(f"🎭 Continue as Demo ({config.demo_mode.demo_user_name})"):
            _initiate_demo_mode(config.demo_mode)

    if not oauth_enabled and not demo_enabled:
        st.error("Authentication is disabled. No login methods available.")


def _initiate_oauth_flow(provider_name: str, resolved_provider: Any) -> None:
    """Initiate an OAuth flow for the given provider."""
    if not resolved_provider:
        st.error(f"Provider '{provider_name}' is not properly configured.")
        return

    logger.info(f"OAuth flow initiated for provider: {provider_name}")
    if provider_name != "microsoft":
        st.info(f"{provider_name.title()} login is not wired yet in this build.")
        return

    try:
        import msal
    except Exception:
        st.error("Microsoft login requires the msal package. Install dependencies and retry.")
        return

    if not resolved_provider.client_id or not resolved_provider.client_secret:
        st.error("Microsoft provider is missing client ID or client secret in environment variables.")
        return

    scopes = _scope_list(resolved_provider.scope)
    state = secrets.token_urlsafe(24)
    st.session_state["auth_state_code"] = state
    st.session_state["auth_provider"] = provider_name

    callback_url = _callback_url()
    authority_url = _normalize_microsoft_authority(resolved_provider.authority_url)
    try:
        app = msal.ConfidentialClientApplication(
            client_id=resolved_provider.client_id,
            client_credential=resolved_provider.client_secret,
            authority=authority_url,
        )
    except ValueError as exc:
        st.error(
            "Microsoft authority URL is invalid. Use https://login.microsoftonline.com/common "
            "or a specific tenant authority URL."
        )
        logger.exception("Invalid Microsoft authority URL", exc_info=exc)
        return
    auth_url = app.get_authorization_request_url(
        scopes=scopes,
        redirect_uri=callback_url,
        state=state,
        prompt="select_account",
    )

    st.link_button("Continue with Microsoft", auth_url, use_container_width=True)
    st.caption("After sign-in, you will be redirected back and your session will be established automatically.")


def _process_oauth_callback(resolved_providers: dict[str, ResolvedOAuthProvider]) -> None:
    """Handle OAuth callback query params and establish app session."""
    params = st.query_params
    auth_code = str(params.get("code", "") or "").strip()
    if not auth_code:
        return

    state = str(params.get("state", "") or "").strip()
    expected_state = str(st.session_state.get("auth_state_code") or "").strip()
    provider_name = str(st.session_state.get("auth_provider") or "microsoft").strip().lower()

    # Streamlit can start a fresh session after external OAuth redirects, which can
    # drop session_state and the originally stored CSRF state token. In that case,
    # proceed for local/dev flows, but still reject explicit mismatches.
    if expected_state and state != expected_state:
        st.error("Login state validation failed. Please try signing in again.")
        _clear_oauth_query_params()
        return

    if not expected_state:
        logger.warning("OAuth callback received without stored state; proceeding with relaxed validation.")

    provider = resolved_providers.get(provider_name)
    if provider_name != "microsoft" or provider is None:
        st.error("Unsupported OAuth callback provider.")
        _clear_oauth_query_params()
        return

    try:
        _complete_microsoft_login(auth_code=auth_code, provider=provider)
        _clear_oauth_query_params()
        st.rerun()
    except Exception as exc:
        logger.exception("OAuth callback processing failed")
        st.error(f"Microsoft sign-in failed: {exc}")
        _clear_oauth_query_params()


def _complete_microsoft_login(*, auth_code: str, provider: ResolvedOAuthProvider) -> None:
    """Exchange auth code for token, normalize claims, and bind session user."""
    import msal

    callback_url = _callback_url()
    authority_url = _normalize_microsoft_authority(provider.authority_url)
    app = msal.ConfidentialClientApplication(
        client_id=provider.client_id,
        client_credential=provider.client_secret,
        authority=authority_url,
    )
    token_result = app.acquire_token_by_authorization_code(
        code=auth_code,
        scopes=_scope_list(provider.scope),
        redirect_uri=callback_url,
    )

    if "id_token_claims" not in token_result:
        error_text = token_result.get("error_description") or token_result.get("error") or "Unknown token exchange error"
        raise RuntimeError(error_text)

    normalized = IdentityClaimNormalizer.normalize("microsoft", token_result["id_token_claims"])
    repo = get_fitness_repository()
    existing_user_id = repo.get_user_by_provider_subject("microsoft", normalized.subject_id)

    display_name = (normalized.name or normalized.email or "Microsoft User").strip()
    if existing_user_id:
        user_id = existing_user_id
        repo.update_provider_login_metadata(
            user_id=user_id,
            last_login_at=utc_now_iso(),
            email=normalized.email,
            email_verified=normalized.email_verified,
        )
    else:
        user_id = generate_app_user_id()
        repo.create_user_with_provider_identity(
            user_id=user_id,
            name=display_name,
            auth_provider="microsoft",
            provider_subject_id=normalized.subject_id,
            email=normalized.email,
            email_verified=normalized.email_verified,
        )

    set_auth_session(
        AuthenticatedSession(
            user_id=user_id,
            display_name=display_name,
            email=normalized.email,
            auth_provider="microsoft",
            is_demo=False,
        )
    )
    st.session_state["fitness_user_name"] = user_id


def _callback_url() -> str:
    host = resolve_env("OAUTH_CALLBACK_HOST", "http://localhost:8501").strip() or "http://localhost:8501"
    return host.rstrip("/")


def _scope_list(scope_text: str) -> list[str]:
    scopes = [part.strip() for part in (scope_text or "").split() if part.strip()]
    # MSAL auto-handles reserved OIDC scopes and rejects them when passed explicitly.
    reserved = {"openid", "profile", "offline_access"}
    effective_scopes = [scope for scope in scopes if scope not in reserved]
    return effective_scopes or ["User.Read"]


def _clear_oauth_query_params() -> None:
    for key in ("code", "state", "session_state", "error", "error_description"):
        if key in st.query_params:
            del st.query_params[key]
    st.session_state["auth_state_code"] = None
    st.session_state["auth_provider"] = None


def _normalize_microsoft_authority(authority_url: str) -> str:
    """Normalize Microsoft authority URL to MSAL-compatible format.

    MSAL expects authorities like:
    - https://login.microsoftonline.com/common
    - https://login.microsoftonline.com/<tenant-id>
    """
    authority = (authority_url or "").strip().rstrip("/")
    if authority.endswith("/v2.0"):
        authority = authority[: -len("/v2.0")]
    return authority


def _initiate_demo_mode(demo_config: Any) -> None:
    """Initiate demo mode session for the anonymous user.

    Creates an AuthenticatedSession bound to the demo user_id.
    """
    demo_session = AuthenticatedSession(
        user_id=demo_config.demo_user_id,
        display_name=demo_config.demo_user_name,
        email=None,
        auth_provider="demo",
        is_demo=True,
    )
    set_auth_session(demo_session)
    st.rerun()


def enforce_auth_for_operation(
    user_id_from_session: str,
    user_id_from_operation: str | None = None,
) -> None:
    """Enforce that a data operation targets only the authenticated user.

    This prevents cross-user data access exploits.

    Args:
        user_id_from_session: The authenticated user's ID
        user_id_from_operation: The user_id being targeted by the operation (if None, uses session)

    Raises:
        PermissionError: If operation targets a different user
    """
    if user_id_from_operation is None:
        # Default to session user
        return

    if user_id_from_operation != user_id_from_session:
        logger.warning(
            f"Authorization violation: user {user_id_from_session} attempted "
            f"to access data for user {user_id_from_operation}"
        )
        raise PermissionError(
            f"You do not have permission to access data for user '{user_id_from_operation}'. "
            f"You are logged in as '{user_id_from_session}'."
        )


def logout() -> None:
    """Clear the current authentication session."""
    set_auth_session(None)
    st.rerun()
