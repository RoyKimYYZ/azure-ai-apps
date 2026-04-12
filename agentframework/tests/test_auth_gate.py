"""Unit tests for authentication gate and session management."""

import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.auth_gate import (
    AuthenticatedSession,
    enforce_auth_for_operation,
    require_auth,
)


class TestAuthenticatedSession:
    """Tests for AuthenticatedSession data class."""

    def test_authenticated_session_oauth_user(self):
        """Test creating an OAuth authenticated session."""
        session = AuthenticatedSession(
            user_id="u_abc12345",
            display_name="John Doe",
            email="john@example.com",
            auth_provider="microsoft",
            is_demo=False,
        )
        assert session.user_id == "u_abc12345"
        assert session.display_name == "John Doe"
        assert session.email == "john@example.com"
        assert session.auth_provider == "microsoft"
        assert not session.is_demo

    def test_authenticated_session_demo_user(self):
        """Test creating a demo mode session."""
        session = AuthenticatedSession(
            user_id="bob",
            display_name="Demo User",
            email=None,
            auth_provider="demo",
            is_demo=True,
        )
        assert session.user_id == "bob"
        assert session.display_name == "Demo User"
        assert session.email is None
        assert session.is_demo

    def test_authenticated_session_is_frozen(self):
        """Test that session is immutable."""
        session = AuthenticatedSession(
            user_id="u_test",
            display_name="Test",
            email=None,
            auth_provider="test",
            is_demo=False,
        )
        with pytest.raises(AttributeError):
            session.user_id = "u_other"


class TestAuthorizationEnforcement:
    """Tests for authorization guard functions."""

    def test_enforce_auth_same_user(self):
        """Test that enforce_auth succeeds when users match."""
        # Should not raise
        enforce_auth_for_operation(
            user_id_from_session="u_alice",
            user_id_from_operation="u_alice",
        )

    def test_enforce_auth_different_user(self):
        """Test that enforce_auth fails when users differ."""
        with pytest.raises(PermissionError, match="not have permission"):
            enforce_auth_for_operation(
                user_id_from_session="u_alice",
                user_id_from_operation="u_bob",
            )

    def test_enforce_auth_operation_none(self):
        """Test that enforce_auth with None operation_user succeeds."""
        # Should not raise (defaults to session user)
        enforce_auth_for_operation(
            user_id_from_session="u_alice",
            user_id_from_operation=None,
        )

    def test_enforce_auth_demo_user(self):
        """Test authorization for demo user (same logic)."""
        # Demo user is still bound to their user_id
        enforce_auth_for_operation(
            user_id_from_session="bob",
            user_id_from_operation="bob",
        )

    def test_enforce_auth_demo_cross_user(self):
        """Test that demo users also cannot access other user data."""
        with pytest.raises(PermissionError, match="not have permission"):
            enforce_auth_for_operation(
                user_id_from_session="bob",
                user_id_from_operation="alice",
            )


class TestAuthSessionRequirement:
    """Tests for require_auth guard."""

    def test_require_auth_without_session(self):
        """Test that require_auth raises when no session exists."""
        # Note: In actual Streamlit app, this would fail because session_state isn't available
        # This test documents the expected behavior
        with pytest.raises(RuntimeError, match="No authenticated session"):
            require_auth()

    # Note: Full testing of require_auth with Streamlit session_state requires @pytest.mark.streamlit decorator
    # or mocking st.session_state, which is beyond unit test scope.
    # See integration tests for full session state testing.


class TestAuthorizationViolationScenarios:
    """Integration-like tests for common authorization scenarios."""

    def test_unauthorized_cross_user_data_read(self):
        """Scenario: User A tries to read User B's fitness data."""
        user_a_session = "u_alice"
        user_b_data = "u_bob"

        with pytest.raises(PermissionError):
            enforce_auth_for_operation(
                user_id_from_session=user_a_session,
                user_id_from_operation=user_b_data,
            )

    def test_authorized_self_data_access(self):
        """Scenario: User accesses their own data."""
        user_id = "u_alice"

        # Should not raise
        enforce_auth_for_operation(
            user_id_from_session=user_id,
            user_id_from_operation=user_id,
        )

    def test_authorized_default_user_context(self):
        """Scenario: Operation uses session context (no explicit user_id)."""
        user_id = "u_alice"

        # Should not raise when operation_user is None
        enforce_auth_for_operation(
            user_id_from_session=user_id,
            user_id_from_operation=None,
        )
