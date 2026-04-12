"""Unit tests for OAuth/OIDC identity handling and claim normalization."""

import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.auth_identity import (
    IdentityClaimNormalizer,
    ProviderIdentityClaim,
    generate_app_user_id,
)


class TestIdentityClaimNormalizer:
    """Tests for provider claim normalization."""

    def test_normalize_microsoft_minimal(self):
        """Test normalization of minimal Microsoft ID token."""
        claims = {
            "oid": "12345-abcd",
            "preferred_username": "user@example.com",
        }
        result = IdentityClaimNormalizer.normalize_microsoft(claims)

        assert result.provider_name == "microsoft"
        assert result.subject_id == "12345-abcd"
        assert result.email is None
        assert not result.email_verified
        assert result.name == "user@example.com"

    def test_normalize_microsoft_full(self):
        """Test normalization of complete Microsoft ID token."""
        claims = {
            "oid": "12345-abcd",
            "email": "john.doe@example.com",
            "email_verified": True,
            "given_name": "John",
            "family_name": "Doe",
            "name": "John Doe",
        }
        result = IdentityClaimNormalizer.normalize_microsoft(claims)

        assert result.provider_name == "microsoft"
        assert result.subject_id == "12345-abcd"
        assert result.email == "john.doe@example.com"
        assert result.email_verified
        assert result.given_name == "John"
        assert result.family_name == "Doe"
        assert result.name == "John Doe"

    def test_normalize_microsoft_missing_oid(self):
        """Test that missing OID raises error."""
        claims = {"email": "user@example.com"}
        with pytest.raises(ValueError, match="oid"):
            IdentityClaimNormalizer.normalize_microsoft(claims)

    def test_normalize_google_minimal(self):
        """Test normalization of minimal Google ID token."""
        claims = {
            "sub": "google-user-id-123",
            "email": "user@gmail.com",
        }
        result = IdentityClaimNormalizer.normalize_google(claims)

        assert result.provider_name == "google"
        assert result.subject_id == "google-user-id-123"
        assert result.email == "user@gmail.com"
        assert result.email_verified  # Google verifies by default

    def test_normalize_google_full(self):
        """Test normalization of complete Google ID token."""
        claims = {
            "sub": "google-user-id-123",
            "email": "john.doe@gmail.com",
            "email_verified": True,
            "given_name": "John",
            "family_name": "Doe",
            "name": "John Doe",
        }
        result = IdentityClaimNormalizer.normalize_google(claims)

        assert result.provider_name == "google"
        assert result.subject_id == "google-user-id-123"
        assert result.email == "john.doe@gmail.com"
        assert result.email_verified
        assert result.given_name == "John"
        assert result.family_name == "Doe"
        assert result.name == "John Doe"

    def test_normalize_google_missing_sub(self):
        """Test that missing sub raises error."""
        claims = {"email": "user@gmail.com"}
        with pytest.raises(ValueError, match="sub"):
            IdentityClaimNormalizer.normalize_google(claims)

    def test_normalize_twitter_minimal(self):
        """Test normalization of minimal Twitter/X ID token."""
        claims = {
            "sub": "123456789",  # Twitter user ID
        }
        result = IdentityClaimNormalizer.normalize_twitter(claims)

        assert result.provider_name == "twitter"
        assert result.subject_id == "123456789"
        assert result.email is None
        assert not result.email_verified
        assert result.name == "123456789"  # Fallback to sub

    def test_normalize_twitter_with_username(self):
        """Test normalization of Twitter with username."""
        claims = {
            "sub": "123456789",
            "username": "johndoe",
            "name": "John Doe",
        }
        result = IdentityClaimNormalizer.normalize_twitter(claims)

        assert result.provider_name == "twitter"
        assert result.subject_id == "123456789"
        assert result.name == "John Doe"
        assert result.email is None

    def test_normalize_twitter_with_email(self):
        """Test normalization of Twitter with email (if permission granted)."""
        claims = {
            "sub": "123456789",
            "username": "johndoe",
            "email": "john@example.com",
            "email_verified": True,
        }
        result = IdentityClaimNormalizer.normalize_twitter(claims)

        assert result.provider_name == "twitter"
        assert result.subject_id == "123456789"
        assert result.email == "john@example.com"
        assert result.email_verified

    def test_normalize_twitter_missing_sub(self):
        """Test that missing sub raises error."""
        claims = {"username": "johndoe"}
        with pytest.raises(ValueError, match="sub"):
            IdentityClaimNormalizer.normalize_twitter(claims)

    def test_normalize_unsupported_provider(self):
        """Test that unsupported provider raises error."""
        with pytest.raises(ValueError, match="Unsupported"):
            IdentityClaimNormalizer.normalize("unsupported_provider", {})

    def test_normalize_dispatch_microsoft(self):
        """Test dispatch to Microsoft normalizer."""
        claims = {"oid": "test-oid"}
        result = IdentityClaimNormalizer.normalize("microsoft", claims)
        assert result.provider_name == "microsoft"
        assert result.subject_id == "test-oid"

    def test_normalize_dispatch_google(self):
        """Test dispatch to Google normalizer."""
        claims = {"sub": "test-sub"}
        result = IdentityClaimNormalizer.normalize("google", claims)
        assert result.provider_name == "google"
        assert result.subject_id == "test-sub"

    def test_normalize_dispatch_twitter(self):
        """Test dispatch to Twitter normalizer."""
        claims = {"sub": "test-sub"}
        result = IdentityClaimNormalizer.normalize("twitter", claims)
        assert result.provider_name == "twitter"
        assert result.subject_id == "test-sub"


class TestGenerateAppUserId:
    """Tests for app user ID generation."""

    def test_generate_app_user_id_format(self):
        """Test that generated user IDs have correct format."""
        user_id = generate_app_user_id()
        assert user_id.startswith("u_")
        assert len(user_id) == 10  # u_ + 8 chars

    def test_generate_app_user_id_uniqueness(self):
        """Test that generated user IDs are unique."""
        user_ids = {generate_app_user_id() for _ in range(100)}
        assert len(user_ids) == 100  # All unique


class TestProviderIdentityClaim:
    """Tests for ProviderIdentityClaim data class."""

    def test_claim_creation_minimal(self):
        """Test creating minimal claim."""
        claim = ProviderIdentityClaim(
            provider_name="google",
            subject_id="sub-123",
        )
        assert claim.provider_name == "google"
        assert claim.subject_id == "sub-123"
        assert claim.email is None
        assert not claim.email_verified

    def test_claim_creation_full(self):
        """Test creating full claim."""
        claim = ProviderIdentityClaim(
            provider_name="microsoft",
            subject_id="oid-123",
            email="john@example.com",
            email_verified=True,
            given_name="John",
            family_name="Doe",
            name="John Doe",
        )
        assert claim.provider_name == "microsoft"
        assert claim.subject_id == "oid-123"
        assert claim.email == "john@example.com"
        assert claim.email_verified
        assert claim.given_name == "John"
        assert claim.family_name == "Doe"
        assert claim.name == "John Doe"

    def test_claim_is_frozen(self):
        """Test that claim is immutable."""
        claim = ProviderIdentityClaim(
            provider_name="google",
            subject_id="sub-123",
        )
        with pytest.raises(AttributeError):
            claim.provider_name = "microsoft"
