"""OAuth/OIDC identity resolution and user linking service.

Handles:
- Provider claim normalization (Microsoft Entra, Google, Twitter/X)
- User creation/linking on first login
- Canonical app user_id generation from provider identity
- Metadata updates (email, last_login_at, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderIdentityClaim:
    """Normalized identity claim from an OAuth provider.

    Fields follow OIDC standard where applicable.
    """

    provider_name: str  # 'microsoft', 'google', 'twitter'
    subject_id: str  # Unique identifier from provider (e.g., OID, sub)
    email: str | None = None  # Email address (may be None for Twitter)
    email_verified: bool = False  # Whether provider verified the email
    given_name: str | None = None  # First name (if available)
    family_name: str | None = None  # Last name (if available)
    name: str | None = None  # Display name
    raw_claims: dict[str, Any] | None = None  # Raw response for debugging


@dataclass(frozen=True)
class ResolvedIdentity:
    """Result of identity resolution/linking."""

    user_id: str  # Canonical app user_id
    is_new_user: bool  # True if user was auto-created on first login
    provider_name: str
    provider_subject_id: str
    email: str | None
    email_verified: bool
    display_name: str | None


class IdentityRepository(Protocol):
    """Protocol for identity resolution and user operations."""

    def resolve_or_create_by_provider(
        self,
        provider_claim: ProviderIdentityClaim,
    ) -> ResolvedIdentity:
        """Resolve or create user from provider identity claim.

        On first login (new provider identity):
        - Creates new user with canonical user_id (e.g., 'u_<uuid>')
        - Records provider identity (auth_provider, provider_subject_id)
        - Sets email and email_verified from claim

        On subsequent logins:
        - Returns existing user_id
        - Updates last_login_at and email fields

        Args:
            provider_claim: Normalized identity from provider

        Returns:
            ResolvedIdentity with canonical user_id and metadata

        Raises:
            ValueError: If claim data is invalid or incomplete
        """
        ...

    def get_user_by_provider_subject(
        self,
        provider_name: str,
        provider_subject_id: str,
    ) -> str | None:
        """Look up user_id by provider identity.

        Args:
            provider_name: OAuth provider (e.g., 'microsoft')
            provider_subject_id: Subject ID from that provider

        Returns:
            user_id if found, None otherwise
        """
        ...


class IdentityClaimNormalizer:
    """Normalizes OAuth provider claims to standard ProviderIdentityClaim format."""

    @staticmethod
    def normalize_microsoft(claims: dict[str, Any]) -> ProviderIdentityClaim:
        """Normalize Microsoft Entra ID ID token claims.

        ID token fields from Microsoft (https://learn.microsoft.com/en-us/entra/identity-platform/id-tokens):
        - oid: Object ID (user's unique identifier)
        - email: Email address (if available; requires email scope)
        - given_name, family_name: Name components
        - name: Full name (if available)
        """
        subject_id = claims.get("oid")
        if not subject_id:
            raise ValueError("Microsoft ID token missing 'oid' field")

        return ProviderIdentityClaim(
            provider_name="microsoft",
            subject_id=str(subject_id),
            email=claims.get("email"),
            email_verified=claims.get("email_verified", False),
            given_name=claims.get("given_name"),
            family_name=claims.get("family_name"),
            name=claims.get("name") or claims.get("preferred_username"),
            raw_claims=claims,
        )

    @staticmethod
    def normalize_google(claims: dict[str, Any]) -> ProviderIdentityClaim:
        """Normalize Google ID token claims.

        ID token fields from Google (https://developers.google.com/identity/openid-connect/openid-connect):
        - sub: Subject (user's unique identifier)
        - email: Email address (always verified if present in ID token)
        - given_name, family_name: Name components
        - name: Full name
        """
        subject_id = claims.get("sub")
        if not subject_id:
            raise ValueError("Google ID token missing 'sub' field")

        return ProviderIdentityClaim(
            provider_name="google",
            subject_id=str(subject_id),
            email=claims.get("email"),
            email_verified=claims.get("email_verified", True),  # Google verifies by default
            given_name=claims.get("given_name"),
            family_name=claims.get("family_name"),
            name=claims.get("name"),
            raw_claims=claims,
        )

    @staticmethod
    def normalize_twitter(claims: dict[str, Any]) -> ProviderIdentityClaim:
        """Normalize Twitter/X ID token claims.

        NOTE: Twitter/X OAuth may differ from standard OIDC:
        - sub: Subject (user's unique identifier / user_id)
        - username: Twitter username (available in some flows)
        - email: Email (may not be available; requires email:readonly scope)
        - Email verification status is not typically provided

        Fallback behavior:
        - If email unavailable, use username or sub as display_name
        - Assume email not verified unless explicitly stated
        """
        subject_id = claims.get("sub")
        if not subject_id:
            raise ValueError("Twitter/X ID token missing 'sub' field")

        # Prefer email if available; fallback to username
        display_name = claims.get("name") or claims.get("username") or str(subject_id)
        email = claims.get("email")

        return ProviderIdentityClaim(
            provider_name="twitter",
            subject_id=str(subject_id),
            email=email,
            email_verified=claims.get("email_verified", False),
            # Twitter doesn't typically provide given/family names
            name=display_name,
            raw_claims=claims,
        )

    @staticmethod
    def normalize(provider_name: str, claims: dict[str, Any]) -> ProviderIdentityClaim:
        """Normalize claims from any supported provider.

        Args:
            provider_name: OAuth provider name (microsoft, google, twitter)
            claims: ID token claims dict

        Returns:
            ProviderIdentityClaim with normalized fields

        Raises:
            ValueError: If provider not supported or claims invalid
        """
        normalizer = {
            "microsoft": IdentityClaimNormalizer.normalize_microsoft,
            "google": IdentityClaimNormalizer.normalize_google,
            "twitter": IdentityClaimNormalizer.normalize_twitter,
        }.get(provider_name)

        if not normalizer:
            raise ValueError(f"Unsupported OAuth provider: {provider_name}")

        return normalizer(claims)


def generate_app_user_id() -> str:
    """Generate a canonical app user_id for provider-linked users.

    Format: u_<8-char-uuid-prefix>
    This distinguishes provider-linked users from legacy username-based users.
    """
    uid = str(uuid4()).replace("-", "")[:8]
    return f"u_{uid}"
