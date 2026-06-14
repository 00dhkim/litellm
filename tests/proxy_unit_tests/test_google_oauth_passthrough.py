import asyncio
from typing import Any, Dict

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from unittest.mock import AsyncMock

from litellm.proxy._types import (
    GoogleOAuthPassthroughConfig,
    GoogleOAuthTokenVerificationResult,
    UserAPIKeyAuth,
    hash_token,
)
from litellm.proxy.auth.google_oauth import (
    GoogleOAuthTokenVerifier,
    maybe_authenticate_google_oauth,
)


def test_google_oauth_verifier_caches(monkeypatch):
    async def _run():
        verifier = GoogleOAuthTokenVerifier.instance()
        verifier.clear_cache()

        claims: Dict[str, Any] = {
            "audience": "test-client",
            "scope": "scope.a scope.b",
            "expires_in": "3600",
            "email": "user@example.com",
        }

        fetch_mock = AsyncMock(return_value=claims)
        monkeypatch.setattr(verifier, "_fetch_token_info", fetch_mock)

        config = GoogleOAuthPassthroughConfig(
            enabled=True,
            allowed_audiences=["test-client"],
            cache_ttl_seconds=600,
        )

        token = "ya29.test-token"
        result_first = await verifier.verify(token=token, config=config)
        result_second = await verifier.verify(token=token, config=config)

        assert result_first.audience == "test-client"
        assert result_second.scopes == ["scope.a", "scope.b"]
        assert fetch_mock.call_count == 1

    asyncio.run(_run())


def test_google_oauth_verifier_missing_scope(monkeypatch):
    async def _run():
        verifier = GoogleOAuthTokenVerifier.instance()
        verifier.clear_cache()

        claims: Dict[str, Any] = {
            "audience": "another-client",
            "scope": "scope.base",
            "expires_in": "3600",
        }

        monkeypatch.setattr(verifier, "_fetch_token_info", AsyncMock(return_value=claims))

        config = GoogleOAuthPassthroughConfig(
            enabled=True,
            allowed_audiences=["another-client"],
            required_scopes=["scope.required"],
        )

        with pytest.raises(HTTPException) as exc_info:
            await verifier.verify(token="ya29.other", config=config)

        assert exc_info.value.status_code == 403
        assert "Missing required Google OAuth scopes" in str(exc_info.value.detail)

    asyncio.run(_run())


def test_maybe_authenticate_google_oauth(monkeypatch):
    async def _run():
        token = "ya29.example-token"
        hashed = hash_token(token)

        async def receive():
            return {"type": "http.request"}

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/gemini/test",
            "headers": [(b"authorization", f"Bearer {token}".encode("utf-8"))],
        }
        request = Request(scope, receive)

        config_dict = {
            "enabled": True,
            "allowed_audiences": ["client-123"],
            "key_alias_prefix": "google-oauth",
            "metadata": {"custom": "value"},
        }

        verification_result = GoogleOAuthTokenVerificationResult(
            token_type="access_token",
            audience="client-123",
            client_id="client-123",
            subject="subject-xyz",
            email="user@example.com",
            scopes=["scope.a"],
            expires_in=3600,
            issued_to="client-123",
            raw_claims={},
        )

        class _StubVerifier:
            async def verify(self, token: str, config: GoogleOAuthPassthroughConfig):
                assert token == "ya29.example-token"
                assert config.enabled is True
                return verification_result

            def clear_cache(self):
                pass

        monkeypatch.setattr(
            GoogleOAuthTokenVerifier,
            "instance",
            classmethod(lambda cls: _StubVerifier()),
        )

        general_settings = {"google_oauth_passthrough": config_dict}
        user: UserAPIKeyAuth | None = await maybe_authenticate_google_oauth(
            request=request, general_settings=general_settings
        )

        assert user is not None
        assert user.user_email == "user@example.com"
        assert user.api_key == f"google-oauth::{hashed}"
        assert user.key_alias.startswith("google-oauth:")
        assert user.metadata["google_oauth"]["audience"] == "client-123"
        assert user.metadata["custom"] == "value"

    asyncio.run(_run())
