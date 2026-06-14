from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException, Request, status

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    GoogleOAuthPassthroughConfig,
    GoogleOAuthTokenVerificationResult,
    LitellmUserRoles,
    UserAPIKeyAuth,
    hash_token,
)

_BEARER_PREFIX = "bearer"


def _extract_bearer_token(header_value: Optional[str]) -> Optional[str]:
    """Extract a bearer token from an Authorization header."""

    if header_value is None:
        return None

    parts = header_value.split(" ", 1)
    if len(parts) != 2:
        return None

    scheme, token = parts[0].strip(), parts[1].strip()
    if scheme.lower() != _BEARER_PREFIX:
        return None

    return token or None


class GoogleOAuthTokenVerifier:
    """Lightweight helper for validating Google OAuth access/id tokens."""

    _instance: "GoogleOAuthTokenVerifier" | None = None

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._cache: Dict[str, tuple[float, GoogleOAuthTokenVerificationResult]] = {}

    @classmethod
    def instance(cls) -> "GoogleOAuthTokenVerifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def verify(
        self,
        *,
        token: str,
        config: GoogleOAuthPassthroughConfig,
    ) -> GoogleOAuthTokenVerificationResult:
        """Validate a Google OAuth token via the token info endpoint."""

        if config.cache_ttl_seconds > 0:
            cached = self._cache.get(token)
            now = time.time()
            if cached and cached[0] > now:
                return cached[1]

        async with self._lock:
            if config.cache_ttl_seconds > 0:
                cached = self._cache.get(token)
                now = time.time()
                if cached and cached[0] > now:
                    return cached[1]

            claims = await self._fetch_token_info(token=token, config=config)
            result = self._transform_claims(claims=claims)
            self._validate_result(result=result, config=config)

            ttl_seconds = config.cache_ttl_seconds
            if ttl_seconds > 0 and result.expires_in is not None:
                ttl_seconds = min(ttl_seconds, max(result.expires_in, 0))

            if ttl_seconds > 0:
                self._cache[token] = (time.time() + ttl_seconds, result)

            return result

    async def _fetch_token_info(
        self,
        *,
        token: str,
        config: GoogleOAuthPassthroughConfig,
    ) -> Dict[str, Any]:
        """Call Google's tokeninfo endpoint to validate a token."""

        params_options = ("access_token", "id_token")
        last_exception: Optional[Exception] = None

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            for param_name in params_options:
                try:
                    response = await client.get(
                        config.token_info_endpoint,
                        params={param_name: token},
                        headers={"Accept": "application/json"},
                    )
                except httpx.HTTPError as exc:  # pragma: no cover - network failure
                    verbose_proxy_logger.error(
                        "google_oauth_passthrough: tokeninfo request failed", exc_info=True
                    )
                    last_exception = exc
                    continue

                if response.status_code == status.HTTP_200_OK:
                    return response.json()

                # Store exception to include in downstream error message
                try:
                    error_detail = response.json()
                except Exception:  # pragma: no cover - invalid json
                    error_detail = response.text
                last_exception = HTTPException(
                    status_code=response.status_code,
                    detail={"error": error_detail},
                )

        if isinstance(last_exception, HTTPException):
            raise last_exception

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid Google OAuth token"},
        )

    def _transform_claims(
        self,
        *,
        claims: Dict[str, Any],
    ) -> GoogleOAuthTokenVerificationResult:
        scope_value = claims.get("scope", "")
        scopes = [scope.strip() for scope in scope_value.split() if scope.strip()]

        expires_in_value = claims.get("expires_in")
        expires_in: Optional[int]
        if expires_in_value is None:
            expires_in = None
        else:
            try:
                expires_in = int(expires_in_value)
            except (TypeError, ValueError):
                expires_in = None

        audience = (
            claims.get("audience")
            or claims.get("aud")
            or claims.get("issued_to")
            or claims.get("client_id")
        )

        client_id = claims.get("aud") or claims.get("client_id") or claims.get("issued_to")

        subject = claims.get("sub") or claims.get("user_id")

        token_type = "access_token" if "scope" in claims else "id_token"

        return GoogleOAuthTokenVerificationResult(
            token_type=token_type,
            audience=audience,
            client_id=client_id,
            subject=subject,
            email=claims.get("email"),
            scopes=scopes,
            expires_in=expires_in,
            issued_to=claims.get("issued_to"),
            raw_claims={k: v for k, v in claims.items() if k != "access_token"},
        )

    def _validate_result(
        self,
        *,
        result: GoogleOAuthTokenVerificationResult,
        config: GoogleOAuthPassthroughConfig,
    ) -> None:
        if config.allowed_audiences and result.audience not in config.allowed_audiences:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Google OAuth audience not permitted"},
            )

        if config.allowed_client_ids and result.client_id not in config.allowed_client_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Google OAuth client not permitted"},
            )

        if config.required_scopes:
            missing_scopes = set(config.required_scopes) - set(result.scopes)
            if missing_scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"error": "Missing required Google OAuth scopes", "missing": sorted(missing_scopes)},
                )

        if config.allowed_email_domains:
            email = result.email
            if email is None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"error": "Google OAuth token missing email claim"},
                )
            domain = email.split("@")[-1].lower()
            if domain not in config.allowed_email_domains:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"error": "Google OAuth email domain not allowed"},
                )

        if config.allowed_emails and result.email is not None:
            if result.email.lower() not in config.allowed_emails:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"error": "Google OAuth email not allowed"},
                )

    def clear_cache(self) -> None:
        self._cache.clear()


async def maybe_authenticate_google_oauth(
    *,
    request: Request,
    general_settings: Dict[str, Any],
) -> Optional[UserAPIKeyAuth]:
    """
    Authenticate the incoming request using a Google OAuth bearer token if configured.

    Returns a ``UserAPIKeyAuth`` object when Google OAuth passthrough is enabled and the
    request carries a valid bearer token. Otherwise returns ``None`` so the caller can
    fall back to the default LiteLLM authentication flow.
    """

    config_raw = general_settings.get("google_oauth_passthrough")
    if config_raw is None:
        return None

    if isinstance(config_raw, GoogleOAuthPassthroughConfig):
        config = config_raw
    elif isinstance(config_raw, dict):
        try:
            config = GoogleOAuthPassthroughConfig.model_validate(config_raw)
        except Exception as exc:
            verbose_proxy_logger.error(
                "google_oauth_passthrough: invalid configuration", exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Invalid google_oauth_passthrough configuration", "detail": str(exc)},
            ) from exc
    else:
        return None

    if not config.enabled:
        return None

    authorization_header = request.headers.get("authorization")
    token = _extract_bearer_token(authorization_header)
    if token is None:
        return None

    verifier = GoogleOAuthTokenVerifier.instance()
    verification_result = await verifier.verify(token=token, config=config)

    hashed_token = hash_token(token)
    identifier = (
        verification_result.email
        or verification_result.subject
        or verification_result.client_id
        or verification_result.audience
        or hashed_token
    )

    key_alias_prefix = config.key_alias_prefix or "google-oauth"
    key_alias = f"{key_alias_prefix}:{identifier}"

    metadata = dict(config.metadata)
    metadata.update(
        {
            "google_oauth": {
                "audience": verification_result.audience,
                "client_id": verification_result.client_id,
                "scopes": verification_result.scopes,
                "email": verification_result.email,
                "token_type": verification_result.token_type,
            }
        }
    )

    user_api_key = UserAPIKeyAuth(
        api_key=f"google-oauth::{hashed_token}",
        key_alias=key_alias,
        user_id=str(identifier),
        user_email=verification_result.email,
        metadata=metadata,
    )

    if config.models is not None:
        user_api_key.models = config.models
    if config.allowed_routes is not None:
        user_api_key.allowed_routes = config.allowed_routes
    if config.default_team_id is not None:
        user_api_key.team_id = config.default_team_id
    if config.default_team_alias is not None:
        user_api_key.team_alias = config.default_team_alias
    if config.default_org_id is not None:
        user_api_key.org_id = config.default_org_id
    if config.default_user_role is not None:
        user_api_key.user_role = (
            config.default_user_role
            if isinstance(config.default_user_role, LitellmUserRoles)
            else LitellmUserRoles(config.default_user_role)
        )

    return user_api_key
