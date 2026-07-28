"""Shared API dependencies."""

import hashlib
import hmac
import logging
import time
from typing import Annotated
from uuid import UUID

from fastapi import Header, HTTPException, Request, status

from app.core.config import settings

DEV_DEFAULT_USER_ID = UUID("00000000-0000-0000-0000-000000000001")
MIN_PROXY_SECRET_BYTES = 32
logger = logging.getLogger(__name__)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "MySwingProxy"},
    )


def _parse_user_id(raw_user_id: str | None) -> UUID:
    if raw_user_id is None:
        raise _unauthorized("Authenticated user identity is required")
    try:
        return UUID(raw_user_id)
    except ValueError as exc:
        raise _unauthorized("Invalid X-User-Id header") from exc


def _canonical_proxy_message(
    *,
    timestamp: str,
    method: str,
    path: str,
    user_id: UUID,
) -> bytes:
    """Build the request identity payload shared with the trusted proxy."""
    return f"{timestamp}\n{method.upper()}\n{path}\n{user_id}".encode()


async def get_current_user_id(
    request: Request,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
    x_auth_timestamp: Annotated[
        str | None,
        Header(alias="X-MySwing-Auth-Timestamp"),
    ] = None,
    x_auth_signature: Annotated[
        str | None,
        Header(alias="X-MySwing-Auth-Signature"),
    ] = None,
) -> UUID:
    """Resolve a user from a development header or signed proxy identity.

    ``signed_proxy`` is the fail-closed default. The reverse proxy must strip
    client-supplied identity headers and sign ``timestamp, method, path, user``
    with the shared HMAC secret after authenticating the request.
    """
    if settings.auth_mode == "development":
        if x_user_id is None:
            return DEV_DEFAULT_USER_ID
        return _parse_user_id(x_user_id)

    user_id = _parse_user_id(x_user_id)
    if x_auth_timestamp is None or x_auth_signature is None:
        raise _unauthorized("Signed proxy authentication headers are required")

    secret = settings.auth_proxy_secret
    secret_value = secret.get_secret_value() if secret is not None else ""
    if len(secret_value.encode()) < MIN_PROXY_SECRET_BYTES:
        logger.error("Signed proxy auth is enabled without MYSWING_AUTH_PROXY_SECRET")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not configured",
        )

    try:
        signed_at = int(x_auth_timestamp)
    except ValueError as exc:
        raise _unauthorized("Invalid authentication timestamp") from exc

    if abs(int(time.time()) - signed_at) > settings.auth_signature_ttl_seconds:
        raise _unauthorized("Authentication signature has expired")

    message = _canonical_proxy_message(
        timestamp=x_auth_timestamp,
        method=request.method,
        path=request.url.path,
        user_id=user_id,
    )
    expected_signature = hmac.new(
        secret_value.encode(),
        message,
        hashlib.sha256,
    ).hexdigest()
    candidate_signature = x_auth_signature.strip().lower()
    if len(candidate_signature) != 64 or any(
        character not in "0123456789abcdef" for character in candidate_signature
    ):
        raise _unauthorized("Invalid authentication signature")
    if not hmac.compare_digest(expected_signature, candidate_signature):
        raise _unauthorized("Invalid authentication signature")

    return user_id
