"""Tests for fail-closed API identity resolution."""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.dependencies import get_current_user_id
from app.core.config import settings

PROXY_SECRET = "unit-test-proxy-secret-with-32-bytes"


@pytest.fixture
def auth_client() -> TestClient:
    auth_app = FastAPI()

    @auth_app.get("/protected")
    async def protected(user_id: uuid.UUID = Depends(get_current_user_id)) -> dict:
        return {"user_id": str(user_id)}

    return TestClient(auth_app)


def _signed_headers(
    user_id: uuid.UUID,
    *,
    method: str,
    path: str,
    timestamp: int | None = None,
    secret: str = PROXY_SECRET,
) -> dict[str, str]:
    signed_at = str(timestamp if timestamp is not None else int(time.time()))
    message = f"{signed_at}\n{method.upper()}\n{path}\n{user_id}".encode()
    signature = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    return {
        "X-User-Id": str(user_id),
        "X-MySwing-Auth-Timestamp": signed_at,
        "X-MySwing-Auth-Signature": signature,
    }


def test_signed_proxy_mode_rejects_unsigned_identity(
    auth_client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "auth_mode", "signed_proxy")
    monkeypatch.setattr(settings, "auth_proxy_secret", SecretStr(PROXY_SECRET))

    response = auth_client.get(
        "/protected",
        headers={"X-User-Id": str(uuid.uuid4())},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "MySwingProxy"


def test_signed_proxy_mode_accepts_valid_request_identity(
    auth_client: TestClient,
    monkeypatch,
) -> None:
    user_id = uuid.uuid4()
    path = "/protected"
    monkeypatch.setattr(settings, "auth_mode", "signed_proxy")
    monkeypatch.setattr(settings, "auth_proxy_secret", SecretStr(PROXY_SECRET))

    response = auth_client.get(
        path,
        headers=_signed_headers(user_id, method="GET", path=path),
    )

    assert response.status_code == 200
    assert response.json() == {"user_id": str(user_id)}


def test_signed_proxy_mode_rejects_expired_signature(
    auth_client: TestClient,
    monkeypatch,
) -> None:
    user_id = uuid.uuid4()
    path = "/protected"
    monkeypatch.setattr(settings, "auth_mode", "signed_proxy")
    monkeypatch.setattr(settings, "auth_proxy_secret", SecretStr(PROXY_SECRET))

    response = auth_client.get(
        path,
        headers=_signed_headers(
            user_id,
            method="GET",
            path=path,
            timestamp=int(time.time()) - settings.auth_signature_ttl_seconds - 1,
        ),
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication signature has expired"


def test_signed_proxy_mode_fails_closed_when_secret_is_missing(
    auth_client: TestClient,
    monkeypatch,
) -> None:
    user_id = uuid.uuid4()
    path = "/protected"
    monkeypatch.setattr(settings, "auth_mode", "signed_proxy")
    monkeypatch.setattr(settings, "auth_proxy_secret", None)

    response = auth_client.get(
        path,
        headers=_signed_headers(user_id, method="GET", path=path),
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Authentication service is not configured"


def test_signature_is_bound_to_http_method_and_path(
    auth_client: TestClient,
    monkeypatch,
) -> None:
    user_id = uuid.uuid4()
    path = "/protected"
    monkeypatch.setattr(settings, "auth_mode", "signed_proxy")
    monkeypatch.setattr(settings, "auth_proxy_secret", SecretStr(PROXY_SECRET))

    response = auth_client.get(
        path,
        headers=_signed_headers(user_id, method="POST", path=path),
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid authentication signature"


def test_non_hex_signature_is_rejected_as_unauthorized(
    auth_client: TestClient,
    monkeypatch,
) -> None:
    user_id = uuid.uuid4()
    path = "/protected"
    headers = _signed_headers(user_id, method="GET", path=path)
    headers["X-MySwing-Auth-Signature"] = "g" * 64
    monkeypatch.setattr(settings, "auth_mode", "signed_proxy")
    monkeypatch.setattr(settings, "auth_proxy_secret", SecretStr(PROXY_SECRET))

    response = auth_client.get(path, headers=headers)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid authentication signature"
