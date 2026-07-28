"""Shared test fixtures and configuration."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def explicit_development_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """API tests opt in to the local-only identity mode explicitly."""
    monkeypatch.setattr(settings, "auth_mode", "development")


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)
