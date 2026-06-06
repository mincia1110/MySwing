"""Tests for health check and basic API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_health_check(client: TestClient) -> None:
    """Health check endpoint returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.unit
def test_api_status(client: TestClient) -> None:
    """API status endpoint returns operational status."""
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    assert response.json() == {"status": "operational"}
