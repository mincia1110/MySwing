"""Unit tests for user history API endpoints (Requirements 8.7, 8.8).

Tests cover:
- GET /api/v1/users/{id}/analyses - Analysis history list (paginated, most recent first)
- GET /api/v1/users/{id}/trends - Trend data retrieval
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


def _make_analysis_row(
    user_id: uuid.UUID,
    status: str = "completed",
    created_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> MagicMock:
    """Create a mock AnalysisTable row."""
    row = MagicMock()
    row.id = uuid.uuid4()
    row.user_id = user_id
    row.video_id = uuid.uuid4()
    row.status = status
    row.created_at = created_at or datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    row.completed_at = completed_at or datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc)
    row.error_message = None
    return row


def _make_analysis_result_row(evaluations_data: list | None = None) -> MagicMock:
    """Create a mock AnalysisResultTable row."""
    row = MagicMock()
    row.id = uuid.uuid4()
    row.processing_time_seconds = 25.3
    row.evaluations_data = evaluations_data or [
        {
            "metric_name": "bat_speed",
            "measured_value": 108.5,
            "unit": "km/h",
            "rating": "within_range",
            "color_code": "green",
        },
        {
            "metric_name": "attack_angle",
            "measured_value": 12.3,
            "unit": "degrees",
            "rating": "within_range",
            "color_code": "green",
        },
    ]
    return row


class TestGetUserAnalyses:
    """Tests for GET /api/v1/users/{user_id}/analyses."""

    @patch("app.api.history.get_async_db")
    def test_empty_history(self, mock_get_db, client):
        """User with no analyses returns empty list with total=0."""
        user_id = uuid.uuid4()

        mock_session = AsyncMock()

        # Mock count query
        mock_count_result = MagicMock()
        mock_count_result.scalar_one.return_value = 0

        # Mock data query
        mock_data_result = MagicMock()
        mock_data_result.all.return_value = []

        mock_session.execute = AsyncMock(
            side_effect=[mock_count_result, mock_data_result]
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/users/{user_id}/analyses")
            assert response.status_code == 200
            body = response.json()
            assert body["items"] == []
            assert body["total"] == 0
            assert body["page"] == 1
            assert body["page_size"] == 20
            assert body["has_next"] is False
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.history.get_async_db")
    def test_my_analyses_uses_current_user_header(self, mock_get_db, client):
        """GET /me/analyses resolves the user from X-User-Id."""
        user_id = uuid.uuid4()
        mock_session = AsyncMock()

        mock_count_result = MagicMock()
        mock_count_result.scalar_one.return_value = 0

        mock_data_result = MagicMock()
        mock_data_result.all.return_value = []

        mock_session.execute = AsyncMock(
            side_effect=[mock_count_result, mock_data_result]
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(
                "/api/v1/me/analyses",
                headers={"X-User-Id": str(user_id)},
            )
            assert response.status_code == 200
            assert response.json()["total"] == 0
            executed_sql = str(mock_session.execute.call_args_list[0].args[0])
            assert "analyses.user_id" in executed_sql
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.history.get_async_db")
    def test_paginated_history(self, mock_get_db, client):
        """Returns paginated results with correct metadata."""
        user_id = uuid.uuid4()

        analysis = _make_analysis_row(user_id)

        mock_session = AsyncMock()

        # Mock count query - total 25 items
        mock_count_result = MagicMock()
        mock_count_result.scalar_one.return_value = 25

        # Mock data query - return one row tuple (analysis, file_name, processing_time)
        mock_row = (analysis, "swing_video.mp4", 25.3)
        mock_data_result = MagicMock()
        mock_data_result.all.return_value = [mock_row]

        mock_session.execute = AsyncMock(
            side_effect=[mock_count_result, mock_data_result]
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(
                f"/api/v1/users/{user_id}/analyses?page=1&page_size=10"
            )
            assert response.status_code == 200
            body = response.json()
            assert body["total"] == 25
            assert body["page"] == 1
            assert body["page_size"] == 10
            assert body["has_next"] is True
            assert len(body["items"]) == 1
            item = body["items"][0]
            assert item["analysis_id"] == str(analysis.id)
            assert item["video_id"] == str(analysis.video_id)
            assert item["status"] == "completed"
            assert item["video_file_name"] == "swing_video.mp4"
            assert item["processing_time_seconds"] == 25.3
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.history.get_async_db")
    def test_last_page_has_next_false(self, mock_get_db, client):
        """Last page returns has_next=False."""
        user_id = uuid.uuid4()

        mock_session = AsyncMock()

        # Total 5 items, page_size 20 → no next page
        mock_count_result = MagicMock()
        mock_count_result.scalar_one.return_value = 5

        mock_data_result = MagicMock()
        mock_data_result.all.return_value = []

        mock_session.execute = AsyncMock(
            side_effect=[mock_count_result, mock_data_result]
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/users/{user_id}/analyses")
            assert response.status_code == 200
            body = response.json()
            assert body["has_next"] is False
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.history.get_async_db")
    def test_invalid_page_param(self, mock_get_db, client):
        """Page < 1 returns 422 validation error."""
        user_id = uuid.uuid4()

        mock_session = AsyncMock()

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/users/{user_id}/analyses?page=0")
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.history.get_async_db")
    def test_invalid_page_size_param(self, mock_get_db, client):
        """page_size > 100 returns 422 validation error."""
        user_id = uuid.uuid4()

        mock_session = AsyncMock()

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/users/{user_id}/analyses?page_size=101")
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.history.get_async_db")
    def test_analysis_without_result(self, mock_get_db, client):
        """Analysis without result shows None for processing_time."""
        user_id = uuid.uuid4()
        analysis = _make_analysis_row(user_id, status="analyzing")

        mock_session = AsyncMock()

        mock_count_result = MagicMock()
        mock_count_result.scalar_one.return_value = 1

        # No processing_time (analysis not completed)
        mock_row = (analysis, "video.mp4", None)
        mock_data_result = MagicMock()
        mock_data_result.all.return_value = [mock_row]

        mock_session.execute = AsyncMock(
            side_effect=[mock_count_result, mock_data_result]
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/users/{user_id}/analyses")
            assert response.status_code == 200
            body = response.json()
            assert len(body["items"]) == 1
            assert body["items"][0]["processing_time_seconds"] is None
            assert body["items"][0]["status"] == "analyzing"
        finally:
            app.dependency_overrides.clear()


class TestGetUserTrends:
    """Tests for GET /api/v1/users/{user_id}/trends."""

    @patch("app.api.history.get_async_db")
    def test_no_recordings_returns_message(self, mock_get_db, client):
        """User with 0 completed analyses returns message (Req 8.8)."""
        user_id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/users/{user_id}/trends")
            assert response.status_code == 200
            body = response.json()
            assert body["total_recordings"] == 0
            assert body["metrics_history"] == {}
            assert body["message"] is not None
            assert "2" in body["message"]  # mentions minimum 2 recordings
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.history.get_async_db")
    def test_one_recording_returns_message(self, mock_get_db, client):
        """User with 1 completed analysis returns message (Req 8.8)."""
        user_id = uuid.uuid4()
        analysis = _make_analysis_row(user_id)
        analysis_result = _make_analysis_result_row()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [(analysis, analysis_result)]
        mock_session.execute = AsyncMock(return_value=mock_result)

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/users/{user_id}/trends")
            assert response.status_code == 200
            body = response.json()
            assert body["total_recordings"] == 1
            assert body["metrics_history"] == {}
            assert body["message"] is not None
            assert "2" in body["message"]
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.history.get_async_db")
    def test_two_recordings_returns_trend_data(self, mock_get_db, client):
        """User with 2 completed analyses returns trend data (Req 8.7)."""
        user_id = uuid.uuid4()

        analysis1 = _make_analysis_row(
            user_id,
            created_at=datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 10, 10, 1, 0, tzinfo=timezone.utc),
        )
        analysis2 = _make_analysis_row(
            user_id,
            created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc),
        )

        result1 = _make_analysis_result_row([
            {"metric_name": "bat_speed", "measured_value": 105.0, "rating": "within_range"},
        ])
        result2 = _make_analysis_result_row([
            {"metric_name": "bat_speed", "measured_value": 108.5, "rating": "within_range"},
        ])

        mock_session = AsyncMock()
        mock_result = MagicMock()
        # Returned in desc order (most recent first) from DB
        mock_result.all.return_value = [(analysis2, result2), (analysis1, result1)]
        mock_session.execute = AsyncMock(return_value=mock_result)

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/users/{user_id}/trends")
            assert response.status_code == 200
            body = response.json()
            assert body["total_recordings"] == 2
            assert body["message"] is None
            assert "bat_speed" in body["metrics_history"]
            bat_speed_points = body["metrics_history"]["bat_speed"]
            assert len(bat_speed_points) == 2
            # Should be in chronological order (oldest first)
            assert bat_speed_points[0]["value"] == 105.0
            assert bat_speed_points[1]["value"] == 108.5
            assert body["date_range_start"] is not None
            assert body["date_range_end"] is not None
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.history.get_async_db")
    def test_trend_data_multiple_metrics(self, mock_get_db, client):
        """Trend data includes all metrics from evaluations."""
        user_id = uuid.uuid4()

        analysis1 = _make_analysis_row(
            user_id,
            created_at=datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 10, 10, 1, 0, tzinfo=timezone.utc),
        )
        analysis2 = _make_analysis_row(
            user_id,
            created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc),
        )

        evaluations = [
            {
                "metric_name": "bat_speed",
                "measured_value": 108.5,
                "rating": "within_range",
            },
            {
                "metric_name": "attack_angle",
                "measured_value": 12.3,
                "rating": "above_range",
            },
            {
                "metric_name": "hand_path_efficiency",
                "measured_value": 0.85,
                "rating": "within_range",
            },
        ]
        result1 = _make_analysis_result_row(evaluations)
        result2 = _make_analysis_result_row(evaluations)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [(analysis2, result2), (analysis1, result1)]
        mock_session.execute = AsyncMock(return_value=mock_result)

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/users/{user_id}/trends")
            assert response.status_code == 200
            body = response.json()
            assert "bat_speed" in body["metrics_history"]
            assert "attack_angle" in body["metrics_history"]
            assert "hand_path_efficiency" in body["metrics_history"]
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.history.get_async_db")
    def test_trend_skips_invalid_evaluations(self, mock_get_db, client):
        """Evaluations with missing metric_name or value are skipped."""
        user_id = uuid.uuid4()

        analysis1 = _make_analysis_row(
            user_id,
            created_at=datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 10, 10, 1, 0, tzinfo=timezone.utc),
        )
        analysis2 = _make_analysis_row(
            user_id,
            created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc),
        )

        # Include invalid entries (missing metric_name, missing value)
        evaluations = [
            {
                "metric_name": "bat_speed",
                "measured_value": 108.5,
                "rating": "within_range",
            },
            {"metric_name": "", "measured_value": 12.3, "rating": "within_range"},  # empty name
            {
                "metric_name": "attack_angle",
                "measured_value": None,
                "rating": "within_range",
            },  # None value
        ]
        result1 = _make_analysis_result_row(evaluations)
        result2 = _make_analysis_result_row(evaluations)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [(analysis2, result2), (analysis1, result1)]
        mock_session.execute = AsyncMock(return_value=mock_result)

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/users/{user_id}/trends")
            assert response.status_code == 200
            body = response.json()
            # Only bat_speed should be present (empty name and None value skipped)
            assert "bat_speed" in body["metrics_history"]
            assert "" not in body["metrics_history"]
            assert "attack_angle" not in body["metrics_history"]
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.history.get_async_db")
    def test_trend_uses_completed_at_for_date(self, mock_get_db, client):
        """Trend data uses completed_at as recorded_at timestamp."""
        user_id = uuid.uuid4()

        completed1 = datetime(2024, 1, 10, 10, 1, 0, tzinfo=timezone.utc)
        completed2 = datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc)

        analysis1 = _make_analysis_row(
            user_id,
            created_at=datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc),
            completed_at=completed1,
        )
        analysis2 = _make_analysis_row(
            user_id,
            created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            completed_at=completed2,
        )

        result1 = _make_analysis_result_row([
            {"metric_name": "bat_speed", "measured_value": 105.0, "rating": "within_range"},
        ])
        result2 = _make_analysis_result_row([
            {"metric_name": "bat_speed", "measured_value": 110.0, "rating": "within_range"},
        ])

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [(analysis2, result2), (analysis1, result1)]
        mock_session.execute = AsyncMock(return_value=mock_result)

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/users/{user_id}/trends")
            assert response.status_code == 200
            body = response.json()
            points = body["metrics_history"]["bat_speed"]
            # First point should use completed_at of analysis1
            assert "2024-01-10" in points[0]["recorded_at"]
            assert "2024-01-15" in points[1]["recorded_at"]
        finally:
            app.dependency_overrides.clear()
