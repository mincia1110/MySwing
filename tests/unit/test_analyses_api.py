"""Unit tests for analysis job API endpoints (Requirements 1.1, 8.1).

Tests cover:
- POST /api/v1/analyses - Create analysis job (202 response)
- POST /api/v1/analyses - Missing profile returns 400
- POST /api/v1/analyses - Video not found returns 404
- GET /api/v1/analyses/{id}/status - Status query for various states
- GET /api/v1/analyses/{id}/status - Not found returns 404
- GET /api/v1/analyses/{id}/report - Report retrieval when completed
- GET /api/v1/analyses/{id}/report - Not completed returns 409
- GET /api/v1/analyses/{id}/overlay - Overlay URL retrieval
- GET /api/v1/analyses/{id}/overlay - No overlay available returns 404
- GET /api/v1/analyses/{id}/metrics - Metrics data retrieval
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.analyses import _quality_check_to_response
from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def mock_analysis():
    """Create a mock analysis record."""
    analysis = MagicMock()
    analysis.id = uuid.uuid4()
    analysis.user_id = uuid.uuid4()
    analysis.video_id = uuid.uuid4()
    analysis.status = "pending"
    analysis.error_message = None
    analysis.started_at = None
    analysis.completed_at = None
    analysis.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return analysis


@pytest.fixture
def mock_completed_analysis():
    """Create a mock completed analysis record."""
    analysis = MagicMock()
    analysis.id = uuid.uuid4()
    analysis.user_id = uuid.uuid4()
    analysis.video_id = uuid.uuid4()
    analysis.status = "completed"
    analysis.error_message = None
    analysis.started_at = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    analysis.completed_at = datetime(2024, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
    analysis.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return analysis


@pytest.fixture
def mock_analysis_result():
    """Create a mock analysis result record."""
    result = MagicMock()
    result.id = uuid.uuid4()
    result.analysis_id = uuid.uuid4()
    result.biomechanics_data = {
        "bat_speed": {"speed_kmh": 120.5, "precision": 1.0},
        "attack_angle": {"angle_degrees": 12.3, "precision": 0.5},
        "hand_path_efficiency": 0.85,
        "processing_time_seconds": 25.3,
    }
    result.swing_phases_data = [
        {"phase": "stance", "start_frame": 0, "end_frame": 30, "duration_ms": 1000.0},
        {"phase": "load", "start_frame": 30, "end_frame": 45, "duration_ms": 500.0},
    ]
    result.evaluations_data = [
        {
            "metric_name": "bat_speed",
            "measured_value": 120.5,
            "unit": "km/h",
            "reference_min": 100.0,
            "reference_max": 140.0,
            "deviation_percent": 0.0,
            "rating": "within_range",
            "color_code": "green",
        }
    ]
    result.improvements_data = [
        {
            "metric_name": "attack_angle",
            "deviation_percent": 15.0,
            "current_value": 3.0,
            "target_range_min": 5.0,
            "target_range_max": 15.0,
            "rank": 1,
        }
    ]
    result.drill_recommendations = [
        {
            "drill_name": "Tee Work - High Tee",
            "target_metric": "attack_angle",
            "description": "Practice hitting off a high tee to increase launch angle.",
        }
    ]
    result.overlay_video_key = "overlays/test-overlay.mp4"
    result.processing_time_seconds = 25.3
    return result


@pytest.fixture
def mock_video():
    """Create a mock video record."""
    video = MagicMock()
    video.id = uuid.uuid4()
    video.user_id = uuid.uuid4()
    video.file_key = "uploads/test-uuid/swing.mp4"
    video.file_name = "swing.mp4"
    video.file_size_bytes = 50_000_000
    video.duration_seconds = 5.0
    video.resolution_width = 1920
    video.resolution_height = 1080
    video.frame_rate = 60.0
    video.format = "mp4"
    return video


@pytest.fixture
def mock_profile():
    """Create a mock user profile record."""
    profile = MagicMock()
    profile.id = uuid.uuid4()
    profile.user_id = uuid.uuid4()
    profile.height = 175.0
    profile.bat_length = 34.0
    profile.batting_direction = "right"
    return profile


def test_quality_check_to_response_includes_details_and_checked_at():
    checked_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    row = MagicMock()
    row.brightness_status = "pass"
    row.framing_status = "warning"
    row.resolution_status = "pass"
    row.frame_rate_stability_status = "pass"
    row.details = {
        "brightness_value": 55.0,
        "swing_arc_visibility_percent": 72.5,
        "frame_rate_variation_percent": 2.1,
        "warnings": ["framing warning"],
    }
    row.checked_at = checked_at

    assert _quality_check_to_response(row) == {
        "brightness_status": "pass",
        "framing_status": "warning",
        "resolution_status": "pass",
        "frame_rate_stability_status": "pass",
        "brightness_value": 55.0,
        "swing_arc_visibility_percent": 72.5,
        "frame_rate_variation_percent": 2.1,
        "warnings": ["framing warning"],
        "checked_at": checked_at,
    }


def _mock_scalar_result(value):
    """Helper to create a mock SQLAlchemy result with scalar_one_or_none."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = value
    return mock_result


class TestCreateAnalysis:
    """Tests for POST /api/v1/analyses."""

    @patch("app.api.analyses.analyze_swing_task")
    @patch("app.api.analyses.get_async_db")
    def test_create_analysis_success(
        self, mock_get_db, mock_task, client, mock_profile, mock_video
    ):
        """Creating analysis with valid data returns 202 with analysis_id."""
        user_id = mock_profile.user_id

        mock_session = AsyncMock()
        # First call: profile lookup, Second call: video lookup
        mock_session.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(mock_profile),
                _mock_scalar_result(mock_video),
            ]
        )
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.post(
                "/api/v1/analyses",
                json={
                    "file_key": mock_video.file_key,
                    "user_id": str(user_id),
                },
            )
            assert response.status_code == 202
            body = response.json()
            assert "analysis_id" in body
            assert body["status"] == "pending"
            mock_task.delay.assert_called_once()
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_create_analysis_no_profile(self, mock_get_db, client):
        """Creating analysis without user profile returns 400."""
        user_id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(None)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.post(
                "/api/v1/analyses",
                json={
                    "file_key": "uploads/test/video.mp4",
                    "user_id": str(user_id),
                },
            )
            assert response.status_code == 400
            body = response.json()
            assert "profile" in body["detail"].lower()
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_create_analysis_video_not_found(self, mock_get_db, client, mock_profile):
        """Creating analysis with non-existent video returns 404."""
        user_id = mock_profile.user_id

        mock_session = AsyncMock()
        # First call returns profile, second returns None (video not found)
        mock_session.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(mock_profile),
                _mock_scalar_result(None),
            ]
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.post(
                "/api/v1/analyses",
                json={
                    "file_key": "uploads/nonexistent/video.mp4",
                    "user_id": str(user_id),
                },
            )
            assert response.status_code == 404
            body = response.json()
            assert "not found" in body["detail"].lower()
        finally:
            app.dependency_overrides.clear()


    @patch("app.api.analyses.analyze_swing_task")
    @patch("app.api.analyses.get_async_db")
    def test_create_analysis_accepts_short_clip_with_validation_result(
        self, mock_get_db, mock_task, client, mock_profile, mock_video
    ):
        """A 5 second single-swing clip is accepted and exposes input validation."""
        mock_video.duration_seconds = 5.0
        user_id = mock_profile.user_id

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(mock_profile),
                _mock_scalar_result(mock_video),
            ]
        )
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.post(
                "/api/v1/analyses",
                json={"file_key": mock_video.file_key, "user_id": str(user_id)},
            )
            assert response.status_code == 202
            body = response.json()
            assert body["status"] == "pending"
            assert body["input_validation"]["accepted"] is True
            assert body["input_validation"]["severity"] == "ok"
            assert body["input_validation"]["duration_sec"] == 5.0
            mock_task.delay.assert_called_once()
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.analyze_swing_task")
    @patch("app.api.analyses.get_async_db")
    def test_create_analysis_uses_current_user_when_body_user_id_is_omitted(
        self, mock_get_db, mock_task, client, mock_profile, mock_video
    ):
        """Current-user context supplies the user id for the new /me-style flow."""
        mock_video.duration_seconds = 5.0
        mock_video.user_id = mock_profile.user_id

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(mock_profile),
                _mock_scalar_result(mock_video),
            ]
        )
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.post(
                "/api/v1/analyses",
                json={"file_key": mock_video.file_key},
                headers={"X-User-Id": str(mock_profile.user_id)},
            )
            assert response.status_code == 202
            mock_task.delay.assert_called_once()
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.analyze_swing_task")
    @patch("app.api.analyses.get_async_db")
    def test_create_analysis_rejects_video_not_owned_by_user(
        self, mock_get_db, mock_task, client, mock_profile, mock_video
    ):
        """A user cannot create an analysis for a file_key they do not own."""
        mock_video.duration_seconds = 5.0

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(mock_profile),
                _mock_scalar_result(None),
            ]
        )
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.post(
                "/api/v1/analyses",
                json={"file_key": mock_video.file_key, "user_id": str(mock_profile.user_id)},
            )
            assert response.status_code == 404
            assert "Video not found" in response.json()["detail"]
            mock_session.add.assert_not_called()
            mock_task.delay.assert_not_called()
        finally:
            app.dependency_overrides.clear()

    def test_create_analysis_rejects_invalid_user_id_before_db_lookup(self, client):
        """The request schema validates user_id as a UUID and returns 422."""
        mock_session = AsyncMock()

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db
        try:
            response = client.post(
                "/api/v1/analyses",
                json={"file_key": "uploads/test/video.mp4", "user_id": "not-a-uuid"},
            )
            assert response.status_code == 422
            mock_session.execute.assert_not_called()
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.analyze_swing_task")
    @patch("app.api.analyses.get_async_db")
    def test_create_analysis_rejects_video_longer_than_10_seconds_before_queue(
        self, mock_get_db, mock_task, client, mock_profile, mock_video
    ):
        """An 11 second video is rejected before the Celery task is queued."""
        mock_video.duration_seconds = 11.0

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(mock_profile),
                _mock_scalar_result(mock_video),
            ]
        )
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.post(
                "/api/v1/analyses",
                json={"file_key": mock_video.file_key, "user_id": str(mock_profile.user_id)},
            )
            assert response.status_code == 400
            detail = response.json()["detail"]
            assert detail["status"] == "video_too_long"
            assert detail["input_validation"]["accepted"] is False
            assert detail["input_validation"]["reason"] == "video_too_long"
            mock_session.add.assert_not_called()
            mock_task.delay.assert_not_called()
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.analyze_swing_task")
    @patch("app.api.analyses.get_async_db")
    def test_create_analysis_metadata_unavailable_does_not_queue_successful_analysis(
        self, mock_get_db, mock_task, client, mock_profile, mock_video
    ):
        """Missing duration metadata is rejected explicitly instead of queued."""
        mock_video.duration_seconds = None

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(mock_profile),
                _mock_scalar_result(mock_video),
            ]
        )
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.post(
                "/api/v1/analyses",
                json={"file_key": mock_video.file_key, "user_id": str(mock_profile.user_id)},
            )
            assert response.status_code == 400
            detail = response.json()["detail"]
            assert detail["status"] == "metadata_unavailable"
            assert detail["input_validation"]["reason"] == "metadata_unavailable"
            mock_session.add.assert_not_called()
            mock_task.delay.assert_not_called()
        finally:
            app.dependency_overrides.clear()

    def test_create_analysis_missing_fields(self, client):
        """Creating analysis with missing fields returns 422."""
        mock_session = AsyncMock()

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.post("/api/v1/analyses", json={})
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()


class TestGetAnalysisStatus:
    """Tests for GET /api/v1/analyses/{id}/status."""

    @patch("app.api.analyses.get_async_db")
    def test_get_status_pending(self, mock_get_db, client, mock_analysis):
        """Getting status of pending analysis returns correct status."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(mock_analysis)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/analyses/{mock_analysis.id}/status")
            assert response.status_code == 200
            body = response.json()
            assert body["analysis_id"] == str(mock_analysis.id)
            assert body["status"] == "pending"
            assert body["error_message"] is None
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_get_status_analyzing(self, mock_get_db, client, mock_analysis):
        """Getting status of analyzing analysis returns correct status."""
        mock_analysis.status = "analyzing"
        mock_analysis.started_at = datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(mock_analysis)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/analyses/{mock_analysis.id}/status")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "analyzing"
            assert body["started_at"] is not None
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_get_status_failed(self, mock_get_db, client, mock_analysis):
        """Getting status of failed analysis returns error message."""
        mock_analysis.status = "failed"
        mock_analysis.error_message = "Pose estimation failed: insufficient keypoints"

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(mock_analysis)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/analyses/{mock_analysis.id}/status")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "failed"
            assert "Pose estimation" in body["error_message"]
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_get_status_not_found(self, mock_get_db, client):
        """Getting status of non-existent analysis returns 404."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(None)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            analysis_id = uuid.uuid4()
            response = client.get(f"/api/v1/analyses/{analysis_id}/status")
            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_get_status_completed(self, mock_get_db, client, mock_completed_analysis):
        """Getting status of completed analysis returns completed with timestamps."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(mock_completed_analysis)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(
                f"/api/v1/analyses/{mock_completed_analysis.id}/status"
            )
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "completed"
            assert body["started_at"] is not None
            assert body["completed_at"] is not None
        finally:
            app.dependency_overrides.clear()


class TestGetAnalysisReport:
    """Tests for GET /api/v1/analyses/{id}/report."""

    @patch("app.api.analyses.get_s3_client")
    @patch("app.api.analyses.get_async_db")
    def test_get_report_success(
        self,
        mock_get_db,
        mock_s3,
        client,
        mock_completed_analysis,
        mock_analysis_result,
        mock_video,
    ):
        """Getting report of completed analysis returns full report data."""
        mock_analysis_result.analysis_id = mock_completed_analysis.id

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(mock_completed_analysis),
                _mock_scalar_result(mock_analysis_result),
                _mock_scalar_result(mock_video),
                _mock_scalar_result(None),
            ]
        )

        mock_s3_instance = MagicMock()
        mock_s3_instance.generate_presigned_download_url.return_value = (
            "https://s3.example.com/overlay.mp4"
        )
        mock_s3.return_value = mock_s3_instance

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(
                f"/api/v1/analyses/{mock_completed_analysis.id}/report"
            )
            assert response.status_code == 200
            body = response.json()
            assert body["analysis_id"] == str(mock_completed_analysis.id)
            assert body["status"] == "completed"
            assert body["overlay_video_url"] == "https://s3.example.com/overlay.mp4"
            assert len(body["metric_evaluations"]) == 1
            assert body["metric_evaluations"][0]["metric_name"] == "bat_speed"
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_get_report_not_completed(self, mock_get_db, client, mock_analysis):
        """Getting report of non-completed analysis returns 409."""
        mock_analysis.status = "analyzing"

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(mock_analysis)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/analyses/{mock_analysis.id}/report")
            assert response.status_code == 409
            body = response.json()
            assert "not yet completed" in body["detail"]
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_get_report_not_found(self, mock_get_db, client):
        """Getting report of non-existent analysis returns 404."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(None)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            analysis_id = uuid.uuid4()
            response = client.get(f"/api/v1/analyses/{analysis_id}/report")
            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()


class TestGetAnalysisOverlay:
    """Tests for GET /api/v1/analyses/{id}/overlay."""

    @patch("app.api.analyses.get_s3_client")
    @patch("app.api.analyses.get_async_db")
    def test_get_overlay_success(
        self, mock_get_db, mock_s3, client, mock_completed_analysis, mock_analysis_result
    ):
        """Getting overlay of completed analysis returns presigned URL."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(mock_completed_analysis),
                _mock_scalar_result(mock_analysis_result),
            ]
        )

        mock_s3_instance = MagicMock()
        mock_s3_instance.generate_presigned_download_url.return_value = (
            "https://s3.example.com/overlay.mp4"
        )
        mock_s3.return_value = mock_s3_instance

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(
                f"/api/v1/analyses/{mock_completed_analysis.id}/overlay"
            )
            assert response.status_code == 200
            body = response.json()
            assert body["overlay_video_url"] == "https://s3.example.com/overlay.mp4"
            assert body["expires_in"] == 3600
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_get_overlay_not_completed(self, mock_get_db, client, mock_analysis):
        """Getting overlay of non-completed analysis returns 409."""
        mock_analysis.status = "preprocessing"

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(mock_analysis)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/analyses/{mock_analysis.id}/overlay")
            assert response.status_code == 409
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_get_overlay_no_overlay_available(self, mock_get_db, client, mock_completed_analysis):
        """Getting overlay when no overlay video exists returns 404."""
        mock_result = MagicMock()
        mock_result.overlay_video_key = None

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(mock_completed_analysis),
                _mock_scalar_result(mock_result),
            ]
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(
                f"/api/v1/analyses/{mock_completed_analysis.id}/overlay"
            )
            assert response.status_code == 404
            body = response.json()
            assert "not available" in body["detail"].lower()
        finally:
            app.dependency_overrides.clear()


class TestGetAnalysisMetrics:
    """Tests for GET /api/v1/analyses/{id}/metrics."""

    @patch("app.api.analyses.get_async_db")
    def test_get_metrics_success(
        self, mock_get_db, client, mock_completed_analysis, mock_analysis_result
    ):
        """Getting metrics of completed analysis returns biomechanics and evaluations."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(mock_completed_analysis),
                _mock_scalar_result(mock_analysis_result),
            ]
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(
                f"/api/v1/analyses/{mock_completed_analysis.id}/metrics"
            )
            assert response.status_code == 200
            body = response.json()
            assert body["analysis_id"] == str(mock_completed_analysis.id)
            assert "biomechanics" in body
            assert body["biomechanics"]["bat_speed"]["speed_kmh"] == 120.5
            assert "evaluations" in body
            assert len(body["evaluations"]) == 1
            assert body["processing_time_seconds"] == 25.3
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_get_metrics_not_completed(self, mock_get_db, client, mock_analysis):
        """Getting metrics of non-completed analysis returns 409."""
        mock_analysis.status = "evaluating"

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(mock_analysis)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/analyses/{mock_analysis.id}/metrics")
            assert response.status_code == 409
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.analyses.get_async_db")
    def test_get_metrics_not_found(self, mock_get_db, client):
        """Getting metrics of non-existent analysis returns 404."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(None)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            analysis_id = uuid.uuid4()
            response = client.get(f"/api/v1/analyses/{analysis_id}/metrics")
            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()


class TestStatusTransitions:
    """Tests verifying all valid status values are handled."""

    @pytest.mark.parametrize(
        "status_value",
        [
            "pending",
            "preprocessing",
            "analyzing",
            "evaluating",
            "generating_report",
            "completed",
            "failed",
        ],
    )
    @patch("app.api.analyses.get_async_db")
    def test_all_status_values_returned(self, mock_get_db, client, status_value):
        """All valid status values are correctly returned by the status endpoint."""
        mock_analysis = MagicMock()
        mock_analysis.id = uuid.uuid4()
        mock_analysis.status = status_value
        mock_analysis.error_message = "Error" if status_value == "failed" else None
        mock_analysis.started_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_analysis.completed_at = (
            datetime(2024, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
            if status_value == "completed"
            else None
        )
        mock_analysis.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_mock_scalar_result(mock_analysis)
        )

        async def override_get_db():
            yield mock_session

        from app.db.session import get_async_db
        app.dependency_overrides[get_async_db] = override_get_db

        try:
            response = client.get(f"/api/v1/analyses/{mock_analysis.id}/status")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == status_value
        finally:
            app.dependency_overrides.clear()
