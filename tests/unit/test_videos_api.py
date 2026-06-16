"""Unit tests for video metadata API input-policy responses."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.video import VideoMetadata


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


class _AsyncSessionContext:
    def __init__(self, session: AsyncMock) -> None:
        self.session = session

    async def __aenter__(self) -> AsyncMock:
        return self.session

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _SessionFactory:
    def __init__(self, session: AsyncMock) -> None:
        self.session = session

    def __call__(self) -> _AsyncSessionContext:
        return _AsyncSessionContext(self.session)


def _mock_scalar_result(value):
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = value
    return mock_result


def _metadata(duration_seconds: float = 5.0) -> VideoMetadata:
    return VideoMetadata(
        file_key="/tmp/swing.mp4",
        file_name="swing.mp4",
        file_size_bytes=1024,
        duration_seconds=duration_seconds,
        resolution_width=1920,
        resolution_height=1080,
        frame_rate=60.0,
        codec="mp4v",
        format="mp4",
    )


class TestGetVideoMetadataInputPolicy:
    @patch("app.api.videos.generate_thumbnail_from_s3")
    @patch("app.api.videos.extract_metadata")
    @patch("app.api.videos.get_s3_client")
    @patch("app.api.videos.async_session_factory")
    def test_metadata_response_includes_input_validation(
        self,
        mock_session_factory,
        mock_get_s3_client,
        mock_extract_metadata,
        mock_generate_thumbnail,
        client,
    ):
        """Metadata endpoint should expose the structured input policy result."""
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {"ContentLength": 1024}
        mock_s3._client.download_file.return_value = None
        mock_s3._bucket = "myswing-videos"
        mock_s3.generate_presigned_download_url.return_value = "https://example.com/thumb.jpg"
        mock_get_s3_client.return_value = mock_s3
        mock_extract_metadata.return_value = _metadata(duration_seconds=8.0)
        mock_generate_thumbnail.return_value = "thumbs/swing.jpg"

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_mock_scalar_result(None))
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session_factory.return_value = _SessionFactory(mock_session)
        current_user_id = uuid.uuid4()

        response = client.post(
            "/api/v1/videos/uploads/test/swing.mp4/metadata",
            headers={"X-User-Id": str(current_user_id)},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["duration_seconds"] == 8.0
        assert body["input_validation"]["accepted"] is True
        assert body["input_validation"]["severity"] == "warning"
        assert body["input_validation"]["reason"] == "video_longer_than_recommended"
        assert body["input_validation"]["max_duration_sec"] == 10.0
        added_record = mock_session.add.call_args.args[0]
        assert added_record.user_id == current_user_id

    @patch("app.api.videos.generate_thumbnail_from_s3")
    @patch("app.api.videos.extract_metadata")
    @patch("app.api.videos.get_s3_client")
    @patch("app.api.videos.async_session_factory")
    def test_metadata_rejects_video_owned_by_another_user(
        self,
        mock_session_factory,
        mock_get_s3_client,
        mock_extract_metadata,
        mock_generate_thumbnail,
        client,
    ):
        """Existing video metadata cannot be updated by a different user."""
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {"ContentLength": 1024}
        mock_s3._client.download_file.return_value = None
        mock_s3._bucket = "myswing-videos"
        mock_s3.generate_presigned_download_url.return_value = "https://example.com/thumb.jpg"
        mock_get_s3_client.return_value = mock_s3
        mock_extract_metadata.return_value = _metadata(duration_seconds=5.0)
        mock_generate_thumbnail.return_value = "thumbs/swing.jpg"

        existing_record = MagicMock()
        existing_record.user_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_mock_scalar_result(existing_record))
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session_factory.return_value = _SessionFactory(mock_session)

        response = client.post(
            "/api/v1/videos/uploads/test/swing.mp4/metadata",
            headers={"X-User-Id": str(uuid.uuid4())},
        )

        assert response.status_code == 403
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()

    @patch("app.api.videos.extract_metadata")
    @patch("app.api.videos.get_s3_client")
    def test_metadata_rejects_oversized_s3_object_before_download(
        self,
        mock_get_s3_client,
        mock_extract_metadata,
        client,
    ):
        """Oversized uploaded objects are rejected before download/metadata work."""
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {"ContentLength": 501 * 1024 * 1024}
        mock_s3._client.download_file.return_value = None
        mock_s3._bucket = "myswing-videos"
        mock_get_s3_client.return_value = mock_s3

        response = client.post("/api/v1/videos/uploads/test/huge.mp4/metadata")

        assert response.status_code == 413
        detail = response.json()["detail"]
        assert detail["status"] == "file_too_large"
        assert detail["file_size_bytes"] == 501 * 1024 * 1024
        mock_s3._client.download_file.assert_not_called()
        mock_extract_metadata.assert_not_called()

    @patch("app.api.videos.extract_metadata")
    @patch("app.api.videos.get_s3_client")
    def test_metadata_unavailable_returns_structured_error(
        self,
        mock_get_s3_client,
        mock_extract_metadata,
        client,
    ):
        """Metadata extraction failures should not look like successful metadata responses."""
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {"ContentLength": 1024}
        mock_s3._client.download_file.return_value = None
        mock_s3._bucket = "myswing-videos"
        mock_get_s3_client.return_value = mock_s3
        mock_extract_metadata.side_effect = ValueError("Cannot open video file")

        response = client.post("/api/v1/videos/uploads/test/broken.mp4/metadata")

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["status"] == "metadata_unavailable"
        assert detail["input_validation"]["accepted"] is False
        assert detail["input_validation"]["reason"] == "metadata_unavailable"
        assert "Cannot process video file" in detail["message"]
