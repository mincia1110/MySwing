"""Unit tests for video metadata API input-policy responses."""

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
        mock_s3.check_file_exists.return_value = True
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

        response = client.post("/api/v1/videos/uploads/test/swing.mp4/metadata")

        assert response.status_code == 200
        body = response.json()
        assert body["duration_seconds"] == 8.0
        assert body["input_validation"]["accepted"] is True
        assert body["input_validation"]["severity"] == "warning"
        assert body["input_validation"]["reason"] == "video_longer_than_recommended"
        assert body["input_validation"]["max_duration_sec"] == 10.0

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
        mock_s3.check_file_exists.return_value = True
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
