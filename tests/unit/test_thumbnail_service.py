"""Unit tests for the thumbnail generation service.

Tests thumbnail generation from valid videos, JPEG format verification,
thumbnail dimensions, error handling for invalid/corrupted videos,
and metadata response structure.
"""

import os

import cv2
import numpy as np
import pytest

from app.schemas.video import ResolutionResponse, VideoMetadataWithThumbnailResponse
from app.services.thumbnail_service import (
    THUMBNAIL_HEIGHT,
    THUMBNAIL_WIDTH,
    generate_thumbnail,
)


class TestGenerateThumbnail:
    """Tests for generate_thumbnail function."""

    @pytest.fixture
    def valid_video_path(self, tmp_path) -> str:
        """Create a small valid test video (1280x720, 30fps, 30 frames)."""
        video_path = str(tmp_path / "test_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))

        for i in range(30):
            # Create frames with varying content so first frame is identifiable
            frame = np.full((720, 1280, 3), fill_value=i * 8, dtype=np.uint8)
            writer.write(frame)

        writer.release()
        return video_path

    @pytest.fixture
    def colored_video_path(self, tmp_path) -> str:
        """Create a test video with a distinct colored first frame."""
        video_path = str(tmp_path / "colored_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1920, 1080))

        # First frame: blue
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue channel in BGR
        writer.write(frame)

        # Remaining frames: green
        for _ in range(29):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:, :, 1] = 255  # Green channel
            writer.write(frame)

        writer.release()
        return video_path

    def test_generates_thumbnail_from_valid_video(self, valid_video_path: str) -> None:
        """Should successfully generate thumbnail bytes from a valid video."""
        result = generate_thumbnail(valid_video_path)

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_thumbnail_is_jpeg_format(self, valid_video_path: str) -> None:
        """Generated thumbnail should be in JPEG format (starts with JPEG magic bytes)."""
        result = generate_thumbnail(valid_video_path)

        # JPEG files start with FF D8 FF
        assert result[:2] == b"\xff\xd8"

    def test_thumbnail_dimensions_are_correct(self, valid_video_path: str) -> None:
        """Generated thumbnail should have dimensions 320x180."""
        result = generate_thumbnail(valid_video_path)

        # Decode the JPEG to check dimensions
        img_array = np.frombuffer(result, dtype=np.uint8)
        decoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        assert decoded is not None
        assert decoded.shape[1] == THUMBNAIL_WIDTH  # width
        assert decoded.shape[0] == THUMBNAIL_HEIGHT  # height

    def test_thumbnail_from_higher_resolution_video(
        self, colored_video_path: str
    ) -> None:
        """Should generate correct thumbnail from 1080p video."""
        result = generate_thumbnail(colored_video_path)

        # Verify it's valid JPEG
        assert result[:2] == b"\xff\xd8"

        # Verify dimensions
        img_array = np.frombuffer(result, dtype=np.uint8)
        decoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        assert decoded.shape[1] == THUMBNAIL_WIDTH
        assert decoded.shape[0] == THUMBNAIL_HEIGHT

    def test_thumbnail_has_three_color_channels(self, valid_video_path: str) -> None:
        """Generated thumbnail should be a color image (3 channels)."""
        result = generate_thumbnail(valid_video_path)

        img_array = np.frombuffer(result, dtype=np.uint8)
        decoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        assert decoded.shape[2] == 3  # BGR channels

    def test_raises_error_for_nonexistent_file(self) -> None:
        """Should raise ValueError for a file that doesn't exist."""
        with pytest.raises(ValueError, match="Cannot open video file"):
            generate_thumbnail("/nonexistent/path/video.mp4")

    def test_raises_error_for_corrupted_file(self, tmp_path) -> None:
        """Should raise ValueError for a corrupted/unreadable video file."""
        corrupted_path = str(tmp_path / "corrupted.mp4")
        with open(corrupted_path, "wb") as f:
            f.write(b"\x00\x01\x02\x03" * 100)

        with pytest.raises(ValueError):
            generate_thumbnail(corrupted_path)

    def test_raises_error_for_empty_file(self, tmp_path) -> None:
        """Should raise ValueError for an empty file."""
        empty_path = str(tmp_path / "empty.mp4")
        with open(empty_path, "wb") as f:
            pass

        with pytest.raises(ValueError):
            generate_thumbnail(empty_path)

    def test_raises_error_for_text_file(self, tmp_path) -> None:
        """Should raise ValueError for a text file with video extension."""
        text_path = str(tmp_path / "fake.mp4")
        with open(text_path, "w") as f:
            f.write("this is not a video file")

        with pytest.raises(ValueError):
            generate_thumbnail(text_path)


class TestMetadataResponseStructure:
    """Tests for the VideoMetadataWithThumbnailResponse schema structure."""

    def test_response_contains_required_fields(self) -> None:
        """Response schema should contain file_name, duration, resolution, file_size."""
        response = VideoMetadataWithThumbnailResponse(
            file_name="swing_video.mp4",
            duration_seconds=3.5,
            resolution=ResolutionResponse(width=1920, height=1080),
            file_size_bytes=15_000_000,
            thumbnail_url="https://s3.example.com/thumbnail.jpg",
        )

        assert response.file_name == "swing_video.mp4"
        assert response.duration_seconds == 3.5
        assert response.resolution.width == 1920
        assert response.resolution.height == 1080
        assert response.file_size_bytes == 15_000_000
        assert response.thumbnail_url == "https://s3.example.com/thumbnail.jpg"

    def test_response_thumbnail_url_is_optional(self) -> None:
        """thumbnail_url should be optional (None when generation fails)."""
        response = VideoMetadataWithThumbnailResponse(
            file_name="video.mp4",
            duration_seconds=2.0,
            resolution=ResolutionResponse(width=1280, height=720),
            file_size_bytes=5_000_000,
        )

        assert response.thumbnail_url is None

    def test_response_serialization(self) -> None:
        """Response should serialize to dict with correct structure."""
        response = VideoMetadataWithThumbnailResponse(
            file_name="test.mp4",
            duration_seconds=1.0,
            resolution=ResolutionResponse(width=1280, height=720),
            file_size_bytes=1_000_000,
            thumbnail_url="https://example.com/thumb.jpg",
        )

        data = response.model_dump()

        assert "file_name" in data
        assert "duration_seconds" in data
        assert "resolution" in data
        assert "width" in data["resolution"]
        assert "height" in data["resolution"]
        assert "file_size_bytes" in data
        assert "thumbnail_url" in data

    def test_response_resolution_nested_structure(self) -> None:
        """Resolution should be a nested object with width and height."""
        response = VideoMetadataWithThumbnailResponse(
            file_name="test.mp4",
            duration_seconds=1.0,
            resolution=ResolutionResponse(width=3840, height=2160),
            file_size_bytes=50_000_000,
        )

        assert isinstance(response.resolution, ResolutionResponse)
        assert response.resolution.width == 3840
        assert response.resolution.height == 2160
