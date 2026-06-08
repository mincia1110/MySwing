"""Unit tests for the video validator service.

Tests format validation, file size boundaries, resolution validation,
frame rate validation, duration validation, metadata extraction,
and the combined validate_video function.
"""

import os

import cv2
import numpy as np
import pytest

from app.core.constants import (
    MAX_FILE_SIZE_BYTES,
)
from app.services.video_validator import (
    extract_metadata,
    validate_duration,
    validate_file_size,
    validate_format,
    validate_frame_rate,
    validate_integrity,
    validate_mime_type,
    validate_resolution,
    validate_single_swing_input_policy,
    validate_video,
)

# --- Format Validation Tests ---


class TestValidateFormat:
    """Tests for validate_format function."""

    @pytest.mark.parametrize(
        "file_path",
        [
            "video.mp4",
            "video.MP4",
            "video.Mp4",
            "video.mov",
            "video.MOV",
            "video.avi",
            "video.AVI",
            "/path/to/file.mp4",
            "s3://bucket/key/file.mov",
        ],
    )
    def test_valid_formats(self, file_path: str) -> None:
        """Supported formats (mp4, mov, avi) should be accepted."""
        assert validate_format(file_path) is True

    @pytest.mark.parametrize(
        "file_path",
        [
            "video.wmv",
            "video.flv",
            "video.mkv",
            "video.webm",
            "video.txt",
            "video.jpg",
            "video",
            "",
        ],
    )
    def test_invalid_formats(self, file_path: str) -> None:
        """Unsupported formats should be rejected."""
        assert validate_format(file_path) is False


# --- MIME Type Validation Tests ---


class TestValidateMimeType:
    """Tests for validate_mime_type function."""

    @pytest.mark.parametrize(
        "file_path",
        [
            "video.mp4",
            "video.mov",
            "video.avi",
        ],
    )
    def test_valid_mime_types(self, file_path: str) -> None:
        """Files with supported MIME types should be accepted."""
        assert validate_mime_type(file_path) is True

    @pytest.mark.parametrize(
        "file_path",
        [
            "video.wmv",
            "video.txt",
            "video.jpg",
        ],
    )
    def test_invalid_mime_types(self, file_path: str) -> None:
        """Files with unsupported MIME types should be rejected."""
        assert validate_mime_type(file_path) is False


# --- File Size Validation Tests ---


class TestValidateFileSize:
    """Tests for validate_file_size function."""

    def test_exactly_500mb(self) -> None:
        """File exactly at 500MB limit should be accepted."""
        assert validate_file_size(MAX_FILE_SIZE_BYTES) is True

    def test_below_500mb(self) -> None:
        """File below 500MB should be accepted."""
        assert validate_file_size(MAX_FILE_SIZE_BYTES - 1) is True

    def test_over_500mb(self) -> None:
        """File over 500MB should be rejected."""
        assert validate_file_size(MAX_FILE_SIZE_BYTES + 1) is False

    def test_zero_size(self) -> None:
        """Zero-size file should be accepted (size check only)."""
        assert validate_file_size(0) is True

    def test_one_byte(self) -> None:
        """One-byte file should be accepted."""
        assert validate_file_size(1) is True

    def test_large_file(self) -> None:
        """1GB file should be rejected."""
        assert validate_file_size(1024 * 1024 * 1024) is False


# --- Resolution Validation Tests ---


class TestValidateResolution:
    """Tests for validate_resolution function."""

    def test_exactly_720p(self) -> None:
        """Exactly 1280x720 should be accepted."""
        assert validate_resolution(1280, 720) is True

    def test_above_720p(self) -> None:
        """1920x1080 (1080p) should be accepted."""
        assert validate_resolution(1920, 1080) is True

    def test_4k_resolution(self) -> None:
        """3840x2160 (4K) should be accepted."""
        assert validate_resolution(3840, 2160) is True

    def test_below_720p_width(self) -> None:
        """Width below 1280 should be rejected."""
        assert validate_resolution(1279, 720) is False

    def test_below_720p_height(self) -> None:
        """Height below 720 should be rejected."""
        assert validate_resolution(1280, 719) is False

    def test_below_720p_both(self) -> None:
        """Both dimensions below minimum should be rejected."""
        assert validate_resolution(640, 480) is False

    def test_width_ok_height_not(self) -> None:
        """Width meeting minimum but height below should be rejected."""
        assert validate_resolution(1920, 540) is False


# --- Frame Rate Validation Tests ---


class TestValidateFrameRate:
    """Tests for validate_frame_rate function."""

    def test_exactly_30fps(self) -> None:
        """Exactly 30fps should be accepted."""
        assert validate_frame_rate(30.0) is True

    def test_above_30fps(self) -> None:
        """60fps should be accepted."""
        assert validate_frame_rate(60.0) is True

    def test_below_30fps(self) -> None:
        """29.97fps should be rejected (below 30)."""
        assert validate_frame_rate(29.97) is False

    def test_24fps(self) -> None:
        """24fps (cinema standard) should be rejected."""
        assert validate_frame_rate(24.0) is False

    def test_120fps(self) -> None:
        """120fps (slow-motion) should be accepted."""
        assert validate_frame_rate(120.0) is True

    def test_zero_fps(self) -> None:
        """0fps should be rejected."""
        assert validate_frame_rate(0.0) is False


# --- Duration Validation Tests ---


class TestValidateDuration:
    """Tests for hard single-swing duration validation."""

    def test_exactly_10_seconds(self) -> None:
        """Exactly 10 seconds should be accepted at the hard maximum."""
        assert validate_duration(10.0) is True

    def test_below_10_seconds(self) -> None:
        """Below 10 seconds should be accepted."""
        assert validate_duration(9.99) is True

    def test_over_10_seconds(self) -> None:
        """Over 10 seconds should be rejected."""
        assert validate_duration(10.01) is False

    def test_zero_duration(self) -> None:
        """Zero duration should not pass the single-swing input policy silently."""
        assert validate_duration(0.0) is False


class TestSingleSwingInputPolicy:
    """Tests for structured short single-swing clip policy results."""

    def test_five_second_video_accepted_ok(self) -> None:
        result = validate_single_swing_input_policy(5.0)

        assert result.accepted is True
        assert result.severity == "ok"
        assert result.reason is None
        assert result.duration_sec == 5.0
        assert result.ideal_duration_sec == 5.0

    def test_two_second_video_accepted_with_warning(self) -> None:
        result = validate_single_swing_input_policy(2.0)

        assert result.accepted is True
        assert result.severity == "warning"
        assert result.reason == "video_too_short"
        assert "3~7초" in result.recommendation

    @pytest.mark.parametrize("duration", [8.0, 9.0])
    def test_eight_or_nine_second_video_accepted_with_warning(self, duration: float) -> None:
        result = validate_single_swing_input_policy(duration)

        assert result.accepted is True
        assert result.severity == "warning"
        assert result.reason == "video_longer_than_recommended"

    def test_eleven_second_video_rejected(self) -> None:
        result = validate_single_swing_input_policy(11.0)

        assert result.accepted is False
        assert result.severity == "error"
        assert result.reason == "video_too_long"
        assert result.max_duration_sec == 10.0

    def test_metadata_unavailable_rejected(self) -> None:
        result = validate_single_swing_input_policy(None)

        assert result.accepted is False
        assert result.severity == "error"
        assert result.reason == "metadata_unavailable"
        assert result.duration_sec is None


# --- Metadata Extraction Tests ---


class TestExtractMetadata:
    """Tests for extract_metadata function using a small generated test video."""

    @pytest.fixture
    def test_video_path(self, tmp_path) -> str:
        """Create a small test video file for metadata extraction tests."""
        video_path = str(tmp_path / "test_video.mp4")
        # Create a small 1280x720 video at 30fps with 30 frames (1 second)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))

        for _ in range(30):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()
        return video_path

    def test_extract_metadata_resolution(self, test_video_path: str) -> None:
        """Extracted resolution should match the created video."""
        metadata = extract_metadata(test_video_path)
        assert metadata.resolution_width == 1280
        assert metadata.resolution_height == 720

    def test_extract_metadata_fps(self, test_video_path: str) -> None:
        """Extracted frame rate should match the created video."""
        metadata = extract_metadata(test_video_path)
        assert metadata.frame_rate == pytest.approx(30.0, abs=0.1)

    def test_extract_metadata_duration(self, test_video_path: str) -> None:
        """Extracted duration should be approximately 1 second."""
        metadata = extract_metadata(test_video_path)
        assert metadata.duration_seconds == pytest.approx(1.0, abs=0.1)

    def test_extract_metadata_format(self, test_video_path: str) -> None:
        """Extracted format should be mp4."""
        metadata = extract_metadata(test_video_path)
        assert metadata.format == "mp4"

    def test_extract_metadata_file_name(self, test_video_path: str) -> None:
        """Extracted file name should match."""
        metadata = extract_metadata(test_video_path)
        assert metadata.file_name == "test_video.mp4"

    def test_extract_metadata_invalid_file(self, tmp_path) -> None:
        """Attempting to extract metadata from invalid file should raise ValueError."""
        invalid_path = str(tmp_path / "not_a_video.txt")
        with open(invalid_path, "w") as f:
            f.write("this is not a video")

        with pytest.raises(ValueError, match="Cannot open video file"):
            extract_metadata(invalid_path)


# --- Integrity Validation Tests ---


class TestValidateIntegrity:
    """Tests for validate_integrity function."""

    @pytest.fixture
    def valid_video_path(self, tmp_path) -> str:
        """Create a valid test video file."""
        video_path = str(tmp_path / "valid_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))

        for _ in range(30):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()
        return video_path

    def test_valid_video_integrity(self, valid_video_path: str) -> None:
        """A properly encoded video should pass integrity check."""
        assert validate_integrity(valid_video_path) is True

    def test_corrupted_file_integrity(self, tmp_path) -> None:
        """A corrupted file should fail integrity check."""
        corrupted_path = str(tmp_path / "corrupted.mp4")
        with open(corrupted_path, "wb") as f:
            f.write(b"\x00\x01\x02\x03" * 100)

        assert validate_integrity(corrupted_path) is False

    def test_nonexistent_file_integrity(self) -> None:
        """A nonexistent file should fail integrity check."""
        assert validate_integrity("/nonexistent/path/video.mp4") is False

    def test_empty_file_integrity(self, tmp_path) -> None:
        """An empty file should fail integrity check."""
        empty_path = str(tmp_path / "empty.mp4")
        with open(empty_path, "w"):
            pass

        assert validate_integrity(empty_path) is False


# --- Combined validate_video Tests ---


class TestValidateVideo:
    """Tests for the combined validate_video function."""

    @pytest.fixture
    def valid_video_path(self, tmp_path) -> str:
        """Create a valid test video that passes all checks."""
        video_path = str(tmp_path / "valid.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))

        # 90 frames = 3 seconds at 30fps
        for _ in range(90):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()
        return video_path

    def test_valid_video_passes_all_checks(self, valid_video_path: str) -> None:
        """A valid video should pass all validation checks."""
        file_size = os.path.getsize(valid_video_path)
        result = validate_video(valid_video_path, file_size)

        assert result.is_valid is True
        assert result.format_ok is True
        assert result.size_ok is True
        assert result.resolution_ok is True
        assert result.frame_rate_ok is True
        assert result.errors == []

    def test_invalid_format_rejected(self, tmp_path) -> None:
        """A file with unsupported format should be rejected."""
        invalid_path = str(tmp_path / "video.wmv")
        with open(invalid_path, "w") as f:
            f.write("fake content")

        result = validate_video(invalid_path, 100)

        assert result.is_valid is False
        assert result.format_ok is False
        assert any("지원" in e for e in result.errors)

    def test_oversized_file_rejected(self, valid_video_path: str) -> None:
        """A file exceeding 500MB should be rejected."""
        result = validate_video(valid_video_path, MAX_FILE_SIZE_BYTES + 1)

        assert result.is_valid is False
        assert result.size_ok is False
        assert any("500MB" in e for e in result.errors)

    def test_low_resolution_rejected(self, tmp_path) -> None:
        """A video with resolution below 720p should be rejected."""
        video_path = str(tmp_path / "low_res.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

        for _ in range(30):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()
        file_size = os.path.getsize(video_path)
        result = validate_video(video_path, file_size)

        assert result.is_valid is False
        assert result.resolution_ok is False
        assert any("해상도" in e for e in result.errors)

    def test_low_frame_rate_rejected(self, tmp_path) -> None:
        """A video with frame rate below 30fps should be rejected."""
        video_path = str(tmp_path / "low_fps.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 24fps - below minimum
        writer = cv2.VideoWriter(video_path, fourcc, 24.0, (1280, 720))

        for _ in range(24):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()
        file_size = os.path.getsize(video_path)
        result = validate_video(video_path, file_size)

        assert result.is_valid is False
        assert result.frame_rate_ok is False
        assert any("프레임레이트" in e for e in result.errors)

    def test_corrupted_file_rejected(self, tmp_path) -> None:
        """A corrupted file should be rejected."""
        corrupted_path = str(tmp_path / "corrupted.mp4")
        with open(corrupted_path, "wb") as f:
            f.write(b"\x00\x01\x02\x03" * 100)

        file_size = os.path.getsize(corrupted_path)
        result = validate_video(corrupted_path, file_size)

        assert result.is_valid is False
        assert any("읽을 수 없" in e for e in result.errors)

    def test_multiple_errors_reported(self, tmp_path) -> None:
        """Multiple validation failures should all be reported."""
        video_path = str(tmp_path / "bad_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Low resolution AND low frame rate
        writer = cv2.VideoWriter(video_path, fourcc, 24.0, (640, 480))

        for _ in range(24):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()
        # Also exceed file size
        result = validate_video(video_path, MAX_FILE_SIZE_BYTES + 1)

        assert result.is_valid is False
        assert result.size_ok is False
        assert result.resolution_ok is False
        assert result.frame_rate_ok is False
        assert len(result.errors) >= 3
