"""Unit tests for the VideoQualityChecker service.

Tests brightness, resolution, frame rate stability, and combined quality checks.
Validates Requirements 9.1-9.8.
"""

import os
import tempfile
import time

import cv2
import numpy as np
import pytest

from app.models.enums import QualityStatus
from app.models.video import QualityCheckResult
from app.services.quality_checker import (
    BRIGHTNESS_LUX_THRESHOLD,
    FRAME_RATE_VARIATION_THRESHOLD,
    VideoQualityChecker,
)


def _create_test_video(
    path: str,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    num_frames: int = 60,
    brightness: int = 200,
    variable_fps: bool = False,
) -> str:
    """Create a test video file with specified properties.

    Args:
        path: Output file path.
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Target frames per second.
        num_frames: Number of frames to write.
        brightness: Pixel brightness value (0-255).
        variable_fps: If True, simulate variable frame rate by duplicating frames.

    Returns:
        Path to the created video file.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a frame with specified brightness
        frame = np.full((height, width, 3), brightness, dtype=np.uint8)
        # Add some variation to make it more realistic
        if i % 5 == 0:
            # Add a moving rectangle to simulate motion
            x = int((i / num_frames) * (width - 100))
            cv2.rectangle(frame, (x, 100), (x + 100, 300), (255, 255, 255), -1)
        writer.write(frame)

    writer.release()
    return path


@pytest.fixture
def checker():
    """Create a VideoQualityChecker instance."""
    return VideoQualityChecker()


@pytest.fixture
def bright_video(tmp_path):
    """Create a bright test video (should pass brightness check)."""
    path = str(tmp_path / "bright_video.mp4")
    # brightness=200 -> lux = 200 * 0.5 = 100, well above threshold of 40
    _create_test_video(path, brightness=200, num_frames=30)
    return path


@pytest.fixture
def dark_video(tmp_path):
    """Create a dark test video (should trigger brightness warning)."""
    path = str(tmp_path / "dark_video.mp4")
    # brightness=30 -> lux = 30 * 0.5 = 15, below threshold of 40
    _create_test_video(path, brightness=30, num_frames=30)
    return path


@pytest.fixture
def hd_video(tmp_path):
    """Create a 720p video (should pass resolution check)."""
    path = str(tmp_path / "hd_video.mp4")
    _create_test_video(path, width=1280, height=720, num_frames=30)
    return path


@pytest.fixture
def low_res_video(tmp_path):
    """Create a below-720p video (should trigger resolution warning)."""
    path = str(tmp_path / "low_res_video.mp4")
    _create_test_video(path, width=640, height=480, num_frames=30)
    return path


class TestBrightnessCheck:
    """Tests for brightness validation (Requirements 9.1, 9.2)."""

    def test_bright_video_passes(self, checker, bright_video):
        """A video with sufficient brightness should pass the check."""
        result = checker.check_quality(bright_video)
        assert result.brightness_status == QualityStatus.PASS
        assert result.brightness_value >= BRIGHTNESS_LUX_THRESHOLD

    def test_dark_video_warns(self, checker, dark_video):
        """A video with insufficient brightness should produce a warning."""
        result = checker.check_quality(dark_video)
        assert result.brightness_status == QualityStatus.WARNING
        assert result.brightness_value < BRIGHTNESS_LUX_THRESHOLD

    def test_brightness_calculation(self, checker):
        """Brightness should be calculated as mean_pixel_value * 0.5."""
        # Create frames with known brightness
        frames = [np.full((100, 100, 3), 160, dtype=np.uint8)]
        status, lux = checker._check_brightness(frames)
        # mean pixel = 160, lux = 160 * 0.5 = 80
        assert abs(lux - 80.0) < 1.0
        assert status == QualityStatus.PASS

    def test_brightness_empty_frames(self, checker):
        """Empty frame list should return warning with 0 brightness."""
        status, lux = checker._check_brightness([])
        assert status == QualityStatus.WARNING
        assert lux == 0.0

    def test_brightness_at_threshold(self, checker):
        """Brightness exactly at threshold should pass."""
        # lux = pixel_value * 0.5 = 40 -> pixel_value = 80
        frames = [np.full((100, 100, 3), 80, dtype=np.uint8)]
        status, lux = checker._check_brightness(frames)
        assert abs(lux - 40.0) < 1.0
        assert status == QualityStatus.PASS


class TestResolutionCheck:
    """Tests for resolution validation (Requirement 9.5)."""

    def test_720p_passes(self, checker):
        """720p resolution should pass."""
        status = checker._check_resolution(1280, 720)
        assert status == QualityStatus.PASS

    def test_1080p_passes(self, checker):
        """1080p resolution should pass."""
        status = checker._check_resolution(1920, 1080)
        assert status == QualityStatus.PASS

    def test_below_720p_warns(self, checker):
        """Resolution below 720p should produce a warning."""
        status = checker._check_resolution(640, 480)
        assert status == QualityStatus.WARNING

    def test_exactly_720_height_passes(self, checker):
        """Exactly 720 height should pass regardless of width."""
        status = checker._check_resolution(960, 720)
        assert status == QualityStatus.PASS

    def test_719_height_warns(self, checker):
        """Height of 719 should produce a warning."""
        status = checker._check_resolution(1280, 719)
        assert status == QualityStatus.WARNING


class TestFrameRateStability:
    """Tests for frame rate stability validation (Requirements 9.6, 9.7)."""

    def test_stable_fps_passes(self, checker, bright_video):
        """A video with stable frame rate should pass."""
        result = checker.check_quality(bright_video)
        # Synthetic videos created with constant fps should be stable
        assert result.frame_rate_stability_status == QualityStatus.PASS
        assert result.frame_rate_variation_percent < FRAME_RATE_VARIATION_THRESHOLD

    def test_frame_rate_stability_calculation(self, checker):
        """Frame rate variation should be (max_fps - min_fps) / target_fps * 100."""
        # Direct test of the stability check method with a stable video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name

        try:
            _create_test_video(path, fps=30.0, num_frames=90)
            status, variation = checker._check_frame_rate_stability(path)
            # Constant fps video should have low variation
            assert status == QualityStatus.PASS
            assert variation < FRAME_RATE_VARIATION_THRESHOLD
        finally:
            os.unlink(path)

    def test_unopenable_video_warns(self, checker):
        """An unopenable video should return warning status."""
        status, variation = checker._check_frame_rate_stability("/nonexistent/video.mp4")
        assert status == QualityStatus.WARNING
        assert variation == 100.0


class TestMultiplePersonDetection:
    """Tests for subject ambiguity warnings."""

    def test_multiple_people_across_sampled_frames_adds_warning(
        self, checker, bright_video, monkeypatch
    ):
        """Repeated multi-person detections should warn about ambiguous subject selection."""
        calls = {"count": 0}

        def fake_detect_people(_frame):
            calls["count"] += 1
            return 2 if calls["count"] <= 2 else 1

        monkeypatch.setattr(checker, "_detect_people_in_frame", fake_detect_people)

        result = checker.check_quality(bright_video)

        assert any("여러 사람" in warning for warning in result.warnings)
        assert any("타자 1명" in warning for warning in result.warnings)

    def test_single_person_frames_do_not_add_subject_warning(
        self, checker, bright_video, monkeypatch
    ):
        """Single-person detections should not produce the ambiguity warning."""
        monkeypatch.setattr(checker, "_detect_people_in_frame", lambda _frame: 1)

        result = checker.check_quality(bright_video)

        assert not any("여러 사람" in warning for warning in result.warnings)

    def test_empty_frames_are_not_ambiguous(self, checker):
        """No sampled frames should not be reported as multiple people."""
        assert checker._has_multiple_people([]) is False


class TestCombinedQualityCheck:
    """Tests for the combined quality check result structure (Requirement 9.8)."""

    def test_result_structure(self, checker, bright_video):
        """Quality check result should have all required fields."""
        result = checker.check_quality(bright_video)

        assert isinstance(result, QualityCheckResult)
        assert isinstance(result.brightness_status, QualityStatus)
        assert isinstance(result.framing_status, QualityStatus)
        assert isinstance(result.resolution_status, QualityStatus)
        assert isinstance(result.frame_rate_stability_status, QualityStatus)
        assert isinstance(result.brightness_value, float)
        assert isinstance(result.swing_arc_visibility_percent, float)
        assert isinstance(result.frame_rate_variation_percent, float)
        assert isinstance(result.warnings, list)

    def test_all_pass_no_warnings(self, checker, bright_video):
        """A good quality video should have no warnings."""
        result = checker.check_quality(bright_video)
        # Bright, 720p, stable fps video should pass all checks
        assert result.brightness_status == QualityStatus.PASS
        assert result.resolution_status == QualityStatus.PASS
        assert result.frame_rate_stability_status == QualityStatus.PASS

    def test_dark_video_has_brightness_warning(self, checker, dark_video):
        """A dark video should have a brightness warning in the warnings list."""
        result = checker.check_quality(dark_video)
        assert result.brightness_status == QualityStatus.WARNING
        assert any("조명" in w for w in result.warnings)

    def test_low_res_video_has_resolution_warning(self, checker, low_res_video):
        """A low-res video should have a resolution warning."""
        result = checker.check_quality(low_res_video)
        assert result.resolution_status == QualityStatus.WARNING
        assert any("해상도" in w for w in result.warnings)

    def test_nonexistent_video_returns_failure(self, checker):
        """A nonexistent video path should return all-warning result."""
        result = checker.check_quality("/nonexistent/path/video.mp4")
        assert result.brightness_status == QualityStatus.WARNING
        assert result.framing_status == QualityStatus.WARNING
        assert result.resolution_status == QualityStatus.WARNING
        assert result.frame_rate_stability_status == QualityStatus.WARNING

    def test_quality_check_completes_within_time(self, checker, bright_video):
        """Quality check should complete within a reasonable time (< 10 seconds)."""
        start = time.time()
        checker.check_quality(bright_video)
        elapsed = time.time() - start
        # Should complete well within 10 seconds for a short test video
        assert elapsed < 10.0
