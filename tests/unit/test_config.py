"""Tests for application configuration."""

import pytest

from app.core.config import Settings


@pytest.mark.unit
def test_default_settings() -> None:
    """Default settings are correctly initialized."""
    settings = Settings()
    assert settings.app_name == "MySwing"
    assert settings.max_file_size_bytes == 500 * 1024 * 1024
    assert settings.max_video_duration_seconds == 300.0
    assert settings.min_resolution_width == 1280
    assert settings.min_resolution_height == 720
    assert settings.min_frame_rate == 30.0
    assert settings.supported_formats == ["mp4", "mov", "avi"]


@pytest.mark.unit
def test_supported_formats_match_requirements() -> None:
    """Supported formats match Requirement 1.1 (MP4, MOV, AVI)."""
    settings = Settings()
    assert "mp4" in settings.supported_formats
    assert "mov" in settings.supported_formats
    assert "avi" in settings.supported_formats
    assert len(settings.supported_formats) == 3


@pytest.mark.unit
def test_max_file_size_is_500mb() -> None:
    """Max file size is 500MB per Requirement 1.2."""
    settings = Settings()
    assert settings.max_file_size_bytes == 500 * 1024 * 1024


@pytest.mark.unit
def test_max_duration_is_5_minutes() -> None:
    """Max video duration is 5 minutes (300 seconds) per Requirement 1.2."""
    settings = Settings()
    assert settings.max_video_duration_seconds == 300.0


@pytest.mark.unit
def test_min_resolution_is_720p() -> None:
    """Min resolution is 720p (1280x720) per Requirement 1.3."""
    settings = Settings()
    assert settings.min_resolution_width == 1280
    assert settings.min_resolution_height == 720


@pytest.mark.unit
def test_min_frame_rate_is_30fps() -> None:
    """Min frame rate is 30fps per Requirement 1.3."""
    settings = Settings()
    assert settings.min_frame_rate == 30.0
