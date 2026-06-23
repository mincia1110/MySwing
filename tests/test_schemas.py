"""Unit tests for Pydantic schemas (validation logic)."""

import pytest
from pydantic import ValidationError

from app.schemas.user_profile import UserProfileCreate, UserProfileResponse, UserProfileUpdate
from app.schemas.video import PresignedUrlRequest, VideoValidationResponse, QualityCheckResponse
from app.schemas.analysis import (
    AnalysisCreateRequest,
    AnalysisStatusResponse,
    MetricEvaluationResponse,
    ImprovementAreaResponse,
    DrillRecommendationResponse,
)


class TestUserProfileCreate:
    """Tests for UserProfileCreate schema validation (Requirements 2.1, 2.5)."""

    def test_valid_profile_required_fields_only(self):
        profile = UserProfileCreate(
            height=175.0,
            bat_length=33.0,
            batting_direction="right",
        )
        assert profile.height == 175.0
        assert profile.bat_length == 33.0
        assert profile.batting_direction == "right"

    def test_valid_profile_all_fields(self):
        profile = UserProfileCreate(
            height=180.0,
            bat_length=34.0,
            batting_direction="left",
            weight=80.0,
            camera_direction="side",
            age_group="adult",
            level="college",
            bat_weight=30.0,
        )
        assert profile.level == "college"
        assert profile.bat_weight == 30.0

    def test_missing_height_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            UserProfileCreate(
                bat_length=33.0,
                batting_direction="right",
            )
        assert "height" in str(exc_info.value)

    def test_missing_bat_length_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            UserProfileCreate(
                height=175.0,
                batting_direction="right",
            )
        assert "bat_length" in str(exc_info.value)

    def test_missing_batting_direction_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            UserProfileCreate(
                height=175.0,
                bat_length=33.0,
            )
        assert "batting_direction" in str(exc_info.value)

    def test_height_below_minimum_raises_error(self):
        with pytest.raises(ValidationError):
            UserProfileCreate(
                height=99.0,  # below 100
                bat_length=33.0,
                batting_direction="right",
            )

    def test_height_above_maximum_raises_error(self):
        with pytest.raises(ValidationError):
            UserProfileCreate(
                height=221.0,  # above 220
                bat_length=33.0,
                batting_direction="right",
            )

    def test_height_at_minimum_boundary(self):
        profile = UserProfileCreate(
            height=100.0,
            bat_length=24.0,
            batting_direction="right",
        )
        assert profile.height == 100.0

    def test_height_at_maximum_boundary(self):
        profile = UserProfileCreate(
            height=220.0,
            bat_length=36.0,
            batting_direction="left",
        )
        assert profile.height == 220.0

    def test_bat_length_below_minimum_raises_error(self):
        with pytest.raises(ValidationError):
            UserProfileCreate(
                height=175.0,
                bat_length=23.0,  # below 24
                batting_direction="right",
            )

    def test_bat_length_above_maximum_raises_error(self):
        with pytest.raises(ValidationError):
            UserProfileCreate(
                height=175.0,
                bat_length=92.0,  # above 91
                batting_direction="right",
            )

    def test_bat_weight_below_minimum_raises_error(self):
        with pytest.raises(ValidationError):
            UserProfileCreate(
                height=175.0,
                bat_length=33.0,
                batting_direction="right",
                bat_weight=15.0,  # below 16
            )

    def test_bat_weight_above_maximum_raises_error(self):
        with pytest.raises(ValidationError):
            UserProfileCreate(
                height=175.0,
                bat_length=33.0,
                batting_direction="right",
                bat_weight=37.0,  # above 36
            )

    def test_bat_weight_at_boundaries(self):
        profile_min = UserProfileCreate(
            height=175.0,
            bat_length=33.0,
            batting_direction="right",
            bat_weight=16.0,
        )
        profile_max = UserProfileCreate(
            height=175.0,
            bat_length=33.0,
            batting_direction="right",
            bat_weight=36.0,
        )
        assert profile_min.bat_weight == 16.0
        assert profile_max.bat_weight == 36.0

    def test_invalid_batting_direction_raises_error(self):
        with pytest.raises(ValidationError):
            UserProfileCreate(
                height=175.0,
                bat_length=33.0,
                batting_direction="center",  # invalid
            )

    def test_invalid_level_raises_error(self):
        with pytest.raises(ValidationError):
            UserProfileCreate(
                height=175.0,
                bat_length=33.0,
                batting_direction="right",
                level="amateur",  # invalid
            )

    def test_invalid_camera_direction_raises_error(self):
        with pytest.raises(ValidationError):
            UserProfileCreate(
                height=175.0,
                bat_length=33.0,
                batting_direction="right",
                camera_direction="top",  # invalid
            )


class TestUserProfileUpdate:
    """Tests for UserProfileUpdate schema (all fields optional)."""

    def test_empty_update(self):
        update = UserProfileUpdate()
        assert update.height is None
        assert update.bat_length is None

    def test_partial_update(self):
        update = UserProfileUpdate(height=180.0)
        assert update.height == 180.0
        assert update.bat_length is None

    def test_validation_still_applies(self):
        with pytest.raises(ValidationError):
            UserProfileUpdate(height=50.0)  # below 100


class TestPresignedUrlRequest:
    def test_valid_mp4(self):
        req = PresignedUrlRequest(file_name="swing.mp4", content_type="video/mp4")
        assert req.content_type == "video/mp4"

    def test_valid_mov(self):
        req = PresignedUrlRequest(file_name="swing.mov", content_type="video/quicktime")
        assert req.content_type == "video/quicktime"

    def test_invalid_content_type(self):
        with pytest.raises(ValidationError):
            PresignedUrlRequest(file_name="swing.wmv", content_type="video/wmv")


class TestAnalysisCreateRequest:
    def test_valid_request(self):
        req = AnalysisCreateRequest(
            file_key="uploads/abc123.mp4",
            user_id="00000000-0000-4000-8000-000000000001",
        )
        assert req.file_key == "uploads/abc123.mp4"


class TestMetricEvaluationResponse:
    def test_valid_response(self):
        resp = MetricEvaluationResponse(
            metric_name="bat_speed",
            measured_value=120.0,
            unit="km/h",
            reference_min=110.0,
            reference_max=130.0,
            deviation_percent=0.0,
            rating="within_range",
            color_code="green",
        )
        assert resp.rating == "within_range"

    def test_invalid_rating(self):
        with pytest.raises(ValidationError):
            MetricEvaluationResponse(
                metric_name="bat_speed",
                measured_value=120.0,
                unit="km/h",
                reference_min=110.0,
                reference_max=130.0,
                deviation_percent=0.0,
                rating="invalid",
                color_code="green",
            )

    def test_invalid_color_code(self):
        with pytest.raises(ValidationError):
            MetricEvaluationResponse(
                metric_name="bat_speed",
                measured_value=120.0,
                unit="km/h",
                reference_min=110.0,
                reference_max=130.0,
                deviation_percent=0.0,
                rating="within_range",
                color_code="blue",  # invalid
            )


class TestImprovementAreaResponse:
    def test_valid_response(self):
        resp = ImprovementAreaResponse(
            metric_name="bat_speed",
            deviation_percent=15.0,
            current_value=95.0,
            target_range_min=110.0,
            target_range_max=130.0,
            rank=1,
        )
        assert resp.rank == 1

    def test_rank_out_of_range(self):
        with pytest.raises(ValidationError):
            ImprovementAreaResponse(
                metric_name="bat_speed",
                deviation_percent=15.0,
                current_value=95.0,
                target_range_min=110.0,
                target_range_max=130.0,
                rank=4,  # above 3
            )
