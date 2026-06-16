"""Unit tests for standalone pipeline wrapper metadata propagation and canonical RHB contract."""

import uuid
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from app.tasks.pipeline import (
    _apply_wrist_bat_fallback,
    _bat_length_to_meters,
    _build_analysis_metadata,
    _quality_check_record_values,
    _run_swing_classification,
    analyze_biomechanics_task,
    classify_swing_task,
    evaluate_swing_task,
)


@dataclass
class _DummyTrajectory:
    detections: list


class _DummyEstimator:
    def __init__(self, *args, **kwargs):
        self.dominant_hand = kwargs.get("dominant_hand")

    def estimate_trajectory(self, pose_sequence, video_width=1, video_height=1):
        return _DummyTrajectory(detections=[])


def _analysis_id() -> str:
    return str(uuid.uuid4())


def test_quality_check_record_values_maps_statuses_and_details():
    values = _quality_check_record_values(
        {
            "brightness_status": "pass",
            "framing_status": "warning",
            "resolution_status": "pass",
            "frame_rate_stability_status": "pass",
            "brightness_value": 55.0,
            "swing_arc_visibility_percent": 72.5,
            "frame_rate_variation_percent": 2.1,
            "warnings": ["framing warning"],
        }
    )

    assert values == {
        "brightness_status": "pass",
        "framing_status": "warning",
        "resolution_status": "pass",
        "frame_rate_stability_status": "pass",
        "details": {
            "brightness_value": 55.0,
            "swing_arc_visibility_percent": 72.5,
            "frame_rate_variation_percent": 2.1,
            "warnings": ["framing warning"],
        },
    }


def test_bat_length_to_meters_converts_inches():
    assert _bat_length_to_meters(34.0) == pytest.approx(0.8636)


def test_bat_length_to_meters_converts_centimeters():
    assert _bat_length_to_meters(84.0) == pytest.approx(0.84)


def test_bat_length_to_meters_preserves_legacy_meter_values():
    assert _bat_length_to_meters(0.86) == pytest.approx(0.86)


def test_build_analysis_metadata_exposes_normalization_contract():
    metadata = _build_analysis_metadata(
        {
            "normalization_applied": True,
            "normalization_target_fps": 30.0,
            "normalization_crop_box": [656, 0, 608, 1080],
            "normalization_sampled_frame_count": 115,
            "original_fps": 59.95,
            "original_video_width": 1080,
            "original_video_height": 1920,
            "original_frame_count": 230,
            "fps": 30.0,
            "video_width": 608,
            "video_height": 1080,
            "frame_count": 115,
            "analysis_coordinate_system": "canonical_rhb",
            "canonical_batting_direction": "right",
        }
    )

    normalization = metadata["video_normalization"]
    assert normalization["normalization_applied"] is True
    assert normalization["normalization_crop_box"] == [656, 0, 608, 1080]
    assert normalization["original_fps"] == 59.95
    assert normalization["analysis_fps"] == 30.0
    assert normalization["analysis_video_width"] == 608
    assert metadata["analysis_coordinate_system"] == "canonical_rhb"
    assert metadata["canonical_batting_direction"] == "right"


@patch("app.tasks.pipeline._run_swing_classification")
@patch("app.tasks.pipeline._get_analysis_data")
def test_classify_swing_task_uses_video_fps(mock_get_data, mock_run_swing_classification):
    analysis_id = _analysis_id()
    mock_get_data.return_value = {"video_fps": 60.0}
    mock_run_swing_classification.return_value = {"status": "completed", "phases": {}}

    classify_swing_task.apply(
        args=[analysis_id, {"pose_sequence": []}, {"bat_trajectory": {}}]
    ).get()

    mock_run_swing_classification.assert_called_once()
    call_args = mock_run_swing_classification.call_args[0]
    assert call_args[0] == analysis_id
    assert call_args[3] == 60.0


@patch("app.tasks.pipeline._run_biomechanics_analysis")
@patch("app.tasks.pipeline._get_analysis_data")
def test_analyze_biomechanics_task_uses_metadata_and_builds_preprocessing_fallback(
    mock_get_data, mock_run_biomechanics
):
    analysis_id = _analysis_id()
    user_profile = {"height": 180.0, "bat_length": 33.0, "batting_direction": "left"}
    mock_get_data.return_value = {
        "video_fps": 120.0,
        "video_width": 1080,
        "video_height": 1920,
        "user_profile": user_profile,
    }
    mock_run_biomechanics.return_value = {"status": "completed"}

    analyze_biomechanics_task.apply(
        args=[analysis_id, {"pose_sequence": []}, {"bat_trajectory": {}}, {"phases": {}}]
    ).get()

    mock_run_biomechanics.assert_called_once()
    call_args = mock_run_biomechanics.call_args[0]
    assert call_args[0] == analysis_id
    assert call_args[4] == user_profile
    assert call_args[5] == 120.0
    assert call_args[6]["video_width"] == 1080
    assert call_args[6]["video_height"] == 1920


@patch("app.tasks.pipeline._run_swing_evaluation")
@patch("app.tasks.pipeline._get_analysis_data")
def test_evaluate_swing_task_uses_video_fps(mock_get_data, mock_run_swing_evaluation):
    analysis_id = _analysis_id()
    mock_get_data.return_value = {"video_fps": 60.0}
    mock_run_swing_evaluation.return_value = {
        "status": "completed",
        "evaluations": {},
        "improvements": {},
    }

    evaluate_swing_task.apply(args=[analysis_id, {"bat_speed": 100.0}, {"height": 175.0}]).get()

    mock_run_swing_evaluation.assert_called_once()
    call_args = mock_run_swing_evaluation.call_args[0]
    assert call_args[0] == analysis_id
    assert call_args[6] == 60.0


def test_apply_wrist_bat_fallback_passes_explicit_dominant_hand(monkeypatch):
    analysis_id = _analysis_id()

    captured = {"dominant_hand": None}

    class CapturingEstimator(_DummyEstimator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["dominant_hand"] = self.dominant_hand

    monkeypatch.setattr("app.pipeline.wrist_bat_estimator.WristBatEstimator", CapturingEstimator)

    pose_result = {"pose_sequence": [{"frame_index": 0, "keypoints": []}]}
    bat_result = {"bat_trajectory": {"detections": []}}
    preprocessing_result = {"frame_count": 1, "video_width": 1920, "video_height": 1080}

    # Ensure fallback path can continue through deserialization.
    with patch("app.tasks.pipeline._deserialize_pose_sequence", return_value=[]):
        result = _apply_wrist_bat_fallback(
            analysis_id,
            bat_result,
            pose_result,
            preprocessing_result,
            dominant_hand="right",
        )

    assert captured["dominant_hand"] == "right"
    assert result["method"] == "wrist_estimation"


def test_run_swing_classification_uses_batting_direction_param():
    # We call with empty pose_sequence so classifier initialization is skipped,
    # but this verifies the function accepts batting_direction and returns contract shape.
    result = _run_swing_classification(
        _analysis_id(),
        {"pose_sequence": []},
        {"bat_trajectory": {}},
        60.0,
        batting_direction="right",
    )

    assert result["status"] == "completed"
    assert result["phases"] == {}
