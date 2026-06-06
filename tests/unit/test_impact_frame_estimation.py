"""Unit tests for impact-frame fallback estimation and confidence metadata."""

from unittest.mock import patch

from app.tasks.pipeline import (
    _estimate_impact_frame_from_bat_speed,
    _run_biomechanics_analysis,
)


def _bat_detection(frame_index: int, x: float, *, detected: bool = True) -> dict:
    return {
        "frame_index": frame_index,
        "detected": detected,
        "position": [x, 0.0],
        "orientation_angle": 0.0,
        "length_pixels": 10.0,
        "confidence": 0.9,
        "is_predicted": False,
        "coordinate_space": "pixel",
    }


def _pose_frame(frame_index: int) -> dict:
    # Minimal shape accepted by _deserialize_pose_sequence
    return {
        "frame_index": frame_index,
        "keypoints": [],
        "person_id": 0,
        "is_primary_batter": True,
        "overall_confidence": 0.9,
        "is_low_confidence": False,
    }


class _StubOrchestrator:
    def analyze(self, **kwargs):
        from app.models.biomechanics import BiomechanicsResult

        return BiomechanicsResult()


def test_estimate_impact_frame_uses_smoothed_peak_not_single_spike():
    # Speed profile by frame-to-frame displacement:
    # [1, 2, 20, 9, 8, 2] -> after median smoothing, dominant region is ~9 near frames 3-4.
    xs = [0, 1, 3, 23, 32, 40, 42]
    detections = [_bat_detection(i, x) for i, x in enumerate(xs)]

    class _Traj:
        def __init__(self, detections):
            self.detections = detections

    impact_frame, confidence, method = _estimate_impact_frame_from_bat_speed(_Traj(detections))

    assert method == "smoothed_peak_speed"
    assert impact_frame in (3, 4)
    assert 0.0 < confidence <= 1.0


def test_estimate_impact_frame_prefers_late_peak_for_full_swing_sequences():
    """Long swing fallback should ignore pre-impact acceleration bursts."""
    # Frame-to-frame displacement has a larger mid-swing burst around frames
    # 54-55 and a smaller later hitting-zone peak around frames 63-65. For a
    # full-length 115-frame swing, impact should be selected from the later
    # hitting-zone window rather than the early acceleration burst.
    xs = [0.0]
    for frame in range(1, 115):
        if frame in (54, 55):
            step = 40.0
        elif frame in (63, 64, 65):
            step = 30.0
        else:
            step = 3.0
        xs.append(xs[-1] + step)
    detections = [_bat_detection(i, x) for i, x in enumerate(xs)]

    class _Traj:
        def __init__(self, detections):
            self.detections = detections

    impact_frame, _, method = _estimate_impact_frame_from_bat_speed(_Traj(detections))

    assert method == "smoothed_peak_speed"
    assert impact_frame in (63, 64, 65)


@patch("app.pipeline.biomechanics_analyzer.BiomechanicsOrchestrator", return_value=_StubOrchestrator())
def test_run_biomechanics_analysis_adds_impact_metadata_when_phase_missing(_mock_orch):
    analysis_id = "analysis-1"
    pose_result = {"pose_sequence": [_pose_frame(i) for i in range(7)]}

    xs = [0, 1, 3, 23, 32, 40, 42]
    bat_result = {
        "bat_trajectory": {
            "detections": [_bat_detection(i, x) for i, x in enumerate(xs)],
            "bat_speed_pixels_per_frame": [],
            "tracking_accuracy": 1.0,
            "tracking_failures": [],
        },
        "method": "wrist_estimation",
    }

    # No impact phase provided -> fallback estimator should run
    swing_phases_result = {"phases": {}}

    result = _run_biomechanics_analysis(
        analysis_id,
        pose_result,
        bat_result,
        swing_phases_result,
        user_profile=None,
        fps=30.0,
    )

    assert result["status"] == "completed"
    assert result["impact_frame"] in (3, 4)
    assert result["impact_frame_method"] == "smoothed_peak_speed"
    assert 0.0 < result["impact_frame_confidence"] <= 1.0
