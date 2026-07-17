"""Unit tests for impact-frame fallback estimation and confidence metadata."""

from unittest.mock import patch

from app.tasks.pipeline import (
    _estimate_impact_frame_from_bat_speed,
    _run_biomechanics_analysis,
)


def _bat_detection(
    frame_index: int,
    x: float,
    *,
    y: float = 0.0,
    detected: bool = True,
    is_predicted: bool = False,
    coordinate_space: str = "pixel",
) -> dict:
    return {
        "frame_index": frame_index,
        "detected": detected,
        "position": [x, y],
        "orientation_angle": 0.0,
        "length_pixels": 10.0,
        "confidence": 0.9,
        "is_predicted": is_predicted,
        "coordinate_space": coordinate_space,
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


def test_estimate_impact_frame_ignores_predicted_motion_outlier():
    """Interpolated wrist motion cannot redefine observed contact timing."""
    detections = [
        _bat_detection(0, 0.0),
        _bat_detection(1, 1.0),
        _bat_detection(2, 3.0),
        _bat_detection(3, 6.0),
        _bat_detection(4, 1000.0, is_predicted=True),
    ]

    class _Traj:
        def __init__(self, detections):
            self.detections = detections

    impact_frame, _, method = _estimate_impact_frame_from_bat_speed(
        _Traj(detections)
    )

    assert method == "smoothed_peak_speed"
    assert impact_frame != 4


def test_estimate_impact_frame_normalizes_sparse_observation_gaps():
    """A long detector gap must not masquerade as a high per-frame speed."""
    detections = [
        _bat_detection(0, 0.0),
        _bat_detection(10, 20.0),
        _bat_detection(11, 25.0),
        _bat_detection(12, 30.0),
    ]

    class _Traj:
        def __init__(self, detections):
            self.detections = detections

    impact_frame, _, _ = _estimate_impact_frame_from_bat_speed(
        _Traj(detections)
    )

    assert impact_frame == 11


def test_estimate_impact_frame_corrects_normalized_aspect_ratio():
    """Equivalent pixel motion yields one contact frame in portrait and landscape."""
    physical_points = [(0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (40.0, 60.0)]
    impact_frames = []
    for width, height in ((200.0, 100.0), (50.0, 100.0)):
        detections = [
            _bat_detection(
                frame_index,
                x / width,
                y=y / height,
                coordinate_space="normalized",
            )
            for frame_index, (x, y) in enumerate(physical_points)
        ]

        class _Traj:
            def __init__(self, detections):
                self.detections = detections

        impact_frame, _, _ = _estimate_impact_frame_from_bat_speed(
            _Traj(detections),
            video_width=width,
            video_height=height,
        )
        impact_frames.append(impact_frame)

    assert impact_frames == [1, 1]


@patch("app.pipeline.biomechanics_analyzer.BiomechanicsOrchestrator")
def test_biomechanics_abstains_without_observed_impact_anchor(mock_orchestrator):
    """Predicted wrist points cannot anchor contact-relative measurements."""
    pose_result = {"pose_sequence": [_pose_frame(i) for i in range(7)]}
    predicted = [
        _bat_detection(i, float(i), is_predicted=True)
        for i in range(7)
    ]

    result = _run_biomechanics_analysis(
        "analysis-predicted-only",
        pose_result,
        {
            "bat_trajectory": {
                "detections": predicted,
                "bat_speed_pixels_per_frame": [],
                "tracking_accuracy": 0.0,
                "tracking_failures": [],
            }
        },
        {"phases": {}},
        user_profile=None,
        fps=30.0,
    )

    mock_orchestrator.assert_not_called()
    assert result["status"] == "completed"
    assert result["impact_frame"] is None
    assert result["impact_frame_method"] == "unavailable"
    assert result["bat_speed"] is None
    assert result["attack_angle"] is None
    assert result["rotation"] is None
    assert result["hand_path_efficiency"] is None
    assert result["unmeasurable_metrics"] == [
        {
            "metric_name": "impact_anchor",
            "reason": (
                "No detector-observed bat evidence or explicit impact phase "
                "was available"
            ),
        }
    ]


@patch("app.pipeline.biomechanics_analyzer.BiomechanicsOrchestrator")
def test_explicit_impact_at_frame_zero_is_not_treated_as_absent(
    mock_orchestrator,
):
    """Frame zero is a valid explicit event anchor, not an absence sentinel."""
    from app.models.biomechanics import BiomechanicsResult

    mock_orchestrator.return_value.analyze.return_value = BiomechanicsResult()
    result = _run_biomechanics_analysis(
        "analysis-frame-zero",
        {"pose_sequence": [_pose_frame(i) for i in range(7)]},
        {"bat_trajectory": {"detections": []}},
        {"phases": {"impact": [0, 0]}},
        user_profile=None,
        fps=30.0,
    )

    assert result["impact_frame"] == 0
    assert result["impact_frame_method"] == "phase_start"
    assert mock_orchestrator.return_value.analyze.call_args.kwargs[
        "impact_frame"
    ] == 0


@patch(
    "app.pipeline.biomechanics_analyzer.BiomechanicsOrchestrator",
    return_value=_StubOrchestrator(),
)
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
