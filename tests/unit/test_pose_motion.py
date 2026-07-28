"""Tests for confidence-gated pose-motion contact anchoring."""

from __future__ import annotations

from unittest.mock import patch

from app.models.pose import Keypoint, PoseResult
from app.pipeline.pose_motion import estimate_impact_from_pose_motion
from app.tasks.pipeline import (
    _run_biomechanics_analysis,
    _run_swing_classification,
    _serialize_dataclass,
)


def _pose(frame_index: int, hand_x: float, *, torso: bool = True) -> PoseResult:
    keypoints = [
        Keypoint(hand_x - 0.01, 0.44, 0.0, 0.9, "left_wrist"),
        Keypoint(hand_x + 0.01, 0.44, 0.0, 0.9, "right_wrist"),
        Keypoint(0.43, 0.82, 0.0, 0.9, "left_ankle"),
        Keypoint(0.57, 0.82, 0.0, 0.9, "right_ankle"),
    ]
    if torso:
        keypoints.extend(
            [
                Keypoint(0.44, 0.30, 0.0, 0.9, "left_shoulder"),
                Keypoint(0.56, 0.30, 0.0, 0.9, "right_shoulder"),
                Keypoint(0.45, 0.58, 0.0, 0.9, "left_hip"),
                Keypoint(0.55, 0.58, 0.0, 0.9, "right_hip"),
                Keypoint(0.46, 0.70, 0.0, 0.9, "left_knee"),
                Keypoint(0.54, 0.70, 0.0, 0.9, "right_knee"),
                Keypoint(0.50, 0.18, 0.0, 0.9, "nose"),
            ]
        )
    return PoseResult(
        frame_index=frame_index,
        keypoints=keypoints,
        person_id=0,
        is_primary_batter=True,
        overall_confidence=0.9,
        is_low_confidence=False,
    )


def _swing_poses() -> list[PoseResult]:
    increments = {
        21: 0.003,
        22: 0.005,
        23: 0.008,
        24: 0.014,
        25: 0.024,
        26: 0.040,
        27: 0.055,
        28: 0.044,
        29: 0.028,
        30: 0.015,
        31: 0.008,
        32: 0.004,
    }
    hand_x = 0.36
    poses = []
    for frame_index in range(50):
        hand_x += increments.get(frame_index, 0.0002)
        poses.append(_pose(frame_index, hand_x))
    return poses


def _repeated_motion_poses() -> list[PoseResult]:
    hand_x = 0.30
    poses = []
    for frame_index in range(80):
        if 18 <= frame_index <= 28:
            step = (0.006, 0.010, 0.018, 0.030, 0.045, 0.050)[
                min(frame_index - 18, 5)
            ]
        elif 55 <= frame_index <= 66:
            step = (0.008, 0.014, 0.024, 0.040, 0.060, 0.070)[
                min(frame_index - 55, 5)
            ]
        else:
            step = 0.0002
        hand_x += step
        poses.append(_pose(frame_index, hand_x))
    return poses


def test_pose_motion_estimates_supported_hand_speed_peak() -> None:
    estimate = estimate_impact_from_pose_motion(
        _swing_poses(),
        video_width=1280,
        video_height=720,
        fps=30.0,
    )

    assert estimate is not None
    assert 25 <= estimate.frame_index <= 29
    assert estimate.confidence >= 0.55
    assert estimate.method == "pose_hand_speed_peak"
    assert estimate.evidence["supports_bat_metrics"] is False


def test_pose_motion_abstains_for_stationary_hands() -> None:
    poses = [_pose(frame_index, 0.4) for frame_index in range(40)]

    assert (
        estimate_impact_from_pose_motion(
            poses,
            video_width=1280,
            video_height=720,
            fps=30.0,
        )
        is None
    )


def test_pose_motion_ignores_camera_translation() -> None:
    poses = []
    shift = 0.0
    for frame_index in range(40):
        if 14 <= frame_index <= 22:
            shift += 0.025
        pose = _pose(frame_index, 0.4 + shift)
        pose.keypoints = [
            keypoint
            if keypoint.name in {"left_wrist", "right_wrist"}
            else Keypoint(
                keypoint.x + shift,
                keypoint.y,
                keypoint.z,
                keypoint.confidence,
                keypoint.name,
            )
            for keypoint in pose.keypoints
        ]
        poses.append(pose)

    assert (
        estimate_impact_from_pose_motion(
            poses,
            video_width=1280,
            video_height=720,
            fps=30.0,
        )
        is None
    )


def test_pose_motion_requires_torso_support() -> None:
    poses = [
        _pose(pose.frame_index, pose.keypoints[0].x + 0.01, torso=False)
        for pose in _swing_poses()
    ]

    assert (
        estimate_impact_from_pose_motion(
            poses,
            video_width=1280,
            video_height=720,
            fps=30.0,
        )
        is None
    )


def test_pose_motion_prefers_swing_burst_before_stronger_reset() -> None:
    estimate = estimate_impact_from_pose_motion(
        _repeated_motion_poses(),
        video_width=1280,
        video_height=720,
        fps=30.0,
    )

    assert estimate is not None
    assert estimate.frame_index < 40
    assert estimate.evidence["segment_selection"] == (
        "earliest_substantial_burst"
    )
    assert estimate.evidence["multiple_swing_detected"] is True
    assert estimate.evidence["substantial_motion_segment_count"] == 2
    assert estimate.evidence["selected_window_end_frame"] < 55


def test_pipeline_isolates_repeated_clip_before_downstream_analysis() -> None:
    poses = _repeated_motion_poses()
    phase_result = _run_swing_classification(
        "analysis-id",
        {"pose_sequence": _serialize_dataclass(poses)},
        {"bat_trajectory": {}},
        30.0,
        video_width=1280,
        video_height=720,
    )

    swing_window = phase_result["swing_window"]
    assert swing_window["multiple_swing_detected"] is True
    assert swing_window["isolation_applied"] is True
    assert swing_window["candidate_count"] == 2
    assert swing_window["end_frame"] < 55
    assert all(
        swing_window["start_frame"] <= frame <= swing_window["end_frame"]
        for phase_range in phase_result["phases"].values()
        for frame in phase_range
    )

    with patch(
        "app.pipeline.biomechanics_analyzer.BiomechanicsOrchestrator"
    ) as orchestrator_class:
        orchestrator_class.return_value.analyze.return_value = {}
        _run_biomechanics_analysis(
            "analysis-id",
            {"pose_sequence": _serialize_dataclass(poses)},
            {"bat_trajectory": {}},
            phase_result,
            {
                "height": 180.0,
                "bat_length": 33.0,
                "batting_direction": "right",
            },
            30.0,
            {"video_width": 1280, "video_height": 720, "fps": 30.0},
        )

    downstream_poses = orchestrator_class.return_value.analyze.call_args.kwargs[
        "pose_sequence"
    ]
    assert downstream_poses[0].frame_index >= swing_window["start_frame"]
    assert downstream_poses[-1].frame_index <= swing_window["end_frame"]
    assert downstream_poses[-1].frame_index < 55


def test_pipeline_adds_pose_motion_anchor_without_claiming_bat_support() -> None:
    result = _run_swing_classification(
        "analysis-id",
        {"pose_sequence": _serialize_dataclass(_swing_poses())},
        {"bat_trajectory": {}},
        30.0,
        video_width=1280,
        video_height=720,
    )

    assert "impact" in result["phases"]
    assert result["phase_source"].endswith("pose_motion_contact")
    assert result["phase_evidence"]["impact_method"] == "pose_hand_speed_peak"
    assert result["phase_evidence"]["impact_support"] == "tracked_body_motion"
    assert result["phase_evidence"]["supports_bat_metrics"] is False


def test_pose_motion_anchor_unlocks_body_metrics_but_not_bat_metrics() -> None:
    poses = _swing_poses()
    phase_result = _run_swing_classification(
        "analysis-id",
        {"pose_sequence": _serialize_dataclass(poses)},
        {"bat_trajectory": {}},
        30.0,
        video_width=1280,
        video_height=720,
    )

    result = _run_biomechanics_analysis(
        "analysis-id",
        {"pose_sequence": _serialize_dataclass(poses)},
        {"bat_trajectory": {}},
        phase_result,
        {
            "height": 180.0,
            "bat_length": 33.0,
            "batting_direction": "right",
        },
        30.0,
        {"video_width": 1280, "video_height": 720, "fps": 30.0},
    )

    unmeasurable_names = {
        metric["metric_name"] for metric in result["unmeasurable_metrics"]
    }
    assert "impact_anchor" not in unmeasurable_names
    assert "all" not in unmeasurable_names
    assert result["bat_speed"] is None
    assert result["attack_angle"] is None
    assert result["hand_path_efficiency"] is not None
    assert result["stride_length_cm"] is not None
    assert result["head_stability_cm"] is not None
    assert result["front_knee_extension_degrees"] is not None
    assert result["spine_angle_degrees"] is not None
