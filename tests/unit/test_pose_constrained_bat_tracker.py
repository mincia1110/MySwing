"""Unit tests for pose-constrained bat line tracking."""

import numpy as np
import pytest

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.pose import Keypoint, PoseResult
from app.pipeline.pose_constrained_bat_tracker import PoseConstrainedBatTracker


def _keypoint(name: str, x: float, y: float, confidence: float = 0.9) -> Keypoint:
    return Keypoint(x=x, y=y, z=0.0, confidence=confidence, name=name)


def _pose(frame_index: int, wrist: tuple[float, float] = (0.5, 0.5)) -> PoseResult:
    wx, wy = wrist
    return PoseResult(
        frame_index=frame_index,
        keypoints=[
            _keypoint("right_wrist", wx, wy),
            _keypoint("right_elbow", wx - 0.12, wy),
        ],
        person_id=0,
        is_primary_batter=True,
        overall_confidence=0.9,
        is_low_confidence=False,
    )


def _frame(width: int = 120, height: int = 100) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _draw_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    value: int = 255,
) -> None:
    x1, y1 = start
    x2, y2 = end
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        frame[max(0, y - 1): y + 2, max(0, x - 1): x + 2] = value


def _prior(count: int = 3) -> BatTrajectory:
    detections = [
        BatDetectionResult(
            frame_index=i,
            detected=True,
            position=(0.60 + i * 0.01, 0.50),
            orientation_angle=0.0,
            length_pixels=0.25,
            confidence=0.5,
            is_predicted=True,
            coordinate_space="normalized",
            bat_head_position=(0.72 + i * 0.01, 0.50),
        )
        for i in range(count)
    ]
    return BatTrajectory(detections=detections, tracking_accuracy=1.0)


def test_empty_frames_returns_fallback_trajectory() -> None:
    prior = _prior(2)
    tracker = PoseConstrainedBatTracker()

    result = tracker.track([], [_pose(0)], 120, 100, 30.0, wrist_prior=prior)

    assert result == prior


def test_empty_pose_sequence_returns_fallback_or_no_detections() -> None:
    tracker = PoseConstrainedBatTracker()
    frames = [_frame()]

    no_prior = tracker.track(frames, [], 120, 100, 30.0)
    with_prior = tracker.track(frames, [], 120, 100, 30.0, wrist_prior=_prior(1))

    assert len(no_prior.detections) == 1
    assert no_prior.detections[0].detected is False
    assert with_prior.detections[0].is_predicted is True


def test_synthetic_white_line_near_wrist_is_detected() -> None:
    frame = _frame()
    _draw_line(frame, (58, 50), (94, 50))
    tracker = PoseConstrainedBatTracker()

    result = tracker.track([frame], [_pose(0)], 120, 100, 30.0)

    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.detected is True
    assert detection.is_predicted is False
    assert detection.confidence > 0.25


def test_farther_endpoint_is_selected_as_bat_head_position() -> None:
    frame = _frame()
    # Hand at (0.5, 0.5) = (60, 50); line from hand toward right
    # The farther endpoint (90, 50) should be chosen as bat_head.
    _draw_line(frame, (60, 50), (90, 50))
    tracker = PoseConstrainedBatTracker()

    result = tracker.track([frame], [_pose(0, wrist=(0.5, 0.5))], 120, 100, 30.0)

    head = result.detections[0].bat_head_position
    assert head is not None
    # 90/120 = 0.75; bat_head must be the endpoint farther from hand
    assert head[0] > 0.65
    assert head[1] == pytest.approx(0.5, abs=0.04)


def test_temporal_smoothing_avoids_one_frame_angle_jump() -> None:
    frames = [_frame() for _ in range(3)]
    for frame in frames:
        _draw_line(frame, (58, 50), (96, 50))
    _draw_line(frames[1], (54, 54), (54, 92))
    tracker = PoseConstrainedBatTracker()

    result = tracker.track(frames, [_pose(0), _pose(1), _pose(2)], 120, 100, 30.0)

    angles = [d.orientation_angle for d in result.detections if d.detected]
    assert len(angles) == 3
    assert max(abs(angle) for angle in angles) < 25.0


def test_low_confidence_no_line_case_falls_back_to_wrist_prior() -> None:
    frames = [_frame() for _ in range(4)]
    prior = _prior(4)
    tracker = PoseConstrainedBatTracker()

    result = tracker.track(frames, [_pose(i) for i in range(4)], 120, 100, 30.0, wrist_prior=prior)

    assert result == prior


def test_tracking_result_uses_normalized_coordinate_space() -> None:
    frame = _frame()
    _draw_line(frame, (58, 50), (94, 50))
    tracker = PoseConstrainedBatTracker()

    result = tracker.track([frame], [_pose(0)], 120, 100, 30.0)

    assert result.detections[0].coordinate_space == "normalized"
    assert 0.0 <= result.detections[0].position[0] <= 1.0
    assert 0.0 <= result.detections[0].position[1] <= 1.0


def test_tracking_accuracy_is_clamped_to_unit_interval() -> None:
    frames = [_frame() for _ in range(3)]
    _draw_line(frames[0], (58, 50), (94, 50))
    _draw_line(frames[1], (58, 50), (94, 50))
    tracker = PoseConstrainedBatTracker()

    result = tracker.track(frames, [_pose(0), _pose(1), _pose(2)], 120, 100, 30.0)

    assert 0.0 <= result.tracking_accuracy <= 1.0
