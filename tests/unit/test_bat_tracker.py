"""Unit tests for the bat tracker module (Task 6.2).

Tests trajectory building, speed calculation, tracking accuracy,
motion blur compensation trigger, and BatTrajectory construction
with tracking failures.
"""

import math

import numpy as np
import pytest

from app.models.bat import BatDetectionResult, BatTrajectory
from app.pipeline.bat_tracker import (
    BatTracker,
    MOTION_BLUR_SPEED_THRESHOLD,
)


def _make_detection(
    frame_index: int,
    position: tuple[float, float] = (100.0, 100.0),
    detected: bool = True,
    confidence: float = 0.95,
    orientation_angle: float = 45.0,
    length_pixels: float = 150.0,
    is_predicted: bool = False,
) -> BatDetectionResult:
    """Helper to create BatDetectionResult instances."""
    return BatDetectionResult(
        frame_index=frame_index,
        detected=detected,
        position=position,
        orientation_angle=orientation_angle,
        length_pixels=length_pixels,
        confidence=confidence,
        is_predicted=is_predicted,
    )


class TestTrackTrajectory:
    """Test track_trajectory builds BatTrajectory from consecutive detections."""

    def test_empty_detections(self):
        """Empty detection list produces empty trajectory."""
        tracker = BatTracker()
        trajectory = tracker.track_trajectory([])

        assert trajectory.detections == []
        assert trajectory.bat_speed_pixels_per_frame == []
        assert trajectory.tracking_accuracy == 0.0
        assert trajectory.tracking_failures == []

    def test_single_detection(self):
        """Single detection produces trajectory with no speeds."""
        tracker = BatTracker()
        detections = [_make_detection(0, position=(100.0, 200.0))]
        trajectory = tracker.track_trajectory(detections)

        assert len(trajectory.detections) == 1
        assert trajectory.bat_speed_pixels_per_frame == []
        assert trajectory.tracking_accuracy == 1.0
        assert trajectory.tracking_failures == []

    def test_consecutive_detections_builds_trajectory(self):
        """Multiple consecutive detections build a complete trajectory."""
        tracker = BatTracker()
        detections = [
            _make_detection(0, position=(100.0, 100.0)),
            _make_detection(1, position=(110.0, 100.0)),
            _make_detection(2, position=(120.0, 100.0)),
            _make_detection(3, position=(130.0, 100.0)),
            _make_detection(4, position=(140.0, 100.0)),
        ]
        trajectory = tracker.track_trajectory(detections)

        assert len(trajectory.detections) == 5
        assert len(trajectory.bat_speed_pixels_per_frame) == 4
        assert trajectory.tracking_accuracy == 1.0
        assert trajectory.tracking_failures == []

    def test_detections_sorted_by_frame_index(self):
        """Detections are sorted by frame_index regardless of input order."""
        tracker = BatTracker()
        detections = [
            _make_detection(3, position=(130.0, 100.0)),
            _make_detection(0, position=(100.0, 100.0)),
            _make_detection(2, position=(120.0, 100.0)),
            _make_detection(1, position=(110.0, 100.0)),
        ]
        trajectory = tracker.track_trajectory(detections)

        frame_indices = [d.frame_index for d in trajectory.detections]
        assert frame_indices == [0, 1, 2, 3]

    def test_trajectory_with_mixed_detected_and_undetected(self):
        """Trajectory handles mix of detected and undetected frames."""
        tracker = BatTracker()
        detections = [
            _make_detection(0, position=(100.0, 100.0), detected=True),
            _make_detection(1, detected=False),
            _make_detection(2, detected=False),
            _make_detection(3, position=(130.0, 100.0), detected=True),
            _make_detection(4, position=(140.0, 100.0), detected=True),
        ]
        trajectory = tracker.track_trajectory(detections)

        assert len(trajectory.detections) == 5
        assert trajectory.tracking_accuracy == 3.0 / 5.0  # 60%


class TestCalculateSpeed:
    """Test _calculate_speed between consecutive detections."""

    def test_horizontal_movement(self):
        """Speed for horizontal movement is correct."""
        tracker = BatTracker()
        det1 = _make_detection(0, position=(100.0, 100.0))
        det2 = _make_detection(1, position=(150.0, 100.0))

        speed = tracker._calculate_speed(det1, det2)
        assert abs(speed - 50.0) < 0.01

    def test_vertical_movement(self):
        """Speed for vertical movement is correct."""
        tracker = BatTracker()
        det1 = _make_detection(0, position=(100.0, 100.0))
        det2 = _make_detection(1, position=(100.0, 200.0))

        speed = tracker._calculate_speed(det1, det2)
        assert abs(speed - 100.0) < 0.01

    def test_diagonal_movement(self):
        """Speed for diagonal movement uses Euclidean distance."""
        tracker = BatTracker()
        det1 = _make_detection(0, position=(0.0, 0.0))
        det2 = _make_detection(1, position=(30.0, 40.0))

        speed = tracker._calculate_speed(det1, det2)
        expected = math.sqrt(30**2 + 40**2)  # 50.0
        assert abs(speed - expected) < 0.01

    def test_no_movement(self):
        """Speed is 0 when position doesn't change."""
        tracker = BatTracker()
        det1 = _make_detection(0, position=(200.0, 300.0))
        det2 = _make_detection(1, position=(200.0, 300.0))

        speed = tracker._calculate_speed(det1, det2)
        assert speed == 0.0

    def test_speed_with_frame_gap(self):
        """Speed accounts for frame gap (non-consecutive frames)."""
        tracker = BatTracker()
        det1 = _make_detection(0, position=(100.0, 100.0))
        det2 = _make_detection(3, position=(400.0, 100.0))

        speed = tracker._calculate_speed(det1, det2)
        # Distance = 300, frame_gap = 3, speed = 100 px/frame
        assert abs(speed - 100.0) < 0.01

    def test_speed_zero_when_first_not_detected(self):
        """Speed is 0 when first detection is not detected."""
        tracker = BatTracker()
        det1 = _make_detection(0, detected=False)
        det2 = _make_detection(1, position=(200.0, 200.0))

        speed = tracker._calculate_speed(det1, det2)
        assert speed == 0.0

    def test_speed_zero_when_second_not_detected(self):
        """Speed is 0 when second detection is not detected."""
        tracker = BatTracker()
        det1 = _make_detection(0, position=(100.0, 100.0))
        det2 = _make_detection(1, detected=False)

        speed = tracker._calculate_speed(det1, det2)
        assert speed == 0.0

    def test_speed_zero_when_same_frame(self):
        """Speed is 0 when both detections are from the same frame."""
        tracker = BatTracker()
        det1 = _make_detection(5, position=(100.0, 100.0))
        det2 = _make_detection(5, position=(200.0, 200.0))

        speed = tracker._calculate_speed(det1, det2)
        assert speed == 0.0

    def test_high_speed_exceeds_threshold(self):
        """Verify high-speed detection exceeds motion blur threshold."""
        tracker = BatTracker()
        det1 = _make_detection(0, position=(100.0, 100.0))
        det2 = _make_detection(1, position=(250.0, 100.0))

        speed = tracker._calculate_speed(det1, det2)
        assert speed > MOTION_BLUR_SPEED_THRESHOLD  # 150 > 100


class TestTrackingAccuracy:
    """Test _calculate_tracking_accuracy."""

    def test_all_detected(self):
        """100% accuracy when all frames are detected."""
        tracker = BatTracker()
        detections = [
            _make_detection(i, detected=True) for i in range(10)
        ]
        accuracy = tracker._calculate_tracking_accuracy(detections)
        assert accuracy == 1.0

    def test_none_detected(self):
        """0% accuracy when no frames are detected."""
        tracker = BatTracker()
        detections = [
            _make_detection(i, detected=False) for i in range(10)
        ]
        accuracy = tracker._calculate_tracking_accuracy(detections)
        assert accuracy == 0.0

    def test_partial_detection(self):
        """Correct accuracy for partial detection."""
        tracker = BatTracker()
        detections = [
            _make_detection(0, detected=True),
            _make_detection(1, detected=True),
            _make_detection(2, detected=False),
            _make_detection(3, detected=True),
            _make_detection(4, detected=False),
        ]
        accuracy = tracker._calculate_tracking_accuracy(detections)
        assert abs(accuracy - 0.6) < 0.01

    def test_empty_detections(self):
        """0% accuracy for empty detection list."""
        tracker = BatTracker()
        accuracy = tracker._calculate_tracking_accuracy([])
        assert accuracy == 0.0

    def test_85_percent_accuracy_target(self):
        """Verify 85% accuracy threshold scenario."""
        tracker = BatTracker()
        # 17 detected out of 20 = 85%
        detections = [
            _make_detection(i, detected=(i not in [3, 7, 15]))
            for i in range(20)
        ]
        accuracy = tracker._calculate_tracking_accuracy(detections)
        assert accuracy == 17.0 / 20.0
        assert accuracy >= 0.85


class TestMotionBlurCompensation:
    """Test _compensate_motion_blur trigger and behavior."""

    def test_no_compensation_below_threshold(self):
        """No motion blur compensation when speed <= 100 px/frame."""
        tracker = BatTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        prev_det = _make_detection(5, position=(200.0, 200.0))

        result = tracker._compensate_motion_blur(frame, prev_det, speed=80.0)

        assert result.frame_index == 6
        assert result.detected is True
        assert result.is_predicted is True
        assert result.position == (200.0, 200.0)
        # Confidence unchanged when below threshold
        assert result.confidence == prev_det.confidence

    def test_compensation_triggered_above_threshold(self):
        """Motion blur compensation is triggered when speed > 100 px/frame."""
        tracker = BatTracker()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        prev_det = _make_detection(5, position=(300.0, 240.0))

        result = tracker._compensate_motion_blur(frame, prev_det, speed=150.0)

        assert result.frame_index == 6
        assert result.detected is True
        assert result.is_predicted is True
        # Confidence is adjusted (reduced) for high-speed compensation
        assert result.confidence < prev_det.confidence

    def test_compensation_at_exact_threshold(self):
        """At exactly 100 px/frame, no compensation is applied."""
        tracker = BatTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        prev_det = _make_detection(5, position=(200.0, 200.0))

        result = tracker._compensate_motion_blur(frame, prev_det, speed=100.0)

        # At threshold, no compensation (<=)
        assert result.is_predicted is True
        assert result.confidence == prev_det.confidence

    def test_compensation_very_high_speed(self):
        """Very high speed (>200 px/frame) still produces a result."""
        tracker = BatTracker()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        prev_det = _make_detection(10, position=(320.0, 240.0))

        result = tracker._compensate_motion_blur(frame, prev_det, speed=250.0)

        assert result.frame_index == 11
        assert result.detected is True
        assert result.confidence > 0.0

    def test_compensation_with_edge_position(self):
        """Compensation handles bat near frame edge."""
        tracker = BatTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Position near top-left corner
        prev_det = _make_detection(5, position=(10.0, 10.0))

        result = tracker._compensate_motion_blur(frame, prev_det, speed=120.0)

        assert result.frame_index == 6
        assert result.detected is True


class TestBatTrajectoryWithFailures:
    """Test BatTrajectory construction with tracking failures."""

    def test_single_failure_interval(self):
        """Single gap of undetected frames creates one failure interval."""
        tracker = BatTracker()
        detections = [
            _make_detection(0, position=(100.0, 100.0), detected=True),
            _make_detection(1, position=(110.0, 100.0), detected=True),
            _make_detection(2, detected=False),
            _make_detection(3, detected=False),
            _make_detection(4, detected=False),
            _make_detection(5, position=(160.0, 100.0), detected=True),
        ]
        trajectory = tracker.track_trajectory(detections)

        assert len(trajectory.tracking_failures) == 1
        assert trajectory.tracking_failures[0] == (2, 4)

    def test_multiple_failure_intervals(self):
        """Multiple gaps create multiple failure intervals."""
        tracker = BatTracker()
        detections = [
            _make_detection(0, detected=True, position=(100.0, 100.0)),
            _make_detection(1, detected=False),
            _make_detection(2, detected=True, position=(120.0, 100.0)),
            _make_detection(3, detected=False),
            _make_detection(4, detected=False),
            _make_detection(5, detected=True, position=(150.0, 100.0)),
        ]
        trajectory = tracker.track_trajectory(detections)

        assert len(trajectory.tracking_failures) == 2
        assert trajectory.tracking_failures[0] == (1, 1)
        assert trajectory.tracking_failures[1] == (3, 4)

    def test_failure_at_start(self):
        """Failure at the beginning of the sequence."""
        tracker = BatTracker()
        detections = [
            _make_detection(0, detected=False),
            _make_detection(1, detected=False),
            _make_detection(2, detected=True, position=(120.0, 100.0)),
            _make_detection(3, detected=True, position=(130.0, 100.0)),
        ]
        trajectory = tracker.track_trajectory(detections)

        assert len(trajectory.tracking_failures) == 1
        assert trajectory.tracking_failures[0] == (0, 1)

    def test_failure_at_end(self):
        """Failure at the end of the sequence."""
        tracker = BatTracker()
        detections = [
            _make_detection(0, detected=True, position=(100.0, 100.0)),
            _make_detection(1, detected=True, position=(110.0, 100.0)),
            _make_detection(2, detected=False),
            _make_detection(3, detected=False),
        ]
        trajectory = tracker.track_trajectory(detections)

        assert len(trajectory.tracking_failures) == 1
        assert trajectory.tracking_failures[0] == (2, 3)

    def test_no_failures(self):
        """No failures when all frames are detected."""
        tracker = BatTracker()
        detections = [
            _make_detection(i, detected=True, position=(100.0 + i * 10, 100.0))
            for i in range(5)
        ]
        trajectory = tracker.track_trajectory(detections)

        assert trajectory.tracking_failures == []

    def test_all_failures(self):
        """All frames undetected creates one large failure interval."""
        tracker = BatTracker()
        detections = [
            _make_detection(i, detected=False) for i in range(5)
        ]
        trajectory = tracker.track_trajectory(detections)

        assert len(trajectory.tracking_failures) == 1
        assert trajectory.tracking_failures[0] == (0, 4)

    def test_trajectory_speeds_with_gaps(self):
        """Speed calculation handles undetected frames (returns 0)."""
        tracker = BatTracker()
        detections = [
            _make_detection(0, position=(100.0, 100.0), detected=True),
            _make_detection(1, detected=False),
            _make_detection(2, position=(200.0, 100.0), detected=True),
        ]
        trajectory = tracker.track_trajectory(detections)

        # Speed between frame 0 (detected) and frame 1 (not detected) = 0
        # Speed between frame 1 (not detected) and frame 2 (detected) = 0
        assert len(trajectory.bat_speed_pixels_per_frame) == 2
        assert trajectory.bat_speed_pixels_per_frame[0] == 0.0
        assert trajectory.bat_speed_pixels_per_frame[1] == 0.0
