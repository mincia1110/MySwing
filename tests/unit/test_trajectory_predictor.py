"""Unit tests for the trajectory predictor module (Task 6.3).

Tests trajectory prediction from previous frames, tracking failure handling,
confidence reduction, and re-detection reset behavior.

Validates: Requirements 4.4, 4.5
"""

import pytest

from app.models.bat import BatDetectionResult, BatTrajectory
from app.pipeline.trajectory_predictor import TrajectoryPredictor, MAX_PREDICTION_FRAMES


def _make_detection(
    frame_index: int,
    position: tuple[float, float] = (100.0, 200.0),
    orientation_angle: float = 45.0,
    length_pixels: float = 150.0,
    confidence: float = 0.95,
    detected: bool = True,
    is_predicted: bool = False,
) -> BatDetectionResult:
    """Helper to create a BatDetectionResult."""
    return BatDetectionResult(
        frame_index=frame_index,
        detected=detected,
        position=position,
        orientation_angle=orientation_angle,
        length_pixels=length_pixels,
        confidence=confidence,
        is_predicted=is_predicted,
    )


class TestPredictionFromSingleFrame:
    """Test prediction from 1 previous frame (constant velocity assumption)."""

    def test_single_frame_returns_same_position(self):
        """With only 1 previous detection, predicted position equals last known."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=5)

        previous = [_make_detection(frame_index=4, position=(100.0, 200.0))]
        result = predictor.predict_position(previous, current_frame_index=5)

        assert result is not None
        assert result.position == (100.0, 200.0)
        assert result.frame_index == 5

    def test_single_frame_preserves_orientation(self):
        """With only 1 previous detection, orientation is preserved."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=5)

        previous = [_make_detection(frame_index=4, orientation_angle=90.0)]
        result = predictor.predict_position(previous, current_frame_index=5)

        assert result is not None
        assert result.orientation_angle == 90.0

    def test_single_frame_preserves_length(self):
        """With only 1 previous detection, length is preserved."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=5)

        previous = [_make_detection(frame_index=4, length_pixels=180.0)]
        result = predictor.predict_position(previous, current_frame_index=5)

        assert result is not None
        assert result.length_pixels == 180.0


class TestPredictionFromMultipleFrames:
    """Test prediction from up to 5 previous frames (linear extrapolation)."""

    def test_two_frames_linear_extrapolation(self):
        """With 2 previous detections, extrapolates linearly."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=2)

        previous = [
            _make_detection(frame_index=0, position=(100.0, 200.0)),
            _make_detection(frame_index=1, position=(110.0, 210.0)),
        ]
        result = predictor.predict_position(previous, current_frame_index=2)

        assert result is not None
        # Velocity: (10, 10) per frame, so next position: (120, 220)
        assert abs(result.position[0] - 120.0) < 0.01
        assert abs(result.position[1] - 220.0) < 0.01

    def test_five_frames_linear_extrapolation(self):
        """With 5 previous detections, uses average velocity for extrapolation."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=5)

        # Constant velocity: +20 pixels/frame in x, +10 in y
        previous = [
            _make_detection(frame_index=0, position=(100.0, 200.0)),
            _make_detection(frame_index=1, position=(120.0, 210.0)),
            _make_detection(frame_index=2, position=(140.0, 220.0)),
            _make_detection(frame_index=3, position=(160.0, 230.0)),
            _make_detection(frame_index=4, position=(180.0, 240.0)),
        ]
        result = predictor.predict_position(previous, current_frame_index=5)

        assert result is not None
        # Average velocity: (180-100)/4 = 20, (240-200)/4 = 10
        # Predicted: (180+20, 240+10) = (200, 250)
        assert abs(result.position[0] - 200.0) < 0.01
        assert abs(result.position[1] - 250.0) < 0.01

    def test_uses_at_most_5_previous_detections(self):
        """When more than 5 detections available, uses only last 5."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=7)

        # 7 detections, but only last 5 should be used
        previous = [
            _make_detection(frame_index=0, position=(0.0, 0.0)),
            _make_detection(frame_index=1, position=(10.0, 10.0)),
            _make_detection(frame_index=2, position=(100.0, 200.0)),
            _make_detection(frame_index=3, position=(120.0, 210.0)),
            _make_detection(frame_index=4, position=(140.0, 220.0)),
            _make_detection(frame_index=5, position=(160.0, 230.0)),
            _make_detection(frame_index=6, position=(180.0, 240.0)),
        ]
        result = predictor.predict_position(previous, current_frame_index=7)

        assert result is not None
        # Last 5: frames 2-6, velocity: (180-100)/4=20, (240-200)/4=10
        # Predicted: (180+20, 240+10) = (200, 250)
        assert abs(result.position[0] - 200.0) < 0.01
        assert abs(result.position[1] - 250.0) < 0.01

    def test_non_uniform_velocity_averages(self):
        """Non-uniform velocity is averaged across available frames."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=3)

        previous = [
            _make_detection(frame_index=0, position=(100.0, 100.0)),
            _make_detection(frame_index=1, position=(110.0, 120.0)),  # +10, +20
            _make_detection(frame_index=2, position=(140.0, 130.0)),  # +30, +10
        ]
        result = predictor.predict_position(previous, current_frame_index=3)

        assert result is not None
        # Average velocity: (140-100)/2=20, (130-100)/2=15
        # Predicted: (140+20, 130+15) = (160, 145)
        assert abs(result.position[0] - 160.0) < 0.01
        assert abs(result.position[1] - 145.0) < 0.01


class TestPredictionCeasesAfterFiveFrames:
    """Test that prediction ceases after 5 consecutive missing frames."""

    def test_prediction_at_5_missing_frames(self):
        """Prediction still works at exactly 5 consecutive missing frames."""
        predictor = TrajectoryPredictor()
        # Simulate 5 consecutive missing frames
        for i in range(5):
            predictor.notify_missing(frame_index=5 + i)

        previous = [_make_detection(frame_index=4, position=(100.0, 200.0))]
        result = predictor.predict_position(previous, current_frame_index=10)

        assert result is not None

    def test_prediction_ceases_at_6_missing_frames(self):
        """Prediction ceases when consecutive missing exceeds 5."""
        predictor = TrajectoryPredictor()
        # Simulate 6 consecutive missing frames
        for i in range(6):
            predictor.notify_missing(frame_index=5 + i)

        previous = [_make_detection(frame_index=4, position=(100.0, 200.0))]
        result = predictor.predict_position(previous, current_frame_index=11)

        assert result is None

    def test_prediction_ceases_at_many_missing_frames(self):
        """Prediction ceases for any number > 5 consecutive missing frames."""
        predictor = TrajectoryPredictor()
        for i in range(10):
            predictor.notify_missing(frame_index=5 + i)

        previous = [_make_detection(frame_index=4, position=(100.0, 200.0))]
        result = predictor.predict_position(previous, current_frame_index=15)

        assert result is None


class TestPredictedResultMarking:
    """Test that predicted results have is_predicted=True."""

    def test_predicted_result_has_is_predicted_true(self):
        """Predicted results are marked with is_predicted=True."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=5)

        previous = [_make_detection(frame_index=4, position=(100.0, 200.0))]
        result = predictor.predict_position(previous, current_frame_index=5)

        assert result is not None
        assert result.is_predicted is True

    def test_predicted_result_has_detected_true(self):
        """Predicted results still have detected=True (position is estimated)."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=5)

        previous = [_make_detection(frame_index=4, position=(100.0, 200.0))]
        result = predictor.predict_position(previous, current_frame_index=5)

        assert result is not None
        assert result.detected is True

    def test_predicted_result_has_correct_frame_index(self):
        """Predicted results have the correct current frame index."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=42)

        previous = [_make_detection(frame_index=41, position=(100.0, 200.0))]
        result = predictor.predict_position(previous, current_frame_index=42)

        assert result is not None
        assert result.frame_index == 42


class TestConfidenceDecreases:
    """Test that confidence decreases with gap length."""

    def test_confidence_decreases_with_gap(self):
        """Confidence decreases as consecutive missing frames increase."""
        base_confidence = 0.95
        previous = [_make_detection(frame_index=0, position=(100.0, 200.0), confidence=base_confidence)]

        confidences = []
        for gap in range(1, 6):
            predictor = TrajectoryPredictor()
            for i in range(gap):
                predictor.notify_missing(frame_index=1 + i)
            result = predictor.predict_position(previous, current_frame_index=1 + gap - 1)
            assert result is not None
            confidences.append(result.confidence)

        # Each subsequent confidence should be lower
        for i in range(1, len(confidences)):
            assert confidences[i] < confidences[i - 1], (
                f"Confidence at gap {i+1} ({confidences[i]}) should be less than "
                f"at gap {i} ({confidences[i-1]})"
            )

    def test_confidence_at_gap_1_less_than_base(self):
        """Confidence at 1 missing frame is less than base confidence."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=1)

        previous = [_make_detection(frame_index=0, confidence=0.95)]
        result = predictor.predict_position(previous, current_frame_index=1)

        assert result is not None
        assert result.confidence < 0.95

    def test_confidence_at_gap_5_still_positive(self):
        """Confidence at 5 missing frames is still positive."""
        predictor = TrajectoryPredictor()
        for i in range(5):
            predictor.notify_missing(frame_index=1 + i)

        previous = [_make_detection(frame_index=0, confidence=0.95)]
        result = predictor.predict_position(previous, current_frame_index=5)

        assert result is not None
        assert result.confidence > 0.0

    def test_confidence_never_negative(self):
        """Confidence is never negative."""
        predictor = TrajectoryPredictor()
        for i in range(5):
            predictor.notify_missing(frame_index=1 + i)

        previous = [_make_detection(frame_index=0, confidence=0.1)]
        result = predictor.predict_position(previous, current_frame_index=5)

        assert result is not None
        assert result.confidence >= 0.0


class TestTrackingFailureRecording:
    """Test tracking failure recording in BatTrajectory."""

    def test_record_failure_interval(self):
        """Records failure interval (start_frame, end_frame) in trajectory."""
        predictor = TrajectoryPredictor()
        trajectory = BatTrajectory()

        # Simulate 6 missing frames starting at frame 10
        for i in range(6):
            predictor.notify_missing(frame_index=10 + i)

        # Record the failure when bat is re-detected at frame 16
        predictor.record_tracking_failure(trajectory, end_frame_index=16)

        assert len(trajectory.tracking_failures) == 1
        assert trajectory.tracking_failures[0] == (10, 16)

    def test_multiple_failure_intervals(self):
        """Can record multiple failure intervals."""
        predictor = TrajectoryPredictor()
        trajectory = BatTrajectory()

        # First failure: frames 5-12
        for i in range(7):
            predictor.notify_missing(frame_index=5 + i)
        predictor.record_tracking_failure(trajectory, end_frame_index=12)

        # Reset and second failure: frames 20-28
        predictor.reset()
        for i in range(8):
            predictor.notify_missing(frame_index=20 + i)
        predictor.record_tracking_failure(trajectory, end_frame_index=28)

        assert len(trajectory.tracking_failures) == 2
        assert trajectory.tracking_failures[0] == (5, 12)
        assert trajectory.tracking_failures[1] == (20, 28)

    def test_no_failure_recorded_without_missing(self):
        """No failure is recorded if no frames were missing."""
        predictor = TrajectoryPredictor()
        trajectory = BatTrajectory()

        predictor.record_tracking_failure(trajectory, end_frame_index=10)

        assert len(trajectory.tracking_failures) == 0


class TestReDetectionResetsCounter:
    """Test that re-detection resets the consecutive missing counter."""

    def test_notify_detected_resets_counter(self):
        """notify_detected resets consecutive_missing to 0."""
        predictor = TrajectoryPredictor()

        # Simulate 3 missing frames
        for i in range(3):
            predictor.notify_missing(frame_index=5 + i)
        assert predictor.consecutive_missing == 3

        # Re-detect
        predictor.notify_detected()
        assert predictor.consecutive_missing == 0

    def test_prediction_works_after_reset(self):
        """After re-detection and new missing frames, prediction works again."""
        predictor = TrajectoryPredictor()

        # First gap: 6 frames (exceeds threshold)
        for i in range(6):
            predictor.notify_missing(frame_index=5 + i)

        previous = [_make_detection(frame_index=4, position=(100.0, 200.0))]
        result = predictor.predict_position(previous, current_frame_index=11)
        assert result is None  # Prediction ceased

        # Re-detect at frame 11
        predictor.notify_detected()

        # New gap: 1 frame
        predictor.notify_missing(frame_index=13)

        new_previous = [_make_detection(frame_index=12, position=(200.0, 300.0))]
        result = predictor.predict_position(new_previous, current_frame_index=13)
        assert result is not None  # Prediction works again
        assert result.position == (200.0, 300.0)

    def test_reset_method_clears_state(self):
        """reset() clears all internal state."""
        predictor = TrajectoryPredictor()

        for i in range(4):
            predictor.notify_missing(frame_index=10 + i)
        assert predictor.consecutive_missing == 4

        predictor.reset()
        assert predictor.consecutive_missing == 0


class TestOrientationExtrapolation:
    """Test orientation angle extrapolation."""

    def test_constant_orientation(self):
        """Constant orientation is preserved."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=3)

        previous = [
            _make_detection(frame_index=0, orientation_angle=90.0),
            _make_detection(frame_index=1, orientation_angle=90.0),
            _make_detection(frame_index=2, orientation_angle=90.0),
        ]
        result = predictor.predict_position(previous, current_frame_index=3)

        assert result is not None
        assert abs(result.orientation_angle - 90.0) < 0.01

    def test_increasing_orientation(self):
        """Linearly increasing orientation is extrapolated."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=3)

        previous = [
            _make_detection(frame_index=0, orientation_angle=10.0),
            _make_detection(frame_index=1, orientation_angle=20.0),
            _make_detection(frame_index=2, orientation_angle=30.0),
        ]
        result = predictor.predict_position(previous, current_frame_index=3)

        assert result is not None
        assert abs(result.orientation_angle - 40.0) < 0.01

    def test_wraparound_orientation(self):
        """Orientation wraps around 360 degrees correctly."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=3)

        previous = [
            _make_detection(frame_index=0, orientation_angle=350.0),
            _make_detection(frame_index=1, orientation_angle=355.0),
            _make_detection(frame_index=2, orientation_angle=0.0),  # wrapped from 360
        ]
        result = predictor.predict_position(previous, current_frame_index=3)

        assert result is not None
        # Average angular velocity: 5 deg/frame, predicted: 0 + 5 = 5
        assert abs(result.orientation_angle - 5.0) < 0.01


class TestEdgeCases:
    """Test edge cases for trajectory prediction."""

    def test_empty_previous_detections(self):
        """Returns None when no previous detections available."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=5)

        result = predictor.predict_position([], current_frame_index=5)
        assert result is None

    def test_all_previous_undetected(self):
        """Returns None when all previous detections have detected=False."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=5)

        previous = [
            _make_detection(frame_index=3, detected=False),
            _make_detection(frame_index=4, detected=False),
        ]
        result = predictor.predict_position(previous, current_frame_index=5)
        assert result is None

    def test_mixed_detected_and_undetected(self):
        """Uses only detected=True entries from previous detections."""
        predictor = TrajectoryPredictor()
        predictor.notify_missing(frame_index=5)

        previous = [
            _make_detection(frame_index=1, position=(100.0, 200.0), detected=True),
            _make_detection(frame_index=2, detected=False),
            _make_detection(frame_index=3, detected=False),
            _make_detection(frame_index=4, position=(120.0, 220.0), detected=True),
        ]
        result = predictor.predict_position(previous, current_frame_index=5)

        assert result is not None
        # Only uses frames 1 and 4: velocity = (20, 20) per gap (1 gap)
        # Predicted: (120+20, 220+20) = (140, 240)
        assert abs(result.position[0] - 140.0) < 0.01
        assert abs(result.position[1] - 240.0) < 0.01
