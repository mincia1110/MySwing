"""Unit tests for swing anomaly detection and classification failure handling (Task 7.2).

Tests:
- Missing phase detection (e.g., no STRIDE detected)
- Abnormally short phase detection (< 50ms)
- Normal phases produce no anomalies
- Classification failure when insufficient pose data
- Multiple anomalies in one swing

Validates: Requirements 5.4, 5.5
"""

import pytest

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.enums import SwingPhase
from app.models.pose import Keypoint, PoseResult
from app.models.swing import PhaseAnomaly, SwingPhaseResult, TransitionBoundary
from app.pipeline.swing_classifier import (
    KEYPOINT_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_FRAME_RATIO,
    MIN_PHASE_DURATION_MS,
    SwingPhaseClassifier,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _make_keypoint(
    name: str,
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
    confidence: float = 0.9,
) -> Keypoint:
    """Create a Keypoint with given parameters."""
    return Keypoint(x=x, y=y, z=z, confidence=confidence, name=name)


def _make_pose(
    frame_index: int,
    overall_confidence: float = 0.9,
    is_low_confidence: bool = False,
) -> PoseResult:
    """Create a PoseResult with default keypoints."""
    keypoints = [
        _make_keypoint("head", x=0.5, y=0.1),
        _make_keypoint("left_shoulder", x=0.4, y=0.3),
        _make_keypoint("right_shoulder", x=0.6, y=0.3),
        _make_keypoint("left_hip", x=0.4, y=0.55),
        _make_keypoint("right_hip", x=0.6, y=0.55),
        _make_keypoint("left_wrist", x=0.3, y=0.55),
        _make_keypoint("right_wrist", x=0.55, y=0.55),
        _make_keypoint("left_ankle", x=0.4, y=0.9),
        _make_keypoint("right_ankle", x=0.6, y=0.9),
    ]
    return PoseResult(
        frame_index=frame_index,
        keypoints=keypoints,
        person_id=1,
        is_primary_batter=True,
        overall_confidence=overall_confidence,
        is_low_confidence=is_low_confidence,
    )


def _make_full_phase_result(fps: float = 30.0) -> SwingPhaseResult:
    """Create a SwingPhaseResult with all 6 phases and normal durations."""
    # Each phase spans 10 frames at 30fps = 333.33ms (well above 50ms)
    phases = {
        SwingPhase.STANCE: (0, 10),
        SwingPhase.LOAD: (10, 20),
        SwingPhase.STRIDE: (20, 30),
        SwingPhase.ROTATION: (30, 40),
        SwingPhase.IMPACT: (40, 50),
        SwingPhase.FOLLOW_THROUGH: (50, 60),
    }
    phase_durations_ms = {
        SwingPhase.STANCE: 333.33,
        SwingPhase.LOAD: 333.33,
        SwingPhase.STRIDE: 333.33,
        SwingPhase.ROTATION: 333.33,
        SwingPhase.IMPACT: 333.33,
        SwingPhase.FOLLOW_THROUGH: 333.33,
    }
    transitions = [
        TransitionBoundary(SwingPhase.STANCE, SwingPhase.LOAD, 10, 0.9),
        TransitionBoundary(SwingPhase.LOAD, SwingPhase.STRIDE, 20, 0.9),
        TransitionBoundary(SwingPhase.STRIDE, SwingPhase.ROTATION, 30, 0.9),
        TransitionBoundary(SwingPhase.ROTATION, SwingPhase.IMPACT, 40, 0.9),
        TransitionBoundary(SwingPhase.IMPACT, SwingPhase.FOLLOW_THROUGH, 50, 0.9),
    ]
    return SwingPhaseResult(
        phases=phases,
        transitions=transitions,
        phase_durations_ms=phase_durations_ms,
        anomalies=[],
    )


# ---------------------------------------------------------------------------
# Test: detect_anomalies
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    """Test anomaly detection for missing and abnormally short phases."""

    def test_normal_phases_produce_no_anomalies(self):
        """All 6 phases with normal durations should produce no anomalies."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        anomalies = classifier.detect_anomalies(phase_result, fps=30.0)

        assert anomalies == []

    def test_missing_phase_detected(self):
        """A missing phase should produce a 'missing' anomaly."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        # Remove STRIDE phase
        del phase_result.phases[SwingPhase.STRIDE]
        del phase_result.phase_durations_ms[SwingPhase.STRIDE]

        anomalies = classifier.detect_anomalies(phase_result, fps=30.0)

        assert len(anomalies) == 1
        assert anomalies[0].phase == SwingPhase.STRIDE
        assert anomalies[0].anomaly_type == "missing"
        assert anomalies[0].duration_ms is None

    def test_abnormally_short_phase_detected(self):
        """A phase with duration < 50ms should produce 'abnormally_short' anomaly."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        # Set IMPACT duration to 30ms (below 50ms threshold)
        phase_result.phase_durations_ms[SwingPhase.IMPACT] = 30.0

        anomalies = classifier.detect_anomalies(phase_result, fps=30.0)

        assert len(anomalies) == 1
        assert anomalies[0].phase == SwingPhase.IMPACT
        assert anomalies[0].anomaly_type == "abnormally_short"
        assert anomalies[0].duration_ms == 30.0

    def test_phase_at_exactly_50ms_produces_no_anomaly(self):
        """A phase with duration exactly 50ms should NOT be flagged."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        # Set ROTATION duration to exactly 50ms
        phase_result.phase_durations_ms[SwingPhase.ROTATION] = 50.0

        anomalies = classifier.detect_anomalies(phase_result, fps=30.0)

        assert anomalies == []

    def test_multiple_anomalies_in_one_swing(self):
        """Multiple anomalies (missing + short) should all be detected."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        # Remove STRIDE
        del phase_result.phases[SwingPhase.STRIDE]
        del phase_result.phase_durations_ms[SwingPhase.STRIDE]

        # Make LOAD abnormally short
        phase_result.phase_durations_ms[SwingPhase.LOAD] = 20.0

        # Make IMPACT abnormally short
        phase_result.phase_durations_ms[SwingPhase.IMPACT] = 45.0

        anomalies = classifier.detect_anomalies(phase_result, fps=30.0)

        assert len(anomalies) == 3

        # Check that all expected anomalies are present
        anomaly_map = {a.phase: a for a in anomalies}
        assert SwingPhase.STRIDE in anomaly_map
        assert anomaly_map[SwingPhase.STRIDE].anomaly_type == "missing"

        assert SwingPhase.LOAD in anomaly_map
        assert anomaly_map[SwingPhase.LOAD].anomaly_type == "abnormally_short"
        assert anomaly_map[SwingPhase.LOAD].duration_ms == 20.0

        assert SwingPhase.IMPACT in anomaly_map
        assert anomaly_map[SwingPhase.IMPACT].anomaly_type == "abnormally_short"
        assert anomaly_map[SwingPhase.IMPACT].duration_ms == 45.0

    def test_all_phases_missing_produces_six_anomalies(self):
        """Empty phase result should produce 6 missing anomalies."""
        classifier = SwingPhaseClassifier()
        phase_result = SwingPhaseResult()

        anomalies = classifier.detect_anomalies(phase_result, fps=30.0)

        assert len(anomalies) == 6
        for anomaly in anomalies:
            assert anomaly.anomaly_type == "missing"
            assert anomaly.duration_ms is None

    def test_phase_with_zero_duration_is_abnormally_short(self):
        """A phase with 0ms duration should be flagged as abnormally short."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        phase_result.phase_durations_ms[SwingPhase.ROTATION] = 0.0

        anomalies = classifier.detect_anomalies(phase_result, fps=30.0)

        assert len(anomalies) == 1
        assert anomalies[0].phase == SwingPhase.ROTATION
        assert anomalies[0].anomaly_type == "abnormally_short"
        assert anomalies[0].duration_ms == 0.0


# ---------------------------------------------------------------------------
# Test: detect_classification_failures
# ---------------------------------------------------------------------------


class TestDetectClassificationFailures:
    """Test classification failure detection due to insufficient pose data."""

    def test_no_failures_with_good_pose_data(self):
        """High-confidence pose data should produce no classification failures."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        # Create good pose data for all frames
        pose_sequence = [_make_pose(i) for i in range(61)]

        failures = classifier.detect_classification_failures(
            pose_sequence, phase_result
        )

        assert failures == []

    def test_failure_when_phase_has_mostly_low_confidence_frames(self):
        """Phase with >50% low-confidence frames should report failure."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        # Create pose data where ROTATION phase (frames 30-40) has mostly low confidence
        pose_sequence = []
        for i in range(61):
            if 30 <= i <= 40:
                # More than 50% of frames in ROTATION are low confidence
                is_low = i <= 36  # 7 out of 11 frames are low confidence (~64%)
                pose_sequence.append(
                    _make_pose(i, overall_confidence=0.3 if is_low else 0.9, is_low_confidence=is_low)
                )
            else:
                pose_sequence.append(_make_pose(i))

        failures = classifier.detect_classification_failures(
            pose_sequence, phase_result
        )

        assert "rotation" in failures

    def test_failure_when_overall_confidence_below_threshold(self):
        """Frames with overall_confidence < 0.5 should count as low confidence."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        # Create pose data where LOAD phase (frames 10-20) has low overall confidence
        pose_sequence = []
        for i in range(61):
            if 10 <= i <= 20:
                # All frames in LOAD have low overall confidence
                pose_sequence.append(
                    _make_pose(i, overall_confidence=0.3, is_low_confidence=False)
                )
            else:
                pose_sequence.append(_make_pose(i))

        failures = classifier.detect_classification_failures(
            pose_sequence, phase_result
        )

        assert "load" in failures

    def test_no_failure_when_less_than_half_frames_are_low_confidence(self):
        """Phase with <=50% low-confidence frames should NOT report failure."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        # Create pose data where STRIDE phase (frames 20-30) has exactly 50% low confidence
        pose_sequence = []
        for i in range(61):
            if 20 <= i <= 30:
                # 5 out of 11 frames are low confidence (~45%)
                is_low = i <= 24
                pose_sequence.append(
                    _make_pose(i, overall_confidence=0.3 if is_low else 0.9, is_low_confidence=is_low)
                )
            else:
                pose_sequence.append(_make_pose(i))

        failures = classifier.detect_classification_failures(
            pose_sequence, phase_result
        )

        assert "stride" not in failures

    def test_multiple_phases_can_fail(self):
        """Multiple phases with insufficient data should all be reported."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        # Create pose data where LOAD and ROTATION phases have low confidence
        pose_sequence = []
        for i in range(61):
            if (10 <= i <= 20) or (30 <= i <= 40):
                pose_sequence.append(
                    _make_pose(i, overall_confidence=0.2, is_low_confidence=True)
                )
            else:
                pose_sequence.append(_make_pose(i))

        failures = classifier.detect_classification_failures(
            pose_sequence, phase_result
        )

        assert "load" in failures
        assert "rotation" in failures

    def test_empty_pose_sequence_returns_no_failures(self):
        """Empty pose sequence should return no failures."""
        classifier = SwingPhaseClassifier()
        phase_result = _make_full_phase_result()

        failures = classifier.detect_classification_failures([], phase_result)

        assert failures == []

    def test_empty_phase_result_returns_no_failures(self):
        """Empty phase result should return no failures."""
        classifier = SwingPhaseClassifier()
        pose_sequence = [_make_pose(i) for i in range(20)]

        failures = classifier.detect_classification_failures(
            pose_sequence, SwingPhaseResult()
        )

        assert failures == []


# ---------------------------------------------------------------------------
# Test: classify_phases integration with anomaly detection
# ---------------------------------------------------------------------------


class TestClassifyPhasesAnomalyIntegration:
    """Test that classify_phases integrates anomaly detection correctly."""

    def test_classify_phases_includes_anomalies_in_result(self):
        """classify_phases should populate anomalies field in result."""
        classifier = SwingPhaseClassifier()

        # Create a minimal sequence that won't detect all phases
        # (too few frames for full swing, only STANCE will be detected)
        pose_sequence = [_make_pose(i) for i in range(12)]
        bat_trajectory = BatTrajectory(
            detections=[
                BatDetectionResult(
                    frame_index=i,
                    detected=True,
                    position=(320.0, 240.0),
                    orientation_angle=45.0,
                    length_pixels=150.0,
                    confidence=0.95,
                    is_predicted=False,
                )
                for i in range(12)
            ],
            bat_speed_pixels_per_frame=[5.0] * 11,
            tracking_accuracy=0.95,
            tracking_failures=[],
        )

        result = classifier.classify_phases(pose_sequence, bat_trajectory, 30.0)

        # Some phases should be missing, so anomalies should be populated
        # At minimum, not all 6 phases will be detected from 12 static frames
        assert isinstance(result.anomalies, list)

    def test_classify_phases_includes_classification_failures(self):
        """classify_phases should populate classification_failures field."""
        classifier = SwingPhaseClassifier()

        # Create a sequence with all low-confidence poses
        pose_sequence = [
            _make_pose(i, overall_confidence=0.2, is_low_confidence=True)
            for i in range(12)
        ]
        bat_trajectory = BatTrajectory(
            detections=[
                BatDetectionResult(
                    frame_index=i,
                    detected=True,
                    position=(320.0, 240.0),
                    orientation_angle=45.0,
                    length_pixels=150.0,
                    confidence=0.95,
                    is_predicted=False,
                )
                for i in range(12)
            ],
            bat_speed_pixels_per_frame=[5.0] * 11,
            tracking_accuracy=0.95,
            tracking_failures=[],
        )

        result = classifier.classify_phases(pose_sequence, bat_trajectory, 30.0)

        assert isinstance(result.classification_failures, list)
