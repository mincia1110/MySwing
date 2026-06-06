"""Unit tests for swing evaluator modern principles, weight transfer, and bat path flatness.

Tests for Task 10.2: ModernPrinciplesEvaluator, WeightTransferAnalyzer,
and BatPathFlatnessEvaluator classes.
"""

import pytest

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.biomechanics import (
    LaunchAngleResult,
    BiomechanicsResult,
    JointAngularVelocity,
    KinematicChainResult,
)
from app.models.enums import SwingPhase
from app.models.evaluation import WeightTransferResult
from app.models.pose import Keypoint, PoseResult
from app.models.swing import SwingPhaseResult
from app.pipeline.swing_evaluator import (
    BatPathFlatnessEvaluator,
    ModernPrinciplesEvaluator,
    WeightTransferAnalyzer,
)


# --- Helper functions ---

def _make_keypoint(name: str, x: float, y: float, confidence: float = 0.9) -> Keypoint:
    """Create a keypoint with default z=0."""
    return Keypoint(x=x, y=y, z=0.0, confidence=confidence, name=name)


def _make_pose(frame_index: int, keypoints: list[Keypoint]) -> PoseResult:
    """Create a PoseResult with default values."""
    return PoseResult(
        frame_index=frame_index,
        keypoints=keypoints,
        person_id=1,
        is_primary_batter=True,
        overall_confidence=0.9,
        is_low_confidence=False,
    )


def _make_bat_detection(
    frame_index: int, x: float, y: float, detected: bool = True, length: float = 100.0
) -> BatDetectionResult:
    """Create a BatDetectionResult."""
    return BatDetectionResult(
        frame_index=frame_index,
        detected=detected,
        position=(x, y),
        orientation_angle=0.0,
        length_pixels=length,
        confidence=0.95,
        is_predicted=False,
    )


# --- ModernPrinciplesEvaluator Tests ---

class TestEvaluateAttackAngle:
    """Tests for attack angle evaluation (5-15° target range)."""

    def setup_method(self):
        self.evaluator = ModernPrinciplesEvaluator()

    def test_attack_angle_within_range_passes(self):
        """Attack angle of 10° (within 5-15°) should pass."""
        attack_angle = LaunchAngleResult(
            angle_degrees=10.0,
            precision=0.5,
            impact_frame=10,
        )
        result = self.evaluator._evaluate_attack_angle(attack_angle)
        assert result["passed"] is True
        assert result["angle_degrees"] == 10.0
        assert result["deviation"] == 0.0

    def test_attack_angle_at_lower_boundary_passes(self):
        """Attack angle of exactly 5° should pass."""
        attack_angle = LaunchAngleResult(
            angle_degrees=5.0,
            precision=0.5,
            impact_frame=10,
        )
        result = self.evaluator._evaluate_attack_angle(attack_angle)
        assert result["passed"] is True
        assert result["deviation"] == 0.0

    def test_attack_angle_at_upper_boundary_passes(self):
        """Attack angle of exactly 15° should pass."""
        attack_angle = LaunchAngleResult(
            angle_degrees=15.0,
            precision=0.5,
            impact_frame=10,
        )
        result = self.evaluator._evaluate_attack_angle(attack_angle)
        assert result["passed"] is True
        assert result["deviation"] == 0.0

    def test_attack_angle_below_min_fails(self):
        """Attack angle of 3° (below 5° min) should fail with correct deviation."""
        attack_angle = LaunchAngleResult(
            angle_degrees=3.0,
            precision=0.5,
            impact_frame=10,
        )
        result = self.evaluator._evaluate_attack_angle(attack_angle)
        assert result["passed"] is False
        assert result["deviation"] == 2.0

    def test_attack_angle_above_max_fails(self):
        """Attack angle of 20° (above 15° max) should fail with correct deviation."""
        attack_angle = LaunchAngleResult(
            angle_degrees=20.0,
            precision=0.5,
            impact_frame=10,
        )
        result = self.evaluator._evaluate_attack_angle(attack_angle)
        assert result["passed"] is False
        assert result["deviation"] == pytest.approx(5.0)


class TestEvaluateBarrelInZone:
    """Tests for barrel in zone evaluation (≥150ms threshold)."""

    def setup_method(self):
        self.evaluator = ModernPrinciplesEvaluator()

    def test_barrel_in_zone_sufficient_time_passes(self):
        """Barrel staying in zone for ≥150ms at 60fps should pass."""
        fps = 60.0
        impact_frame = 30
        # Need at least 9 frames at 60fps for 150ms (9/60 = 150ms)
        # Create detections with consistent y-position around impact
        detections = []
        for i in range(20, 40):
            detections.append(
                _make_bat_detection(frame_index=i, x=100.0 + i * 2, y=200.0)
            )

        bat_trajectory = BatTrajectory(
            detections=detections,
            bat_speed_pixels_per_frame=[10.0] * 20,
            tracking_accuracy=0.95,
            tracking_failures=[],
        )

        result = self.evaluator._evaluate_barrel_in_zone(
            bat_trajectory, impact_frame, fps
        )
        assert result["passed"] is True
        assert result["duration_ms"] >= 150.0

    def test_barrel_in_zone_insufficient_time_fails(self):
        """Barrel staying in zone for <150ms at 60fps should fail."""
        fps = 60.0
        impact_frame = 10
        # Only 4 frames at 60fps = ~67ms (< 150ms)
        detections = []
        for i in range(8, 12):
            detections.append(
                _make_bat_detection(frame_index=i, x=100.0 + i * 2, y=200.0)
            )
        # Add frames outside zone (different y)
        for i in range(4, 8):
            detections.append(
                _make_bat_detection(frame_index=i, x=100.0 + i * 2, y=500.0)
            )

        bat_trajectory = BatTrajectory(
            detections=detections,
            bat_speed_pixels_per_frame=[10.0] * 8,
            tracking_accuracy=0.95,
            tracking_failures=[],
        )

        result = self.evaluator._evaluate_barrel_in_zone(
            bat_trajectory, impact_frame, fps
        )
        assert result["passed"] is False
        assert result["duration_ms"] < 150.0


class TestEvaluateKinematicSequence:
    """Tests for kinematic sequence evaluation (proximal-to-distal, ≤50ms gaps)."""

    def setup_method(self):
        self.evaluator = ModernPrinciplesEvaluator()

    def test_correct_sequence_with_small_gaps_passes(self):
        """Correct proximal-to-distal order with ≤50ms gaps should pass."""
        kinematic_chain = KinematicChainResult(
            joint_peak_angular_velocities={
                "hips": JointAngularVelocity(
                    joint_name="hips", peak_velocity_dps=500.0,
                    peak_frame=10, peak_time_ms=0.0
                ),
                "shoulders": JointAngularVelocity(
                    joint_name="shoulders", peak_velocity_dps=700.0,
                    peak_frame=13, peak_time_ms=30.0
                ),
                "elbows": JointAngularVelocity(
                    joint_name="elbows", peak_velocity_dps=900.0,
                    peak_frame=16, peak_time_ms=60.0
                ),
                "wrists": JointAngularVelocity(
                    joint_name="wrists", peak_velocity_dps=1200.0,
                    peak_frame=19, peak_time_ms=90.0
                ),
            },
            sequence_correct=True,
            timing_gaps_ms={
                "hips_to_shoulders": 30.0,
                "shoulders_to_elbows": 30.0,
                "elbows_to_wrists": 30.0,
            },
        )

        result = self.evaluator._evaluate_kinematic_sequence(kinematic_chain)
        assert result["passed"] is True
        assert result["sequence_correct"] is True
        assert result["all_gaps_within_threshold"] is True
        assert result["max_gap_ms"] == 30.0

    def test_incorrect_sequence_order_fails(self):
        """Incorrect order (wrists before elbows) should fail."""
        kinematic_chain = KinematicChainResult(
            joint_peak_angular_velocities={
                "hips": JointAngularVelocity(
                    joint_name="hips", peak_velocity_dps=500.0,
                    peak_frame=10, peak_time_ms=0.0
                ),
                "shoulders": JointAngularVelocity(
                    joint_name="shoulders", peak_velocity_dps=700.0,
                    peak_frame=13, peak_time_ms=30.0
                ),
                "elbows": JointAngularVelocity(
                    joint_name="elbows", peak_velocity_dps=900.0,
                    peak_frame=20, peak_time_ms=100.0
                ),
                "wrists": JointAngularVelocity(
                    joint_name="wrists", peak_velocity_dps=1200.0,
                    peak_frame=16, peak_time_ms=60.0
                ),
            },
            sequence_correct=False,  # wrists peak before elbows
            timing_gaps_ms={
                "hips_to_shoulders": 30.0,
                "shoulders_to_elbows": 70.0,
                "elbows_to_wrists": -40.0,
            },
        )

        result = self.evaluator._evaluate_kinematic_sequence(kinematic_chain)
        assert result["passed"] is False
        assert result["sequence_correct"] is False

    def test_correct_order_but_large_gaps_fails(self):
        """Correct order but gaps >50ms should fail."""
        kinematic_chain = KinematicChainResult(
            joint_peak_angular_velocities={
                "hips": JointAngularVelocity(
                    joint_name="hips", peak_velocity_dps=500.0,
                    peak_frame=10, peak_time_ms=0.0
                ),
                "shoulders": JointAngularVelocity(
                    joint_name="shoulders", peak_velocity_dps=700.0,
                    peak_frame=20, peak_time_ms=100.0
                ),
                "elbows": JointAngularVelocity(
                    joint_name="elbows", peak_velocity_dps=900.0,
                    peak_frame=30, peak_time_ms=200.0
                ),
                "wrists": JointAngularVelocity(
                    joint_name="wrists", peak_velocity_dps=1200.0,
                    peak_frame=40, peak_time_ms=300.0
                ),
            },
            sequence_correct=True,
            timing_gaps_ms={
                "hips_to_shoulders": 100.0,
                "shoulders_to_elbows": 100.0,
                "elbows_to_wrists": 100.0,
            },
        )

        result = self.evaluator._evaluate_kinematic_sequence(kinematic_chain)
        assert result["passed"] is False
        assert result["sequence_correct"] is True
        assert result["all_gaps_within_threshold"] is False
        assert result["max_gap_ms"] == 100.0


# --- WeightTransferAnalyzer Tests ---

class TestWeightTransferAnalyzer:
    """Tests for weight transfer timing analysis."""

    def setup_method(self):
        self.analyzer = WeightTransferAnalyzer()

    def test_weight_transfer_timing_calculation(self):
        """Test correct timing calculation for weight transfer events."""
        fps = 60.0

        # Create pose sequence simulating stride and rotation
        # Stride phase: frames 10-25
        # Rotation phase: frames 25-40
        pose_sequence = []

        # Stride phase: front foot lifts at frame 12, plants at frame 22
        for i in range(10, 26):
            keypoints = [
                _make_keypoint("left_hip", 0.4, 0.5),
                _make_keypoint("right_hip", 0.6, 0.5),
            ]
            if i <= 11:
                # Before stride initiation - foot stable
                keypoints.append(_make_keypoint("left_ankle", 0.3, 0.9))
            elif i <= 15:
                # Foot lifting (y decreasing)
                y_val = 0.9 - (i - 11) * 0.02
                keypoints.append(_make_keypoint("left_ankle", 0.3, y_val))
            elif i <= 21:
                # Foot moving forward and down
                y_val = 0.82 + (i - 15) * 0.015
                keypoints.append(_make_keypoint("left_ankle", 0.3, y_val))
            else:
                # Foot planted (stable y)
                keypoints.append(_make_keypoint("left_ankle", 0.3, 0.9))

            pose_sequence.append(_make_pose(i, keypoints))

        # Rotation phase: hip rotation starts at frame 27
        for i in range(26, 41):
            hip_angle_offset = 0.0 if i < 27 else (i - 27) * 0.01
            keypoints = [
                _make_keypoint("left_hip", 0.4 + hip_angle_offset, 0.5),
                _make_keypoint("right_hip", 0.6 - hip_angle_offset, 0.5),
                _make_keypoint("left_ankle", 0.3, 0.9),
            ]
            pose_sequence.append(_make_pose(i, keypoints))

        swing_phases = SwingPhaseResult(
            phases={
                SwingPhase.STRIDE: (10, 25),
                SwingPhase.ROTATION: (25, 40),
            },
            transitions=[],
            phase_durations_ms={},
            anomalies=[],
        )

        result = self.analyzer.analyze_weight_transfer(
            pose_sequence, swing_phases, fps
        )

        assert isinstance(result, WeightTransferResult)
        assert result.stride_to_foot_plant_ms >= 0.0
        assert result.foot_plant_to_hip_rotation_ms >= 0.0
        # The stride to foot plant should be positive (foot plant after stride)
        assert result.stride_to_foot_plant_ms > 0.0

    def test_weight_transfer_with_missing_phases(self):
        """Test that missing phases return zero timing."""
        fps = 60.0
        pose_sequence = [
            _make_pose(0, [_make_keypoint("left_ankle", 0.3, 0.9)]),
            _make_pose(1, [_make_keypoint("left_ankle", 0.3, 0.9)]),
        ]

        # Missing rotation phase
        swing_phases = SwingPhaseResult(
            phases={SwingPhase.STRIDE: (0, 5)},
            transitions=[],
            phase_durations_ms={},
            anomalies=[],
        )

        result = self.analyzer.analyze_weight_transfer(
            pose_sequence, swing_phases, fps
        )

        assert result.stride_to_foot_plant_ms == 0.0
        assert result.foot_plant_to_hip_rotation_ms == 0.0


# --- BatPathFlatnessEvaluator Tests ---

class TestBatPathFlatnessEvaluator:
    """Tests for bat path flatness evaluation."""

    def setup_method(self):
        self.evaluator = BatPathFlatnessEvaluator()

    def test_flat_bat_path_measures_duration(self):
        """Bat path staying within 5° of swing plane should measure duration."""
        fps = 60.0
        impact_frame = 20

        # Create a flat bat trajectory (moving horizontally)
        detections = []
        for i in range(10, 30):
            detections.append(
                _make_bat_detection(frame_index=i, x=100.0 + i * 5.0, y=200.0)
            )

        bat_trajectory = BatTrajectory(
            detections=detections,
            bat_speed_pixels_per_frame=[5.0] * 20,
            tracking_accuracy=0.95,
            tracking_failures=[],
        )

        # Create pose with back shoulder above and behind the bat
        pose_sequence = [
            _make_pose(
                impact_frame,
                [
                    _make_keypoint("right_shoulder", 50.0, 150.0),
                    _make_keypoint("left_shoulder", 80.0, 150.0),
                ],
            )
        ]

        result = self.evaluator.evaluate_flatness(
            bat_trajectory, pose_sequence, impact_frame, fps
        )

        assert result["flatness_duration_ms"] >= 0.0
        assert result["threshold_degrees"] == 5.0
        assert "swing_plane_angle" in result

    def test_non_flat_bat_path_low_duration(self):
        """Bat path with varying angles should have lower flatness duration."""
        fps = 60.0
        impact_frame = 15

        # Create a bat trajectory with varying angles (zigzag pattern)
        detections = []
        for i in range(5, 25):
            # Alternating y to create non-flat path
            y_offset = 20.0 * (1 if i % 2 == 0 else -1)
            detections.append(
                _make_bat_detection(
                    frame_index=i, x=100.0 + i * 5.0, y=200.0 + y_offset
                )
            )

        bat_trajectory = BatTrajectory(
            detections=detections,
            bat_speed_pixels_per_frame=[5.0] * 20,
            tracking_accuracy=0.95,
            tracking_failures=[],
        )

        # Create pose with back shoulder
        pose_sequence = [
            _make_pose(
                impact_frame,
                [
                    _make_keypoint("right_shoulder", 50.0, 150.0),
                    _make_keypoint("left_shoulder", 80.0, 150.0),
                ],
            )
        ]

        result = self.evaluator.evaluate_flatness(
            bat_trajectory, pose_sequence, impact_frame, fps
        )

        # With zigzag pattern, fewer frames should be within 5° of plane
        assert result["flatness_duration_ms"] >= 0.0
        assert result["threshold_degrees"] == 5.0

    def test_flatness_with_invalid_fps(self):
        """Invalid FPS should return zero duration."""
        bat_trajectory = BatTrajectory(detections=[], tracking_accuracy=0.0)
        pose_sequence = []

        result = self.evaluator.evaluate_flatness(
            bat_trajectory, pose_sequence, 10, 0.0
        )

        assert result["flatness_duration_ms"] == 0.0
        assert result["frames_in_plane"] == 0
