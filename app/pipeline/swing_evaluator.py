"""Swing evaluation module (Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6).

Provides reference comparison, modern hitting principles evaluation, weight transfer
timing analysis, bat path flatness evaluation, and improvement area ranking for
baseball swing assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.biomechanics import (
    BiomechanicsResult,
    KinematicChainResult,
    LaunchAngleResult,
)
from app.models.enums import MetricRating, SwingPhase
from app.models.evaluation import ImprovementArea, MetricEvaluation, WeightTransferResult
from app.models.pose import Keypoint, PoseResult
from app.models.swing import SwingPhaseResult

# Modern hitting principles constants
ATTACK_ANGLE_MIN_DEGREES = 5.0
ATTACK_ANGLE_MAX_DEGREES = 15.0
BARREL_IN_ZONE_MIN_MS = 150.0
KINEMATIC_SEQUENCE_MAX_GAP_MS = 50.0

# Default level/age_group constants (Requirement 7.7)
DEFAULT_LEVEL = "recreational"
DEFAULT_AGE_GROUP = "adult"


@dataclass
class ReferenceRange:
    """Reference range for a metric at a specific level/age_group."""

    metric_name: str
    ref_min: float  # acceptable min
    ref_max: float  # acceptable max
    optimal_min: float  # optimal min (green zone)
    optimal_max: float  # optimal max (green zone)
    unit: str


# Reference data organized by level → age_group → list of ReferenceRange
_REFERENCE_DATA: Dict[str, Dict[str, List[ReferenceRange]]] = {
    "recreational": {
        "adult": [
            ReferenceRange("bat_speed", 70.0, 100.0, 80.0, 95.0, "km/h"),
            ReferenceRange("attack_angle", 5.0, 25.0, 10.0, 20.0, "degrees"),
            ReferenceRange("hip_shoulder_separation", 25.0, 50.0, 30.0, 45.0, "degrees"),
            ReferenceRange("hand_path_efficiency", 0.55, 0.85, 0.65, 0.80, "ratio"),
            ReferenceRange("stride_length_cm", 40.0, 120.0, 60.0, 100.0, "cm"),
            ReferenceRange("cog_sway_cm", 0.0, 12.0, 0.0, 6.0, "cm"),
            ReferenceRange("cog_drop_cm", 1.0, 18.0, 2.0, 10.0, "cm"),
            ReferenceRange("head_stability_cm", 0.0, 12.0, 0.0, 6.0, "cm"),
            ReferenceRange("front_knee_flexion_degrees", 115.0, 155.0, 125.0, 145.0, "degrees"),
            ReferenceRange("spine_angle_degrees", -15.0, 20.0, -5.0, 12.0, "degrees"),
        ],
    },
    "professional": {
        "adult": [
            ReferenceRange("bat_speed", 110.0, 130.0, 115.0, 125.0, "km/h"),
            ReferenceRange("attack_angle", 10.0, 30.0, 15.0, 25.0, "degrees"),
            ReferenceRange("hip_shoulder_separation", 35.0, 55.0, 40.0, 50.0, "degrees"),
            ReferenceRange("hand_path_efficiency", 0.75, 0.95, 0.80, 0.90, "ratio"),
            ReferenceRange("stride_length_cm", 60.0, 130.0, 80.0, 110.0, "cm"),
            ReferenceRange("cog_sway_cm", 0.0, 8.0, 0.0, 4.0, "cm"),
            ReferenceRange("cog_drop_cm", 2.0, 15.0, 3.0, 8.0, "cm"),
            ReferenceRange("head_stability_cm", 0.0, 8.0, 0.0, 4.0, "cm"),
            ReferenceRange("front_knee_flexion_degrees", 120.0, 150.0, 130.0, 140.0, "degrees"),
            ReferenceRange("spine_angle_degrees", -10.0, 15.0, 0.0, 10.0, "degrees"),
        ],
    },
    "college": {
        "adult": [
            ReferenceRange("bat_speed", 100.0, 120.0, 105.0, 115.0, "km/h"),
            ReferenceRange("attack_angle", 8.0, 28.0, 12.0, 22.0, "degrees"),
            ReferenceRange("hip_shoulder_separation", 30.0, 52.0, 35.0, 47.0, "degrees"),
            ReferenceRange("hand_path_efficiency", 0.70, 0.90, 0.75, 0.85, "ratio"),
            ReferenceRange("stride_length_cm", 50.0, 125.0, 70.0, 105.0, "cm"),
            ReferenceRange("cog_sway_cm", 0.0, 10.0, 0.0, 5.0, "cm"),
            ReferenceRange("cog_drop_cm", 1.5, 16.0, 2.5, 9.0, "cm"),
            ReferenceRange("head_stability_cm", 0.0, 10.0, 0.0, 5.0, "cm"),
            ReferenceRange("front_knee_flexion_degrees", 118.0, 152.0, 128.0, 142.0, "degrees"),
            ReferenceRange("spine_angle_degrees", -12.0, 18.0, -2.0, 11.0, "degrees"),
        ],
    },
    "high_school": {
        "adult": [
            ReferenceRange("bat_speed", 85.0, 110.0, 90.0, 105.0, "km/h"),
            ReferenceRange("attack_angle", 6.0, 26.0, 10.0, 20.0, "degrees"),
            ReferenceRange("hip_shoulder_separation", 28.0, 50.0, 32.0, 45.0, "degrees"),
            ReferenceRange("hand_path_efficiency", 0.60, 0.88, 0.70, 0.82, "ratio"),
            ReferenceRange("stride_length_cm", 45.0, 115.0, 65.0, 100.0, "cm"),
            ReferenceRange("cog_sway_cm", 0.0, 12.0, 0.0, 6.0, "cm"),
            ReferenceRange("cog_drop_cm", 1.0, 17.0, 2.0, 10.0, "cm"),
            ReferenceRange("head_stability_cm", 0.0, 12.0, 0.0, 6.0, "cm"),
            ReferenceRange("front_knee_flexion_degrees", 115.0, 155.0, 125.0, 145.0, "degrees"),
            ReferenceRange("spine_angle_degrees", -15.0, 20.0, -5.0, 12.0, "degrees"),
        ],
        "youth": [
            ReferenceRange("bat_speed", 70.0, 95.0, 75.0, 90.0, "km/h"),
            ReferenceRange("attack_angle", 5.0, 22.0, 8.0, 18.0, "degrees"),
            ReferenceRange("hip_shoulder_separation", 20.0, 45.0, 25.0, 40.0, "degrees"),
            ReferenceRange("hand_path_efficiency", 0.50, 0.80, 0.60, 0.75, "ratio"),
            ReferenceRange("stride_length_cm", 35.0, 100.0, 50.0, 85.0, "cm"),
            ReferenceRange("cog_sway_cm", 0.0, 14.0, 0.0, 8.0, "cm"),
            ReferenceRange("cog_drop_cm", 0.5, 20.0, 1.5, 12.0, "cm"),
            ReferenceRange("head_stability_cm", 0.0, 14.0, 0.0, 8.0, "cm"),
            ReferenceRange("front_knee_flexion_degrees", 110.0, 160.0, 120.0, 150.0, "degrees"),
            ReferenceRange("spine_angle_degrees", -18.0, 22.0, -8.0, 15.0, "degrees"),
        ],
    },
}

# Mapping from metric_name to how to extract value from BiomechanicsResult
_METRIC_EXTRACTORS: Dict[str, str] = {
    "bat_speed": "bat_speed",
    "attack_angle": "attack_angle",
    "hip_shoulder_separation": "rotation",
    "hand_path_efficiency": "hand_path_efficiency",
    "stride_length_cm": "stride_length_cm",
    "cog_sway_cm": "cog_sway_cm",
    "cog_drop_cm": "cog_drop_cm",
    "head_stability_cm": "head_stability_cm",
    "front_knee_flexion_degrees": "front_knee_flexion_degrees",
    "spine_angle_degrees": "spine_angle_degrees",
}


class ReferenceComparator:
    """Compares biomechanics metrics against reference ranges (Requirement 7.2, 7.3, 7.7).

    Selects reference data based on user's level and age_group.
    Defaults to recreational adult when not specified.
    """

    def compare_with_reference(
        self,
        biomechanics: BiomechanicsResult,
        level: Optional[str] = None,
        age_group: Optional[str] = None,
    ) -> List[MetricEvaluation]:
        """Compare biomechanics metrics against reference ranges.

        Args:
            biomechanics: Complete biomechanics analysis result.
            level: User's skill level (recreational, professional, college, high_school).
            age_group: User's age group (adult, youth).

        Returns:
            List of MetricEvaluation for each available metric.
        """
        # Apply defaults
        effective_level = level if level else DEFAULT_LEVEL
        effective_age_group = age_group if age_group else DEFAULT_AGE_GROUP

        # Get reference ranges for the level/age_group
        ranges = self._get_reference_ranges(effective_level, effective_age_group)

        evaluations: List[MetricEvaluation] = []

        for ref_range in ranges:
            value = self._extract_metric_value(biomechanics, ref_range.metric_name)
            if value is None:
                continue

            deviation, rating = self._calculate_deviation(
                value, ref_range.ref_min, ref_range.ref_max
            )

            color_code = self._determine_color_code(
                rating,
                value,
                ref_range.optimal_min,
                ref_range.optimal_max,
                ref_range.ref_min,
                ref_range.ref_max,
            )

            evaluations.append(
                MetricEvaluation(
                    metric_name=ref_range.metric_name,
                    measured_value=value,
                    unit=ref_range.unit,
                    reference_min=ref_range.ref_min,
                    reference_max=ref_range.ref_max,
                    deviation_percent=deviation,
                    rating=rating,
                    color_code=color_code,
                )
            )

        return evaluations

    def _get_reference_ranges(
        self, level: str, age_group: str
    ) -> List[ReferenceRange]:
        """Get reference ranges for the given level and age_group.

        Falls back to recreational adult if level/age_group not found.
        """
        level_data = _REFERENCE_DATA.get(level)
        if level_data is None:
            level_data = _REFERENCE_DATA[DEFAULT_LEVEL]

        age_data = level_data.get(age_group)
        if age_data is None:
            # Fall back to adult within the same level
            age_data = level_data.get(DEFAULT_AGE_GROUP)
            if age_data is None:
                # Fall back to recreational adult
                age_data = _REFERENCE_DATA[DEFAULT_LEVEL][DEFAULT_AGE_GROUP]

        return age_data

    def _extract_metric_value(
        self, biomechanics: BiomechanicsResult, metric_name: str
    ) -> Optional[float]:
        """Extract a metric value from BiomechanicsResult.

        Args:
            biomechanics: The biomechanics result.
            metric_name: Name of the metric to extract.

        Returns:
            The metric value, or None if not available.
        """
        if metric_name == "bat_speed":
            if biomechanics.bat_speed is not None:
                return biomechanics.bat_speed.speed_kmh
        elif metric_name == "attack_angle":
            if biomechanics.attack_angle is not None:
                return biomechanics.attack_angle.angle_degrees
        elif metric_name == "hip_shoulder_separation":
            if biomechanics.rotation is not None:
                return biomechanics.rotation.hip_shoulder_separation_degrees
        elif metric_name == "hand_path_efficiency":
            if biomechanics.hand_path_efficiency is not None:
                return biomechanics.hand_path_efficiency
        # Swing quality metrics
        elif metric_name == "stride_length_cm":
            return biomechanics.stride_length_cm
        elif metric_name == "cog_sway_cm":
            return biomechanics.cog_sway_cm
        elif metric_name == "cog_drop_cm":
            return biomechanics.cog_drop_cm
        elif metric_name == "head_stability_cm":
            return biomechanics.head_stability_cm
        elif metric_name == "front_knee_flexion_degrees":
            return biomechanics.front_knee_flexion_degrees
        elif metric_name == "spine_angle_degrees":
            return biomechanics.spine_angle_degrees
        return None

    def _calculate_deviation(
        self, value: float, ref_min: float, ref_max: float
    ) -> Tuple[float, MetricRating]:
        """Calculate deviation percentage from reference range.

        Args:
            value: Measured value.
            ref_min: Reference range minimum.
            ref_max: Reference range maximum.

        Returns:
            Tuple of (deviation_percent, MetricRating).
            - Below range: deviation = (ref_min - value) / ref_min * 100
            - Above range: deviation = (value - ref_max) / ref_max * 100
            - Within range: deviation = 0
        """
        if value < ref_min:
            deviation = (ref_min - value) / ref_min * 100.0
            return deviation, MetricRating.BELOW_RANGE
        elif value > ref_max:
            deviation = (value - ref_max) / ref_max * 100.0
            return deviation, MetricRating.ABOVE_RANGE
        else:
            return 0.0, MetricRating.WITHIN_RANGE

    def _determine_color_code(
        self,
        rating: MetricRating,
        value: float,
        optimal_min: float,
        optimal_max: float,
        ref_min: float,
        ref_max: float,
    ) -> str:
        """Determine color code for a metric evaluation.

        Args:
            rating: The MetricRating classification.
            value: Measured value.
            optimal_min: Optimal range minimum (green zone).
            optimal_max: Optimal range maximum (green zone).
            ref_min: Acceptable range minimum.
            ref_max: Acceptable range maximum.

        Returns:
            "green" if within optimal range,
            "yellow" if within acceptable but outside optimal,
            "red" if outside acceptable range.
        """
        if rating == MetricRating.BELOW_RANGE or rating == MetricRating.ABOVE_RANGE:
            return "red"

        # Within range - check if optimal or acceptable
        if optimal_min <= value <= optimal_max:
            return "green"
        else:
            return "yellow"

# Expected proximal-to-distal order
_PROXIMAL_TO_DISTAL_ORDER = ["hips", "shoulders", "elbows", "wrists"]

# Bat path flatness threshold
BAT_PATH_FLATNESS_THRESHOLD_DEGREES = 5.0


class ModernPrinciplesEvaluator:
    """Evaluates swing against modern hitting principles (Requirement 7.1).

    Checks:
    - Attack angle within target range (5-15°)
    - Bat barrel remaining in hitting zone for ≥150ms
    - Kinematic sequence in correct proximal-to-distal order with ≤50ms gaps
    """

    def evaluate_principles(
        self,
        biomechanics: BiomechanicsResult,
        bat_trajectory: BatTrajectory,
        fps: float,
    ) -> Dict[str, Any]:
        """Evaluate swing against modern hitting principles.

        Args:
            biomechanics: Complete biomechanics analysis result.
            bat_trajectory: Bat trajectory data.
            fps: Video frame rate.

        Returns:
            Dictionary with evaluation results for each principle:
            - attack_angle: Attack angle evaluation
            - barrel_in_zone: Barrel zone time evaluation
            - kinematic_sequence: Kinematic sequence evaluation
        """
        results: Dict[str, Any] = {}

        # Evaluate attack angle
        if biomechanics.attack_angle is not None:
            results["attack_angle"] = self._evaluate_attack_angle(
                biomechanics.attack_angle
            )
        else:
            results["attack_angle"] = {
                "passed": False,
                "reason": "Attack angle could not be measured",
            }

        # Evaluate barrel in zone
        if biomechanics.attack_angle is not None:
            impact_frame = biomechanics.attack_angle.impact_frame
            results["barrel_in_zone"] = self._evaluate_barrel_in_zone(
                bat_trajectory, impact_frame, fps
            )
        else:
            results["barrel_in_zone"] = {
                "passed": False,
                "reason": "Impact frame could not be determined",
            }

        # Evaluate kinematic sequence
        if biomechanics.kinematic_chain is not None:
            results["kinematic_sequence"] = self._evaluate_kinematic_sequence(
                biomechanics.kinematic_chain
            )
        else:
            results["kinematic_sequence"] = {
                "passed": False,
                "reason": "Kinematic chain could not be measured",
            }

        return results

    def _evaluate_attack_angle(self, attack_angle: LaunchAngleResult) -> Dict[str, Any]:
        """Evaluate if attack angle is within the target range (5-15°).

        Args:
            attack_angle: Attack angle measurement result.

        Returns:
            Dictionary with:
            - passed: Whether the angle is within range
            - angle_degrees: Measured angle
            - target_min: Minimum target (5°)
            - target_max: Maximum target (15°)
            - deviation: How far outside range (0 if within)
        """
        angle = attack_angle.angle_degrees
        passed = ATTACK_ANGLE_MIN_DEGREES <= angle <= ATTACK_ANGLE_MAX_DEGREES

        if angle < ATTACK_ANGLE_MIN_DEGREES:
            deviation = ATTACK_ANGLE_MIN_DEGREES - angle
        elif angle > ATTACK_ANGLE_MAX_DEGREES:
            deviation = angle - ATTACK_ANGLE_MAX_DEGREES
        else:
            deviation = 0.0

        return {
            "passed": passed,
            "angle_degrees": angle,
            "target_min": ATTACK_ANGLE_MIN_DEGREES,
            "target_max": ATTACK_ANGLE_MAX_DEGREES,
            "deviation": deviation,
        }

    def _evaluate_barrel_in_zone(
        self,
        bat_trajectory: BatTrajectory,
        impact_frame: int,
        fps: float,
    ) -> Dict[str, Any]:
        """Evaluate if bat barrel stays in the hitting zone for ≥150ms.

        The hitting zone is centered around the impact frame. We measure
        how long the barrel stays within a consistent vertical band
        (small angle variation) around the impact point.

        Args:
            bat_trajectory: Bat trajectory data.
            impact_frame: Frame index of impact.
            fps: Video frame rate.

        Returns:
            Dictionary with:
            - passed: Whether barrel stays in zone ≥150ms
            - duration_ms: Actual duration in zone
            - threshold_ms: Required minimum (150ms)
        """
        if fps <= 0:
            return {
                "passed": False,
                "duration_ms": 0.0,
                "threshold_ms": BARREL_IN_ZONE_MIN_MS,
                "reason": "Invalid FPS",
            }

        # Find detections around impact that are in the hitting zone
        # The hitting zone is defined by consistent bat angle (within 5°)
        detections_by_frame = {
            d.frame_index: d
            for d in bat_trajectory.detections
            if d.detected
        }

        if impact_frame not in detections_by_frame:
            return {
                "passed": False,
                "duration_ms": 0.0,
                "threshold_ms": BARREL_IN_ZONE_MIN_MS,
                "reason": "No detection at impact frame",
            }

        impact_detection = detections_by_frame[impact_frame]
        impact_y = impact_detection.position[1]

        # Define the hitting zone as a vertical band around impact position
        # Barrel is "in zone" when its y-position is within a threshold
        # of the impact y-position (representing the pitch plane)
        zone_height_threshold = impact_detection.length_pixels * 0.3

        # Count consecutive frames where barrel is in zone around impact
        in_zone_frames = 0

        # Search backward from impact
        frame = impact_frame
        while frame in detections_by_frame:
            det = detections_by_frame[frame]
            if abs(det.position[1] - impact_y) <= zone_height_threshold:
                in_zone_frames += 1
                frame -= 1
            else:
                break

        # Search forward from impact (excluding impact itself, already counted)
        frame = impact_frame + 1
        while frame in detections_by_frame:
            det = detections_by_frame[frame]
            if abs(det.position[1] - impact_y) <= zone_height_threshold:
                in_zone_frames += 1
                frame += 1
            else:
                break

        duration_ms = (in_zone_frames / fps) * 1000.0
        passed = duration_ms >= BARREL_IN_ZONE_MIN_MS

        return {
            "passed": passed,
            "duration_ms": duration_ms,
            "threshold_ms": BARREL_IN_ZONE_MIN_MS,
        }

    def _evaluate_kinematic_sequence(
        self, kinematic_chain: KinematicChainResult
    ) -> Dict[str, Any]:
        """Evaluate if kinematic sequence follows proximal-to-distal order.

        Checks that:
        1. All joints peak in the correct order (hips→shoulders→elbows→wrists)
        2. Each segment initiates within 50ms of the previous segment

        Args:
            kinematic_chain: Kinematic chain analysis result.

        Returns:
            Dictionary with:
            - passed: Whether sequence is correct with ≤50ms gaps
            - sequence_correct: Whether order is correct
            - all_gaps_within_threshold: Whether all gaps ≤50ms
            - timing_gaps_ms: Individual gap values
            - max_gap_ms: Maximum gap between consecutive joints
        """
        # Check if all joints are present
        required_joints = _PROXIMAL_TO_DISTAL_ORDER
        available_joints = list(
            kinematic_chain.joint_peak_angular_velocities.keys()
        )

        missing_joints = [
            j for j in required_joints if j not in available_joints
        ]
        if missing_joints:
            return {
                "passed": False,
                "sequence_correct": False,
                "all_gaps_within_threshold": False,
                "timing_gaps_ms": {},
                "max_gap_ms": 0.0,
                "reason": f"Missing joints: {missing_joints}",
            }

        # Check sequence order
        sequence_correct = kinematic_chain.sequence_correct

        # Check timing gaps
        timing_gaps = kinematic_chain.timing_gaps_ms
        max_gap_ms = max(timing_gaps.values()) if timing_gaps else 0.0
        all_gaps_within_threshold = all(
            0 <= gap <= KINEMATIC_SEQUENCE_MAX_GAP_MS
            for gap in timing_gaps.values()
        )

        passed = sequence_correct and all_gaps_within_threshold

        return {
            "passed": passed,
            "sequence_correct": sequence_correct,
            "all_gaps_within_threshold": all_gaps_within_threshold,
            "timing_gaps_ms": timing_gaps,
            "max_gap_ms": max_gap_ms,
        }


class WeightTransferAnalyzer:
    """Analyzes weight transfer timing during the swing (Requirement 7.4).

    Measures timing between:
    - Stride initiation → front foot plant
    - Front foot plant → hip rotation onset
    """

    def analyze_weight_transfer(
        self,
        pose_sequence: List[PoseResult],
        swing_phases: SwingPhaseResult,
        fps: float,
    ) -> WeightTransferResult:
        """Analyze weight transfer timing from pose sequence.

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.
            swing_phases: Swing phase classification result with phase boundaries.
            fps: Video frame rate.

        Returns:
            WeightTransferResult with stride_to_foot_plant_ms and
            foot_plant_to_hip_rotation_ms timing values.
        """
        if fps <= 0 or len(pose_sequence) < 2:
            return WeightTransferResult(
                stride_to_foot_plant_ms=0.0,
                foot_plant_to_hip_rotation_ms=0.0,
            )

        # Get stride and rotation phase boundaries
        stride_phase = swing_phases.phases.get(SwingPhase.STRIDE)
        rotation_phase = swing_phases.phases.get(SwingPhase.ROTATION)

        if stride_phase is None or rotation_phase is None:
            return WeightTransferResult(
                stride_to_foot_plant_ms=0.0,
                foot_plant_to_hip_rotation_ms=0.0,
            )

        # Detect key events
        stride_initiation = self._detect_stride_initiation(
            pose_sequence, stride_phase
        )
        foot_plant = self._detect_foot_plant(pose_sequence, stride_phase)
        hip_rotation_onset = self._detect_hip_rotation_onset(
            pose_sequence, rotation_phase
        )

        # Calculate timing
        stride_to_foot_plant_ms = (
            (foot_plant - stride_initiation) / fps * 1000.0
        )
        foot_plant_to_hip_rotation_ms = (
            (hip_rotation_onset - foot_plant) / fps * 1000.0
        )

        return WeightTransferResult(
            stride_to_foot_plant_ms=max(0.0, stride_to_foot_plant_ms),
            foot_plant_to_hip_rotation_ms=max(0.0, foot_plant_to_hip_rotation_ms),
        )

    def _detect_stride_initiation(
        self,
        pose_sequence: List[PoseResult],
        stride_phase: tuple[int, int],
    ) -> int:
        """Detect the frame where the front foot begins lifting (stride initiation).

        The stride initiation is detected by finding the frame where the
        front foot (left_ankle for right-handed batters) begins to move
        upward (y-coordinate decreases in image space).

        Args:
            pose_sequence: List of PoseResult.
            stride_phase: Tuple of (start_frame, end_frame) for stride phase.

        Returns:
            Frame index of stride initiation.
        """
        start_frame, end_frame = stride_phase

        # Filter poses within stride phase
        stride_poses = sorted(
            [p for p in pose_sequence if start_frame <= p.frame_index <= end_frame],
            key=lambda p: p.frame_index,
        )

        if len(stride_poses) < 2:
            return start_frame

        # Look for front foot (left_ankle) beginning to lift
        # In image coordinates, lifting means y decreases
        for i in range(1, len(stride_poses)):
            prev_ankle = self._find_keypoint(stride_poses[i - 1], "left_ankle")
            curr_ankle = self._find_keypoint(stride_poses[i], "left_ankle")

            if prev_ankle is not None and curr_ankle is not None:
                # Foot is lifting if y decreases (moves up in image)
                if curr_ankle.y < prev_ankle.y - 0.005:
                    return stride_poses[i].frame_index

        # Default to stride phase start
        return start_frame

    def _detect_foot_plant(
        self,
        pose_sequence: List[PoseResult],
        stride_phase: tuple[int, int],
    ) -> int:
        """Detect the frame where the front foot stabilizes (foot plant).

        The foot plant is detected by finding the frame where the front
        foot's vertical velocity approaches zero after the stride
        (y-coordinate stops changing significantly).

        Args:
            pose_sequence: List of PoseResult.
            stride_phase: Tuple of (start_frame, end_frame) for stride phase.

        Returns:
            Frame index of foot plant.
        """
        start_frame, end_frame = stride_phase

        # Filter poses within stride phase
        stride_poses = sorted(
            [p for p in pose_sequence if start_frame <= p.frame_index <= end_frame],
            key=lambda p: p.frame_index,
        )

        if len(stride_poses) < 3:
            return end_frame

        # Look for front foot stabilization (velocity near zero after movement)
        # Search from the end of stride phase backward
        stability_threshold = 0.003  # Minimal y-change indicates stability

        for i in range(len(stride_poses) - 1, 1, -1):
            curr_ankle = self._find_keypoint(stride_poses[i], "left_ankle")
            prev_ankle = self._find_keypoint(stride_poses[i - 1], "left_ankle")

            if curr_ankle is not None and prev_ankle is not None:
                y_change = abs(curr_ankle.y - prev_ankle.y)
                if y_change <= stability_threshold:
                    return stride_poses[i].frame_index

        # Default to stride phase end
        return end_frame

    def _detect_hip_rotation_onset(
        self,
        pose_sequence: List[PoseResult],
        rotation_phase: tuple[int, int],
    ) -> int:
        """Detect the frame where hip rotation begins.

        Hip rotation onset is detected by finding the frame where the
        angle of the hip line (left_hip to right_hip) begins to change
        significantly.

        Args:
            pose_sequence: List of PoseResult.
            rotation_phase: Tuple of (start_frame, end_frame) for rotation phase.

        Returns:
            Frame index of hip rotation onset.
        """
        start_frame, end_frame = rotation_phase

        # Filter poses within rotation phase
        rotation_poses = sorted(
            [p for p in pose_sequence if start_frame <= p.frame_index <= end_frame],
            key=lambda p: p.frame_index,
        )

        if len(rotation_poses) < 2:
            return start_frame

        # Calculate hip angle at each frame
        rotation_threshold = 2.0  # degrees per frame indicating rotation start

        for i in range(1, len(rotation_poses)):
            prev_angle = self._get_hip_angle(rotation_poses[i - 1])
            curr_angle = self._get_hip_angle(rotation_poses[i])

            if prev_angle is not None and curr_angle is not None:
                angle_change = abs(curr_angle - prev_angle)
                if angle_change >= rotation_threshold:
                    return rotation_poses[i].frame_index

        # Default to rotation phase start
        return start_frame

    def _get_hip_angle(self, pose: PoseResult) -> Optional[float]:
        """Calculate the angle of the hip line (left_hip to right_hip) in degrees.

        Args:
            pose: PoseResult with keypoints.

        Returns:
            Angle in degrees, or None if keypoints not available.
        """
        left_hip = self._find_keypoint(pose, "left_hip")
        right_hip = self._find_keypoint(pose, "right_hip")

        if left_hip is None or right_hip is None:
            return None

        dx = right_hip.x - left_hip.x
        dy = right_hip.y - left_hip.y

        if dx == 0.0 and dy == 0.0:
            return None

        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _find_keypoint(
        pose: PoseResult, name: str, min_confidence: float = 0.5
    ) -> Optional[Keypoint]:
        """Find a keypoint by name with minimum confidence."""
        for kp in pose.keypoints:
            if kp.name == name and kp.confidence >= min_confidence:
                return kp
        return None


class BatPathFlatnessEvaluator:
    """Evaluates bat path flatness through the hitting zone (Requirement 7.5).

    Measures the duration (ms) that the bat barrel stays within 5° of the
    swing plane. The swing plane is defined by the line from the batter's
    back shoulder to the pitch location at contact.
    """

    def evaluate_flatness(
        self,
        bat_trajectory: BatTrajectory,
        pose_sequence: List[PoseResult],
        impact_frame: int,
        fps: float,
    ) -> Dict[str, Any]:
        """Evaluate bat path flatness through the hitting zone.

        Measures how long the bat barrel stays within 5° of the swing plane.
        The swing plane is defined as the line from the batter's back shoulder
        to the pitch location (bat position) at contact.

        Args:
            bat_trajectory: Bat trajectory data.
            pose_sequence: List of PoseResult.
            impact_frame: Frame index of impact.
            fps: Video frame rate.

        Returns:
            Dictionary with:
            - flatness_duration_ms: Duration bat stays within 5° of plane
            - threshold_degrees: The 5° threshold
            - swing_plane_angle: Angle of the swing plane
            - frames_in_plane: Number of frames within threshold
        """
        if fps <= 0:
            return {
                "flatness_duration_ms": 0.0,
                "threshold_degrees": BAT_PATH_FLATNESS_THRESHOLD_DEGREES,
                "swing_plane_angle": 0.0,
                "frames_in_plane": 0,
            }

        # Find the swing plane angle
        swing_plane_angle = self._calculate_swing_plane_angle(
            pose_sequence, bat_trajectory, impact_frame
        )

        if swing_plane_angle is None:
            return {
                "flatness_duration_ms": 0.0,
                "threshold_degrees": BAT_PATH_FLATNESS_THRESHOLD_DEGREES,
                "swing_plane_angle": 0.0,
                "frames_in_plane": 0,
                "reason": "Could not determine swing plane",
            }

        # Get bat velocity angles at each frame and compare to swing plane
        detections_by_frame = {
            d.frame_index: d
            for d in bat_trajectory.detections
            if d.detected
        }

        # Sort frame indices
        sorted_frames = sorted(detections_by_frame.keys())

        # Calculate bat path angle at each consecutive frame pair
        frames_in_plane = 0
        for i in range(1, len(sorted_frames)):
            prev_frame = sorted_frames[i - 1]
            curr_frame = sorted_frames[i]

            prev_det = detections_by_frame[prev_frame]
            curr_det = detections_by_frame[curr_frame]

            dx = curr_det.position[0] - prev_det.position[0]
            dy = curr_det.position[1] - prev_det.position[1]

            if dx == 0.0 and dy == 0.0:
                continue

            # Calculate bat path angle (negate dy for image coordinates)
            path_angle = math.degrees(math.atan2(-dy, dx))

            # Check if within threshold of swing plane
            angle_diff = abs(path_angle - swing_plane_angle)
            # Handle angle wrapping
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            if angle_diff <= BAT_PATH_FLATNESS_THRESHOLD_DEGREES:
                frames_in_plane += 1

        flatness_duration_ms = (frames_in_plane / fps) * 1000.0

        return {
            "flatness_duration_ms": flatness_duration_ms,
            "threshold_degrees": BAT_PATH_FLATNESS_THRESHOLD_DEGREES,
            "swing_plane_angle": swing_plane_angle,
            "frames_in_plane": frames_in_plane,
        }

    def _calculate_swing_plane_angle(
        self,
        pose_sequence: List[PoseResult],
        bat_trajectory: BatTrajectory,
        impact_frame: int,
    ) -> Optional[float]:
        """Calculate the swing plane angle.

        The swing plane is defined by the line from the batter's back shoulder
        to the pitch location at contact (bat position at impact).

        For a right-handed batter, the back shoulder is the right shoulder.

        Args:
            pose_sequence: List of PoseResult.
            bat_trajectory: Bat trajectory data.
            impact_frame: Frame index of impact.

        Returns:
            Swing plane angle in degrees, or None if cannot be determined.
        """
        # Find bat position at impact
        bat_at_impact = None
        for d in bat_trajectory.detections:
            if d.frame_index == impact_frame and d.detected:
                bat_at_impact = d
                break

        if bat_at_impact is None:
            return None

        # Find pose at or near impact frame
        impact_pose = None
        for pose in pose_sequence:
            if pose.frame_index == impact_frame:
                impact_pose = pose
                break

        if impact_pose is None:
            # Find closest pose
            closest_dist = float("inf")
            for pose in pose_sequence:
                dist = abs(pose.frame_index - impact_frame)
                if dist < closest_dist:
                    closest_dist = dist
                    impact_pose = pose

        if impact_pose is None:
            return None

        # Find back shoulder (right_shoulder for right-handed batter)
        # Try right_shoulder first, fall back to left_shoulder
        back_shoulder = self._find_keypoint(impact_pose, "right_shoulder")
        if back_shoulder is None:
            back_shoulder = self._find_keypoint(impact_pose, "left_shoulder")

        if back_shoulder is None:
            return None

        # Calculate angle from back shoulder to bat position at impact
        # Note: bat position is in pixel coordinates, shoulder is normalized
        # For this calculation, we use the bat position directly since
        # both define the plane direction
        dx = bat_at_impact.position[0] - back_shoulder.x
        dy = bat_at_impact.position[1] - back_shoulder.y

        if dx == 0.0 and dy == 0.0:
            return None

        # Negate dy for image coordinate inversion
        angle = math.degrees(math.atan2(-dy, dx))
        return angle

    @staticmethod
    def _find_keypoint(
        pose: PoseResult, name: str, min_confidence: float = 0.5
    ) -> Optional[Keypoint]:
        """Find a keypoint by name with minimum confidence."""
        for kp in pose.keypoints:
            if kp.name == name and kp.confidence >= min_confidence:
                return kp
        return None


class ImprovementRanker:
    """Ranks improvement areas by deviation magnitude (Requirement 7.6).

    Identifies the top 3 areas for improvement from metric evaluations,
    ranked by the magnitude of deviation from the reference range
    (largest deviation first).
    """

    def rank_improvements(
        self, evaluations: List[MetricEvaluation]
    ) -> List[ImprovementArea]:
        """Rank improvement areas by deviation magnitude.

        Filters evaluations to only those with non-zero deviation (below_range
        or above_range), sorts by absolute deviation_percent in descending order,
        takes the top 3, and assigns ranks 1-3.

        Args:
            evaluations: List of MetricEvaluation from reference comparison.

        Returns:
            List of up to 3 ImprovementArea objects, ranked 1 (largest deviation)
            to 3 (smallest of top 3).
        """
        # Filter to only metrics with non-zero deviation (outside reference range)
        with_deviation = [
            e for e in evaluations if e.deviation_percent > 0
        ]

        # Sort by absolute deviation_percent in descending order
        sorted_evals = sorted(
            with_deviation,
            key=lambda e: abs(e.deviation_percent),
            reverse=True,
        )

        # Take top 3
        top_evals = sorted_evals[:3]

        # Build ImprovementArea list with rank assignment
        improvements: List[ImprovementArea] = []
        for rank, evaluation in enumerate(top_evals, start=1):
            improvement = ImprovementArea(
                metric_name=evaluation.metric_name,
                deviation_percent=evaluation.deviation_percent,
                current_value=evaluation.measured_value,
                target_range=(evaluation.reference_min, evaluation.reference_max),
                rank=rank,
                rating=evaluation.rating.value,
            )
            improvements.append(improvement)

        return improvements
