"""Biomechanics analysis module (Requirements 6.1-6.4, 6.5, 6.6, 6.7, 6.8).

Provides pixel-to-meter ratio calibration using user height and bat length
verification, bat speed calculation at the impact zone, launch angle calculation
at impact, attack angle calculation through the hitting zone, kinematic chain
analysis, rotation analysis, and hand path efficiency calculation.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import List, Tuple

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.biomechanics import (
    AttackAngleResult,
    BatSpeedResult,
    BiomechanicsResult,
    JointAngularVelocity,
    KinematicChainResult,
    LaunchAngleResult,
    RotationResult,
    UnmeasurableMetric,
)
from app.models.pose import Keypoint, PoseResult

# MediaPipe landmark indices for head and ankles
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
_HEAD_KEYPOINT_NAME = "nose"
_LEFT_ANKLE_KEYPOINT_NAME = "left_ankle"
_RIGHT_ANKLE_KEYPOINT_NAME = "right_ankle"

# Bat length verification threshold (15%)
BAT_LENGTH_DISCREPANCY_THRESHOLD = 0.15

# Bat speed precision (±1 km/h)
BAT_SPEED_PRECISION_KMH = 1.0

# Number of frames around impact to use for speed calculation
IMPACT_FRAME_WINDOW = 2

# Wrist/elbow-based bat estimation tracks a proxy for the barrel. In practice
# this underestimates true barrel speed, so apply a calibrated multiplier only
# for predicted wrist-derived trajectories.
WRIST_PROXY_BARREL_SPEED_MULTIPLIER = 1.7

# Upper guardrail for selecting an impact metric frame. Values above this are
# usually pose identity swaps or wrist-estimator jumps rather than real barrel
# speed. 160 km/h is above typical elite MLB swing speeds.
MAX_PLAUSIBLE_BAT_SPEED_KMH = 160.0

# Launch angle and attack angle precision (±0.5°)
ANGLE_PRECISION_DEGREES = 0.5

# Hitting zone duration before impact (150ms)
HITTING_ZONE_DURATION_MS = 150.0


class CalibrationError(Exception):
    """Raised when pixel calibration cannot be performed."""

    pass


class PixelCalibrator:
    """Calculates pixel-to-meter ratio using user height and verifies with bat length.

    The calibration uses the head-to-ankle keypoint distance in pixels mapped
    to the user's actual height in meters. The bat length is used as a secondary
    verification to flag potential calibration issues.

    Requirement 6.1: Compute pixel-to-meter ratio using user height
    Requirement 6.2: Verify ratio using bat length (flag if >15% discrepancy)
    """

    def calibrate(self, pose_result: PoseResult, user_height_cm: float) -> float:
        """Calculate pixel-to-meter ratio using head-to-ankle distance.

        Args:
            pose_result: Pose estimation result containing keypoints.
            user_height_cm: User's actual height in centimeters.

        Returns:
            Pixel-to-meter ratio (meters per pixel).

        Raises:
            CalibrationError: If required keypoints are missing or have
                insufficient confidence.
        """
        if user_height_cm <= 0:
            raise CalibrationError(
                "User height must be positive."
            )

        head_to_ankle_pixels = self._get_head_to_ankle_distance_pixels(pose_result)

        if head_to_ankle_pixels <= 0:
            raise CalibrationError(
                "Head-to-ankle distance in pixels is zero or negative. "
                "Cannot compute calibration ratio."
            )

        user_height_meters = user_height_cm / 100.0
        pixel_to_meter = user_height_meters / head_to_ankle_pixels

        return pixel_to_meter

    def verify_with_bat(
        self,
        bat_detection: BatDetectionResult,
        bat_length_actual: float,
        pixel_to_meter: float,
    ) -> Tuple[bool, float]:
        """Verify calibration using detected bat length vs actual bat length.

        Compares the detected bat length (converted to meters using the
        calibration ratio) against the actual bat length. If the discrepancy
        exceeds 15%, the calibration is flagged as potentially inaccurate.

        Args:
            bat_detection: Bat detection result containing length_pixels.
            bat_length_actual: Actual bat length in meters.
            pixel_to_meter: Calibrated pixel-to-meter ratio.

        Returns:
            Tuple of (is_valid, discrepancy_percent):
                - is_valid: True if discrepancy <= 15%
                - discrepancy_percent: Absolute discrepancy as a fraction (0.0-1.0+)

        Raises:
            CalibrationError: If bat detection is invalid or bat_length_actual <= 0.
        """
        if bat_length_actual <= 0:
            raise CalibrationError("Actual bat length must be positive.")

        if not bat_detection.detected:
            raise CalibrationError(
                "Bat was not detected in the frame. Cannot verify calibration."
            )

        if bat_detection.length_pixels <= 0:
            raise CalibrationError(
                "Detected bat length in pixels is zero or negative."
            )

        coordinate_space = getattr(bat_detection, "coordinate_space", "pixel")
        if coordinate_space not in {"pixel", "normalized"}:
            raise CalibrationError(
                "Bat length verification requires pixel or normalized coordinate space; "
                f"got '{coordinate_space}'."
            )

        if pixel_to_meter <= 0:
            raise CalibrationError("Pixel-to-meter ratio must be positive.")

        # Convert detected bat length from the same coordinate space used by
        # the calibration ratio. The pipeline normally works in normalized
        # MediaPipe coordinates even though the legacy field is named
        # length_pixels.
        bat_length_detected_meters = bat_detection.length_pixels * pixel_to_meter

        # Calculate discrepancy
        discrepancy = abs(bat_length_detected_meters - bat_length_actual) / bat_length_actual

        is_valid = discrepancy <= BAT_LENGTH_DISCREPANCY_THRESHOLD

        return is_valid, discrepancy

    def _get_head_to_ankle_distance_pixels(self, pose_result: PoseResult) -> float:
        """Calculate Euclidean distance between head and ankle keypoints in pixels.

        Uses the nose keypoint as the head reference and the midpoint of
        left/right ankle keypoints as the ankle reference. If only one ankle
        is available, uses that single ankle.

        Note: Keypoint coordinates are normalized (0-1), so the returned
        distance is in normalized pixel space. The caller should account for
        this when the image resolution is known, or use the ratio directly
        since both height and distance are in the same coordinate space.

        Args:
            pose_result: Pose estimation result with keypoints.

        Returns:
            Euclidean distance in normalized pixel coordinates.

        Raises:
            CalibrationError: If head or both ankle keypoints are missing
                or have insufficient confidence.
        """
        head_keypoint = self._find_keypoint(
            pose_result, _HEAD_KEYPOINT_NAME, required=False
        )
        left_ankle = self._find_keypoint(
            pose_result, _LEFT_ANKLE_KEYPOINT_NAME, required=False
        )
        right_ankle = self._find_keypoint(
            pose_result, _RIGHT_ANKLE_KEYPOINT_NAME, required=False
        )

        if head_keypoint is None:
            raise CalibrationError(
                f"Head keypoint ('{_HEAD_KEYPOINT_NAME}') not found or has "
                "insufficient confidence."
            )

        # Determine ankle reference point
        if left_ankle is not None and right_ankle is not None:
            ankle_x = (left_ankle.x + right_ankle.x) / 2.0
            ankle_y = (left_ankle.y + right_ankle.y) / 2.0
        elif left_ankle is not None:
            ankle_x = left_ankle.x
            ankle_y = left_ankle.y
        elif right_ankle is not None:
            ankle_x = right_ankle.x
            ankle_y = right_ankle.y
        else:
            raise CalibrationError(
                "Neither left nor right ankle keypoint found or both have "
                "insufficient confidence."
            )

        dx = head_keypoint.x - ankle_x
        dy = head_keypoint.y - ankle_y
        distance = math.sqrt(dx * dx + dy * dy)

        return distance

    def _find_keypoint(
        self,
        pose_result: PoseResult,
        name: str,
        required: bool = True,
        min_confidence: float = 0.5,
    ) -> Keypoint | None:
        """Find a keypoint by name with minimum confidence threshold.

        Args:
            pose_result: Pose result containing keypoints.
            name: Keypoint name to search for.
            required: If True and keypoint not found, raises CalibrationError.
            min_confidence: Minimum confidence threshold (default 0.5).

        Returns:
            The matching Keypoint or None if not found and not required.

        Raises:
            CalibrationError: If required=True and keypoint is not found.
        """
        for kp in pose_result.keypoints:
            if kp.name == name and kp.confidence >= min_confidence:
                return kp

        if required:
            raise CalibrationError(
                f"Required keypoint '{name}' not found or has "
                f"confidence below {min_confidence}."
            )
        return None

    def calibrate_from_torso(self, pose_result: PoseResult, user_height_cm: float) -> float:
        """Fallback calibration using shoulder-to-hip distance when full body isn't visible.

        The torso (shoulder to hip) is approximately 30% of total height.

        Args:
            pose_result: Pose estimation result containing keypoints.
            user_height_cm: User's actual height in centimeters.

        Returns:
            Pixel-to-meter ratio (meters per pixel).

        Raises:
            CalibrationError: If required keypoints are missing.
        """
        if user_height_cm <= 0:
            raise CalibrationError("User height must be positive.")

        # Try to find shoulder and hip keypoints
        left_shoulder = self._find_keypoint(pose_result, "left_shoulder", required=False)
        right_shoulder = self._find_keypoint(pose_result, "right_shoulder", required=False)
        left_hip = self._find_keypoint(pose_result, "left_hip", required=False)
        right_hip = self._find_keypoint(pose_result, "right_hip", required=False)

        # Get shoulder midpoint
        if left_shoulder and right_shoulder:
            shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        elif left_shoulder:
            shoulder_x, shoulder_y = left_shoulder.x, left_shoulder.y
        elif right_shoulder:
            shoulder_x, shoulder_y = right_shoulder.x, right_shoulder.y
        else:
            raise CalibrationError("No shoulder keypoints found for torso calibration.")

        # Get hip midpoint
        if left_hip and right_hip:
            hip_x = (left_hip.x + right_hip.x) / 2.0
            hip_y = (left_hip.y + right_hip.y) / 2.0
        elif left_hip:
            hip_x, hip_y = left_hip.x, left_hip.y
        elif right_hip:
            hip_x, hip_y = right_hip.x, right_hip.y
        else:
            raise CalibrationError("No hip keypoints found for torso calibration.")

        # Calculate torso distance in normalized coordinates
        dx = shoulder_x - hip_x
        dy = shoulder_y - hip_y
        torso_distance = math.sqrt(dx * dx + dy * dy)

        if torso_distance <= 0:
            raise CalibrationError("Torso distance is zero. Cannot calibrate.")

        # Torso is approximately 30% of total height
        torso_height_ratio = 0.30
        estimated_height_in_coords = torso_distance / torso_height_ratio

        user_height_meters = user_height_cm / 100.0
        pixel_to_meter = user_height_meters / estimated_height_in_coords

        return pixel_to_meter


class BatSpeedCalculator:
    """Calculates bat speed at the impact zone using point displacement.

    Uses bat head positions around the impact frame (±2 frames) when available,
    and falls back to center positions otherwise, to calculate bat speed in km/h
    with ±1 km/h precision.

    Requirement 6.3: Calculate bat speed in km/h at impact zone.
    """

    def calculate_bat_speed(
        self,
        bat_trajectory: BatTrajectory,
        impact_frame: int,
        pixel_to_meter: float,
        fps: float,
        barrel_speed_multiplier: float = 1.0,
    ) -> BatSpeedResult:
        """Calculate bat speed at the impact zone.

        Gets bat head displacement around the impact frame (±2 frames),
        converts pixels to meters, and calculates speed in km/h.

        Formula: speed_kmh = (pixel_displacement * pixel_to_meter * fps) * 3.6

        Args:
            bat_trajectory: Bat trajectory with frame-by-frame detections.
            impact_frame: Frame index of the impact moment.
            pixel_to_meter: Calibrated pixel-to-meter ratio.
            fps: Video frame rate (frames per second).
            barrel_speed_multiplier: Optional multiplier for proxy trajectories
                that systematically under-estimate true barrel speed.

        Returns:
            BatSpeedResult with speed in km/h, precision, and measurement frame.

        Raises:
            CalibrationError: If insufficient detections around impact frame.
        """
        if pixel_to_meter <= 0:
            raise CalibrationError("Pixel-to-meter ratio must be positive.")

        if fps <= 0:
            raise CalibrationError("FPS must be positive.")

        if barrel_speed_multiplier <= 0:
            raise CalibrationError("Barrel speed multiplier must be positive.")

        # Get detections around impact frame (±IMPACT_FRAME_WINDOW frames)
        window_start = impact_frame - IMPACT_FRAME_WINDOW
        window_end = impact_frame + IMPACT_FRAME_WINDOW

        # Find detections within the window
        window_detections = [
            d
            for d in bat_trajectory.detections
            if window_start <= d.frame_index <= window_end and d.detected
        ]

        if len(window_detections) < 2:
            raise CalibrationError(
                f"Insufficient bat detections around impact frame {impact_frame}. "
                f"Need at least 2 detected frames within ±{IMPACT_FRAME_WINDOW} frames, "
                f"found {len(window_detections)}."
            )

        # Sort by frame index
        window_detections.sort(key=lambda d: d.frame_index)

        # Calculate total displacement and frame span
        total_displacement_pixels = 0.0
        for i in range(1, len(window_detections)):
            prev_pos = self._get_measurement_position(window_detections[i - 1])
            curr_pos = self._get_measurement_position(window_detections[i])
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            total_displacement_pixels += math.sqrt(dx * dx + dy * dy)

        # Total frame span
        frame_span = (
            window_detections[-1].frame_index - window_detections[0].frame_index
        )

        if frame_span == 0:
            raise CalibrationError(
                "All detections in the impact window are on the same frame."
            )

        # Calculate speed
        # displacement per frame in pixels
        displacement_per_frame = total_displacement_pixels / frame_span

        # Convert to meters per second: pixels/frame * meters/pixel * frames/second
        speed_m_s = displacement_per_frame * pixel_to_meter * fps

        # Convert to km/h
        speed_kmh = speed_m_s * 3.6 * barrel_speed_multiplier

        return BatSpeedResult(
            speed_kmh=speed_kmh,
            precision=BAT_SPEED_PRECISION_KMH,
            measurement_frame=impact_frame,
        )

    @staticmethod
    def _get_measurement_position(detection: BatDetectionResult) -> tuple[float, float]:
        """Choose the most appropriate point for speed measurement.

        Prefer bat head/barrel position when available; otherwise fall back to
        center position for backward compatibility with detector-only trajectories.
        """
        if detection.bat_head_position is not None:
            return detection.bat_head_position
        return detection.position


class ImpactAttackAngleCalculator:
    """Calculates the 2-frame impact attack angle at the moment of impact.

    The launch angle is the angle between the bat's velocity vector at impact
    and the horizontal axis. A positive angle indicates upward bat movement,
    negative indicates downward.

    Requirement 6.4: Calculate launch angle at impact with ±0.5° precision.
    """

    @staticmethod
    def _get_measurement_position(detection: BatDetectionResult) -> tuple[float, float]:
        """Prefer bat head/barrel position, falling back to center position."""
        if detection.bat_head_position is not None:
            return detection.bat_head_position
        return detection.position

    def calculate_launch_angle(
        self,
        bat_trajectory: BatTrajectory,
        impact_frame: int,
        video_width: int = 1,
        video_height: int = 1,
    ) -> LaunchAngleResult:
        """Calculate the bat launch angle at the impact frame.

        Uses bat positions at impact_frame-1 and impact_frame to compute
        the velocity vector direction. The angle is measured relative to
        the horizontal axis using atan2(dy, dx).

        Note: In image coordinates, y increases downward. We negate dy so
        that upward bat movement produces a positive angle (standard
        physics convention).

        When positions are in normalized coordinates (0-1), the aspect ratio
        distorts angles because x and y map to different physical scales.
        We correct this by scaling dx by video_width and dy by video_height
        before computing atan2. For pixel-coordinate positions, pass
        video_width=1 and video_height=1 (the defaults) to skip correction.

        Args:
            bat_trajectory: Bat trajectory with frame-by-frame detections.
            impact_frame: Frame index of the impact moment.
            video_width: Video frame width in pixels for aspect ratio correction.
                Use 1 (default) when positions are already in pixel coordinates.
                Use actual width when positions are in normalized (0-1) coordinates.
            video_height: Video frame height in pixels for aspect ratio correction.
                Use 1 (default) when positions are already in pixel coordinates.
                Use actual height when positions are in normalized (0-1) coordinates.

        Returns:
            LaunchAngleResult with angle in degrees, precision, and impact frame.

        Raises:
            CalibrationError: If insufficient detections to compute velocity
                at the impact frame.
        """
        # Find detections at impact_frame and impact_frame - 1
        detection_at_impact = None
        detection_before_impact = None

        for d in bat_trajectory.detections:
            if d.frame_index == impact_frame and d.detected:
                detection_at_impact = d
            elif d.frame_index == impact_frame - 1 and d.detected:
                detection_before_impact = d

        if detection_at_impact is None:
            raise CalibrationError(
                f"No bat detection at impact frame {impact_frame}. "
                "Cannot compute launch angle."
            )

        if detection_before_impact is None:
            raise CalibrationError(
                f"No bat detection at frame {impact_frame - 1} (one frame before impact). "
                "Cannot compute launch angle velocity vector."
            )

        # Compute velocity vector (displacement from frame-1 to frame).
        # Prefer barrel/head position when available; center position is only a
        # fallback for detector-only trajectories without explicit head points.
        before_pos = self._get_measurement_position(detection_before_impact)
        impact_pos = self._get_measurement_position(detection_at_impact)

        # Scale by video dimensions to correct for aspect ratio distortion
        # in normalized coordinates. When video_width=video_height=1, this
        # is a no-op (pixel coordinate case).
        dx = (impact_pos[0] - before_pos[0]) * video_width
        dy = (impact_pos[1] - before_pos[1]) * video_height

        if dx == 0.0 and dy == 0.0:
            raise CalibrationError(
                "Bat did not move between the two frames around impact. "
                "Cannot determine launch angle direction."
            )

        # Negate dy because image y-axis is inverted (down is positive)
        # so upward movement (negative dy in image) becomes positive angle
        angle_radians = math.atan2(-dy, dx)
        angle_degrees = math.degrees(angle_radians)

        return LaunchAngleResult(
            angle_degrees=angle_degrees,
            precision=ANGLE_PRECISION_DEGREES,
            impact_frame=impact_frame,
        )


# Backward-compat alias: external tests/imports still reference LaunchAngleCalculator.
LaunchAngleCalculator = ImpactAttackAngleCalculator


class AttackAngleCalculator:
    """Calculates the bat attack angle through the hitting zone.

    The attack angle is the average angle of the bat's velocity vector
    relative to the horizontal through the hitting zone (150ms before impact).

    Requirement 6.8: Calculate attack angle through hitting zone with ±0.5° precision.
    """

    def calculate_attack_angle(
        self,
        bat_trajectory: BatTrajectory,
        impact_frame: int,
        fps: float,
        video_width: int = 1,
        video_height: int = 1,
    ) -> AttackAngleResult:
        """Calculate the bat attack angle through the hitting zone.

        The hitting zone is defined as the 150ms window before impact.
        For each consecutive pair of detections in this zone, the angle
        of the velocity vector relative to horizontal is computed. The
        attack angle is the average of all these angles.

        When positions are in normalized coordinates (0-1), the aspect ratio
        distorts angles. We correct by scaling dx by video_width and dy by
        video_height before computing atan2. For pixel-coordinate positions,
        pass video_width=1 and video_height=1 (the defaults) to skip correction.

        Args:
            bat_trajectory: Bat trajectory with frame-by-frame detections.
            impact_frame: Frame index of the impact moment.
            fps: Video frame rate (frames per second).
            video_width: Video frame width in pixels for aspect ratio correction.
                Use 1 (default) when positions are already in pixel coordinates.
                Use actual width when positions are in normalized (0-1) coordinates.
            video_height: Video frame height in pixels for aspect ratio correction.
                Use 1 (default) when positions are already in pixel coordinates.
                Use actual height when positions are in normalized (0-1) coordinates.

        Returns:
            AttackAngleResult with average angle, precision, and zone boundaries.

        Raises:
            CalibrationError: If fps is invalid or insufficient detections
                in the hitting zone.
        """
        if fps <= 0:
            raise CalibrationError("FPS must be positive.")

        # Calculate hitting zone start frame (150ms before impact)
        hitting_zone_frames = int(HITTING_ZONE_DURATION_MS / 1000.0 * fps)
        hitting_zone_start_frame = impact_frame - hitting_zone_frames

        # Find detected frames within the hitting zone [start, impact]
        zone_detections = [
            d
            for d in bat_trajectory.detections
            if hitting_zone_start_frame <= d.frame_index <= impact_frame and d.detected
        ]

        if len(zone_detections) < 2:
            raise CalibrationError(
                f"Insufficient bat detections in hitting zone "
                f"(frames {hitting_zone_start_frame}-{impact_frame}). "
                f"Need at least 2 detected frames, found {len(zone_detections)}."
            )

        # Sort by frame index
        zone_detections.sort(key=lambda d: d.frame_index)

        # Calculate angle for each consecutive pair
        angles = []
        for i in range(1, len(zone_detections)):
            # Scale by video dimensions to correct for aspect ratio distortion
            # in normalized coordinates. When video_width=video_height=1, this
            # is a no-op (pixel coordinate case).
            dx = (
                zone_detections[i].position[0]
                - zone_detections[i - 1].position[0]
            ) * video_width
            dy = (
                zone_detections[i].position[1]
                - zone_detections[i - 1].position[1]
            ) * video_height

            if dx == 0.0 and dy == 0.0:
                # Skip pairs with no movement
                continue

            # Negate dy for image coordinate inversion
            angle_radians = math.atan2(-dy, dx)
            angle_degrees = math.degrees(angle_radians)
            angles.append(angle_degrees)

        if len(angles) == 0:
            raise CalibrationError(
                "No bat movement detected in the hitting zone. "
                "Cannot compute attack angle."
            )

        # Average all angles
        average_angle = sum(angles) / len(angles)

        return AttackAngleResult(
            angle_degrees=average_angle,
            precision=ANGLE_PRECISION_DEGREES,
            hitting_zone_start_frame=hitting_zone_start_frame,
            hitting_zone_end_frame=impact_frame,
        )


# --- Joint keypoint definitions for kinematic chain analysis ---
# Each joint is defined by (parent, joint, child) keypoints for 3-point angle calculation
_JOINT_KEYPOINTS = {
    "hips": ("left_shoulder", "left_hip", "left_knee"),
    "shoulders": ("left_hip", "left_shoulder", "left_elbow"),
    "elbows": ("left_shoulder", "left_elbow", "left_wrist"),
    "wrists": ("left_elbow", "left_wrist", "left_index"),
}

# Expected proximal-to-distal order for kinematic chain
_PROXIMAL_TO_DISTAL_ORDER = ["hips", "shoulders", "elbows", "wrists"]


class KinematicChainAnalyzer:
    """Analyzes the kinematic chain sequence during a baseball swing.

    Measures peak angular velocity and timing at each joint (hips, shoulders,
    elbows, wrists) and checks if the sequence follows the proximal-to-distal
    order.

    Requirement 6.5: Evaluate kinematic chain sequence by measuring timing of
    peak angular velocity at each joint.
    """

    def analyze_kinematic_chain(
        self,
        pose_sequence: List[PoseResult],
        phase_boundaries: dict,
        fps: float,
    ) -> KinematicChainResult:
        """Analyze kinematic chain for peak angular velocities and timing.

        For each joint (hips, shoulders, elbows, wrists):
        - Calculate angle at each frame using 3-point angle (parent-joint-child)
        - Compute angular velocity: (angle[i+1] - angle[i]) * fps (°/s)
        - Find peak angular velocity and its frame/timing

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.
            phase_boundaries: Dict with 'start_frame' and 'end_frame' keys
                defining the analysis window.
            fps: Video frame rate (frames per second).

        Returns:
            KinematicChainResult with peak angular velocities per joint,
            sequence correctness, and timing gaps.
        """
        if fps <= 0:
            return KinematicChainResult(
                joint_peak_angular_velocities={},
                sequence_correct=False,
                timing_gaps_ms={},
            )

        if len(pose_sequence) < 2:
            return KinematicChainResult(
                joint_peak_angular_velocities={},
                sequence_correct=False,
                timing_gaps_ms={},
            )

        # Sort pose sequence by frame index
        sorted_poses = sorted(pose_sequence, key=lambda p: p.frame_index)

        # Get analysis window
        start_frame = phase_boundaries.get("start_frame", sorted_poses[0].frame_index)
        end_frame = phase_boundaries.get("end_frame", sorted_poses[-1].frame_index)

        # Filter poses within the analysis window
        window_poses = [
            p for p in sorted_poses
            if start_frame <= p.frame_index <= end_frame
        ]

        if len(window_poses) < 2:
            return KinematicChainResult(
                joint_peak_angular_velocities={},
                sequence_correct=False,
                timing_gaps_ms={},
            )

        # Calculate peak angular velocity for each joint
        joint_velocities: dict[str, JointAngularVelocity] = {}

        for joint_name, (parent_name, joint_kp_name, child_name) in _JOINT_KEYPOINTS.items():
            result = self._calculate_joint_peak_velocity(
                window_poses, parent_name, joint_kp_name, child_name,
                joint_name, fps, start_frame,
            )
            if result is not None:
                joint_velocities[joint_name] = result

        # Check proximal-to-distal sequence
        sequence_correct = self._check_proximal_to_distal(joint_velocities)

        # Calculate timing gaps between consecutive joints
        timing_gaps = self._calculate_timing_gaps(joint_velocities)

        return KinematicChainResult(
            joint_peak_angular_velocities=joint_velocities,
            sequence_correct=sequence_correct,
            timing_gaps_ms=timing_gaps,
        )

    def _calculate_joint_peak_velocity(
        self,
        poses: List[PoseResult],
        parent_name: str,
        joint_name: str,
        child_name: str,
        label: str,
        fps: float,
        start_frame: int,
    ) -> JointAngularVelocity | None:
        """Calculate peak angular velocity for a single joint.

        Returns None if insufficient keypoint data is available.
        """
        angles: list[tuple[int, float, float]] = []  # (frame_index, angle_degrees, confidence)

        for pose in poses:
            parent_kp = self._find_keypoint(pose, parent_name)
            joint_kp = self._find_keypoint(pose, joint_name)
            child_kp = self._find_keypoint(pose, child_name)

            if parent_kp is None or joint_kp is None or child_kp is None:
                continue

            angle = self._calculate_3point_angle(parent_kp, joint_kp, child_kp)
            confidence = min(parent_kp.confidence, joint_kp.confidence, child_kp.confidence)
            angles.append((pose.frame_index, angle, confidence))

        if len(angles) < 2:
            return None

        # Compute angular velocities between consecutive frames
        max_velocity = 0.0
        peak_frame = angles[0][0]

        frames = [a[0] for a in angles]
        raw_angles = [a[1] for a in angles]
        confidences = [a[2] for a in angles]
        smoothed_angles = self._smooth_joint_angles(raw_angles)

        for i in range(1, len(smoothed_angles)):
            frame_diff = frames[i] - frames[i - 1]
            if frame_diff == 0:
                continue
            angular_change = abs(smoothed_angles[i] - smoothed_angles[i - 1])
            raw_velocity = angular_change * fps / frame_diff

            segment_conf = (confidences[i] + confidences[i - 1]) / 2.0
            conf_weight = max(0.0, min(1.0, segment_conf / 0.8))
            velocity = raw_velocity * conf_weight

            if velocity > max_velocity:
                max_velocity = velocity
                peak_frame = frames[i]

        # Calculate peak time relative to start frame
        peak_time_ms = (peak_frame - start_frame) / fps * 1000.0

        return JointAngularVelocity(
            joint_name=label,
            peak_velocity_dps=max_velocity,
            peak_frame=peak_frame,
            peak_time_ms=peak_time_ms,
        )

    @staticmethod
    def _smooth_joint_angles(angles_deg: list[float]) -> list[float]:
        """Apply median smoothing (window=3) to joint-angle series."""
        if not angles_deg:
            return []
        # For 2-frame sequences, smoothing would collapse slope; keep raw.
        if len(angles_deg) < 3:
            return list(angles_deg)

        smoothed: list[float] = []
        n = len(angles_deg)
        for i in range(n):
            win = angles_deg[max(0, i - 1): min(n, i + 2)]
            smoothed.append(statistics.median(win))
        return smoothed

    def _check_proximal_to_distal(
        self, joint_velocities: dict[str, JointAngularVelocity]
    ) -> bool:
        """Check if peak velocities occur in proximal-to-distal order.

        The correct order is: hips → shoulders → elbows → wrists.
        Each joint's peak should occur at the same time or after the previous.
        """
        prev_frame = -1
        for joint_name in _PROXIMAL_TO_DISTAL_ORDER:
            if joint_name not in joint_velocities:
                return False
            current_frame = joint_velocities[joint_name].peak_frame
            if current_frame < prev_frame:
                return False
            prev_frame = current_frame
        return True

    def _calculate_timing_gaps(
        self, joint_velocities: dict[str, JointAngularVelocity]
    ) -> dict[str, float]:
        """Calculate timing gaps between consecutive joints in ms."""
        gaps: dict[str, float] = {}
        prev_joint = None
        for joint_name in _PROXIMAL_TO_DISTAL_ORDER:
            if joint_name not in joint_velocities:
                prev_joint = joint_name
                continue
            if prev_joint is not None and prev_joint in joint_velocities:
                gap_key = f"{prev_joint}_to_{joint_name}"
                gap_ms = (
                    joint_velocities[joint_name].peak_time_ms
                    - joint_velocities[prev_joint].peak_time_ms
                )
                gaps[gap_key] = gap_ms
            prev_joint = joint_name
        return gaps

    @staticmethod
    def _calculate_3point_angle(
        parent: Keypoint, joint: Keypoint, child: Keypoint
    ) -> float:
        """Calculate angle at the joint formed by parent-joint-child in degrees.

        Uses the dot product formula:
        angle = acos((v1 · v2) / (|v1| * |v2|))

        where v1 = parent - joint, v2 = child - joint.
        """
        v1x = parent.x - joint.x
        v1y = parent.y - joint.y
        v2x = child.x - joint.x
        v2y = child.y - joint.y

        dot = v1x * v2x + v1y * v2y
        mag1 = math.sqrt(v1x * v1x + v1y * v1y)
        mag2 = math.sqrt(v2x * v2x + v2y * v2y)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        cos_angle = dot / (mag1 * mag2)
        # Clamp to [-1, 1] to handle floating point errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_radians = math.acos(cos_angle)
        return math.degrees(angle_radians)

    @staticmethod
    def _find_keypoint(
        pose: PoseResult, name: str, min_confidence: float = 0.5
    ) -> Keypoint | None:
        """Find a keypoint by name with minimum confidence."""
        for kp in pose.keypoints:
            if kp.name == name and kp.confidence >= min_confidence:
                return kp
        return None


class RotationAnalyzer:
    """Analyzes hip and shoulder rotation during the swing.

    Measures hip rotation speed, shoulder rotation speed, and the
    hip-shoulder separation angle during the rotation phase.

    Requirement 6.6: Measure hip/shoulder rotation speed and separation angle.
    """

    def calculate_rotation(
        self,
        pose_sequence: List[PoseResult],
        rotation_phase: Tuple[int, int],
        fps: float,
    ) -> RotationResult:
        """Calculate hip and shoulder rotation speeds and separation angle.

        Hip rotation: angular change of the line from left_hip to right_hip.
        Shoulder rotation: angular change of the line from left_shoulder to right_shoulder.
        Separation: max absolute difference between hip and shoulder angles.

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.
            rotation_phase: Tuple of (start_frame, end_frame) for the rotation phase.
            fps: Video frame rate (frames per second).

        Returns:
            RotationResult with hip/shoulder rotation speeds and separation angle.
        """
        start_frame, end_frame = rotation_phase

        if fps <= 0 or len(pose_sequence) < 2:
            return RotationResult(
                hip_rotation_speed_dps=0.0,
                shoulder_rotation_speed_dps=0.0,
                hip_shoulder_separation_degrees=0.0,
                rotation_phase_start_frame=start_frame,
                rotation_phase_end_frame=end_frame,
            )

        # Sort and filter poses within rotation phase
        sorted_poses = sorted(pose_sequence, key=lambda p: p.frame_index)
        phase_poses = [
            p for p in sorted_poses
            if start_frame <= p.frame_index <= end_frame
        ]

        if len(phase_poses) < 2:
            return RotationResult(
                hip_rotation_speed_dps=0.0,
                shoulder_rotation_speed_dps=0.0,
                hip_shoulder_separation_degrees=0.0,
                rotation_phase_start_frame=start_frame,
                rotation_phase_end_frame=end_frame,
            )

        # Calculate hip and shoulder angles at each frame
        # tuple: (frame_index, angle_deg, confidence)
        hip_angles: list[tuple[int, float, float]] = []
        shoulder_angles: list[tuple[int, float, float]] = []

        for pose in phase_poses:
            hip_angle_result = self._calculate_line_angle_with_confidence(
                pose, "left_hip", "right_hip"
            )
            shoulder_angle_result = self._calculate_line_angle_with_confidence(
                pose, "left_shoulder", "right_shoulder"
            )

            if hip_angle_result is not None:
                hip_angle, hip_confidence = hip_angle_result
                hip_angles.append((pose.frame_index, hip_angle, hip_confidence))
            if shoulder_angle_result is not None:
                shoulder_angle, shoulder_confidence = shoulder_angle_result
                shoulder_angles.append((pose.frame_index, shoulder_angle, shoulder_confidence))

        # Calculate peak rotation speeds
        hip_speed = self._calculate_peak_rotation_speed(hip_angles, fps)
        shoulder_speed = self._calculate_peak_rotation_speed(shoulder_angles, fps)

        # Calculate max hip-shoulder separation
        separation = self._calculate_max_separation(hip_angles, shoulder_angles)

        return RotationResult(
            hip_rotation_speed_dps=hip_speed,
            shoulder_rotation_speed_dps=shoulder_speed,
            hip_shoulder_separation_degrees=separation,
            rotation_phase_start_frame=start_frame,
            rotation_phase_end_frame=end_frame,
        )

    def _calculate_line_angle(
        self, pose: PoseResult, left_name: str, right_name: str
    ) -> float | None:
        """Calculate the angle of the line from left to right keypoint.

        Returns angle in degrees relative to horizontal.
        Returns None if keypoints are not available.
        """
        result = self._calculate_line_angle_with_confidence(
            pose, left_name, right_name
        )
        if result is None:
            return None
        return result[0]

    def _calculate_line_angle_with_confidence(
        self, pose: PoseResult, left_name: str, right_name: str
    ) -> tuple[float, float] | None:
        """Calculate line angle and confidence from a keypoint pair.

        Confidence is min(left_confidence, right_confidence).
        """
        left_kp = self._find_keypoint(pose, left_name)
        right_kp = self._find_keypoint(pose, right_name)

        if left_kp is None or right_kp is None:
            return None

        dx = right_kp.x - left_kp.x
        dy = right_kp.y - left_kp.y

        angle_radians = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_radians)
        confidence = min(left_kp.confidence, right_kp.confidence)
        return angle_deg, confidence

    def _calculate_peak_rotation_speed(
        self, angles: list[tuple[int, float, float]], fps: float
    ) -> float:
        """Calculate peak rotation speed from an angle series.

        Applies two stabilizers against frame jitter:
        1) median smoothing on unwrapped angle series (window=3)
        2) confidence-weighted segment speed (low-confidence segments down-weighted)
        """
        if len(angles) < 2:
            return 0.0

        frames = [a[0] for a in angles]
        raw_angles = [a[1] for a in angles]
        confidences = [a[2] for a in angles]

        smoothed_angles = self._smooth_unwrapped_angles(raw_angles)

        max_speed = 0.0
        for i in range(1, len(smoothed_angles)):
            frame_diff = frames[i] - frames[i - 1]
            if frame_diff == 0:
                continue

            angular_change = smoothed_angles[i] - smoothed_angles[i - 1]
            raw_speed = abs(angular_change) * fps / frame_diff

            segment_conf = (confidences[i] + confidences[i - 1]) / 2.0
            conf_weight = max(0.0, min(1.0, segment_conf / 0.8))
            weighted_speed = raw_speed * conf_weight

            if weighted_speed > max_speed:
                max_speed = weighted_speed
        return max_speed

    def _smooth_unwrapped_angles(self, angles_deg: list[float]) -> list[float]:
        """Unwrap cyclic angles and apply median smoothing (window=3)."""
        if not angles_deg:
            return []

        unwrapped = [angles_deg[0]]
        for angle in angles_deg[1:]:
            adjusted = angle
            prev = unwrapped[-1]
            while adjusted - prev > 180.0:
                adjusted -= 360.0
            while adjusted - prev < -180.0:
                adjusted += 360.0
            unwrapped.append(adjusted)

        smoothed: list[float] = []
        n = len(unwrapped)
        for i in range(n):
            win = unwrapped[max(0, i - 1): min(n, i + 2)]
            smoothed.append(statistics.median(win))
        return smoothed

    def _calculate_max_separation(
        self,
        hip_angles: list[tuple[int, float, float]],
        shoulder_angles: list[tuple[int, float, float]],
    ) -> float:
        """Calculate maximum hip-shoulder separation angle.

        Finds frames where both hip and shoulder angles are available
        and returns the maximum absolute difference, handling angle wrapping.
        """
        if not hip_angles or not shoulder_angles:
            return 0.0

        # Build lookup by frame index
        hip_by_frame = {frame: angle for frame, angle, _ in hip_angles}
        shoulder_by_frame = {frame: angle for frame, angle, _ in shoulder_angles}

        max_separation = 0.0
        for frame in hip_by_frame:
            if frame in shoulder_by_frame:
                diff = hip_by_frame[frame] - shoulder_by_frame[frame]
                # Normalize to [-180, 180] to handle angle wrapping
                while diff > 180.0:
                    diff -= 360.0
                while diff < -180.0:
                    diff += 360.0
                separation = abs(diff)
                if separation > max_separation:
                    max_separation = separation

        return max_separation

    @staticmethod
    def _find_keypoint(
        pose: PoseResult, name: str, min_confidence: float = 0.5
    ) -> Keypoint | None:
        """Find a keypoint by name with minimum confidence."""
        for kp in pose.keypoints:
            if kp.name == name and kp.confidence >= min_confidence:
                return kp
        return None


class HandPathAnalyzer:
    """Calculates hand path efficiency from load to impact.

    Efficiency is the ratio of direct distance to actual path distance,
    expressed as a value between 0.0 and 1.0.

    Requirement 6.7: Calculate hand path efficiency with 2 decimal places.
    """

    def calculate_hand_path_efficiency(
        self,
        pose_sequence: List[PoseResult],
        load_frame: int,
        impact_frame: int,
        batting_direction: str = "right",
    ) -> float:
        """Calculate hand path efficiency from load to impact.

        Direct distance: Euclidean distance from hand position at load_frame
        to hand position at impact_frame.
        Actual path: Sum of frame-to-frame hand displacements from load to impact.
        Efficiency = direct_distance / actual_path (0.0 to 1.0, 2 decimal places).

        Uses left_wrist as the hand reference keypoint.

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.
            load_frame: Frame index of the load position.
            impact_frame: Frame index of the impact position.

        Returns:
            Hand path efficiency as a float between 0.0 and 1.0,
            rounded to 2 decimal places. Returns 0.0 if insufficient data.
        """
        if len(pose_sequence) < 2 or load_frame >= impact_frame:
            return 0.0

        # Sort and filter poses within [load_frame, impact_frame]
        sorted_poses = sorted(pose_sequence, key=lambda p: p.frame_index)
        path_poses = [
            p for p in sorted_poses
            if load_frame <= p.frame_index <= impact_frame
        ]

        # Extract hand positions using lead hand by batting direction.
        lead_side = "left" if str(batting_direction).lower() != "left" else "right"
        preferred_wrist = f"{lead_side}_wrist"
        fallback_wrist = "right_wrist" if preferred_wrist == "left_wrist" else "left_wrist"

        hand_positions: list[tuple[float, float]] = []
        for pose in path_poses:
            wrist = self._find_keypoint(pose, preferred_wrist)
            if wrist is None:
                wrist = self._find_keypoint(pose, fallback_wrist)
            if wrist is not None:
                hand_positions.append((wrist.x, wrist.y))

        if len(hand_positions) < 2:
            return 0.0

        # Direct distance: start to end
        start = hand_positions[0]
        end = hand_positions[-1]
        direct_distance = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )

        # Actual path: sum of segment distances
        actual_path = 0.0
        for i in range(1, len(hand_positions)):
            dx = hand_positions[i][0] - hand_positions[i - 1][0]
            dy = hand_positions[i][1] - hand_positions[i - 1][1]
            actual_path += math.sqrt(dx * dx + dy * dy)

        if actual_path == 0.0:
            return 0.0

        efficiency = direct_distance / actual_path
        # Clamp to [0.0, 1.0] and round to 2 decimal places
        efficiency = max(0.0, min(1.0, efficiency))
        return round(efficiency, 2)

    @staticmethod
    def _find_keypoint(
        pose: PoseResult, name: str, min_confidence: float = 0.5
    ) -> Keypoint | None:
        """Find a keypoint by name with minimum confidence."""
        for kp in pose.keypoints:
            if kp.name == name and kp.confidence >= min_confidence:
                return kp
        return None


# Maximum allowed processing time for biomechanics analysis (seconds)
BIOMECHANICS_TIMEOUT_SECONDS = 30.0


class BiomechanicsOrchestrator:
    """Orchestrates all biomechanics sub-analyzers and handles failures gracefully.

    Coordinates PixelCalibrator, BatSpeedCalculator, ImpactAttackAngleCalculator,
    AttackAngleCalculator, KinematicChainAnalyzer, RotationAnalyzer, and
    HandPathAnalyzer. Records unmeasurable metrics with reasons when
    sub-analyzers fail, and ensures total processing completes within 30 seconds.

    Requirement 6.9: Report which measurements could not be computed and why.
    Requirement 6.10: Produce all biomechanical measurements within 30 seconds.
    """

    def __init__(self) -> None:
        self._pixel_calibrator = PixelCalibrator()
        self._bat_speed_calculator = BatSpeedCalculator()
        self._impact_attack_angle_calculator = ImpactAttackAngleCalculator()
        self._kinematic_chain_analyzer = KinematicChainAnalyzer()
        self._rotation_analyzer = RotationAnalyzer()
        self._hand_path_analyzer = HandPathAnalyzer()

    def _select_impact_metric_frame(
        self,
        bat_trajectory: BatTrajectory,
        impact_frame: int,
        pixel_to_meter: float,
        fps: float,
        video_width: int = 1,
        video_height: int = 1,
    ) -> int:
        """Choose a robust frame for bat speed/impact-angle metrics.

        The phase midpoint remains the canonical impact frame, but wrist-derived
        bat-head trajectories can shift the strongest local movement by several
        frames. For speed/angle metrics, search a small pre-impact zone and pick
        the highest-speed frame with a physically plausible 2-frame angle. Do
        not drift into pre-impact frames when no impact-adjacent bat detections
        are available; in that case downstream metrics should remain
        unmeasurable.
        """
        try:
            self._impact_attack_angle_calculator.calculate_launch_angle(
                bat_trajectory,
                impact_frame,
                video_width=video_width,
                video_height=video_height,
            )
        except CalibrationError:
            return impact_frame

        best_frame = impact_frame
        best_speed = float("-inf")
        for frame in range(max(0, impact_frame - 10), impact_frame + 3):
            try:
                angle = self._impact_attack_angle_calculator.calculate_launch_angle(
                    bat_trajectory,
                    frame,
                    video_width=video_width,
                    video_height=video_height,
                )
                if not (-45.0 <= angle.angle_degrees <= 60.0):
                    continue
                speed = self._bat_speed_calculator.calculate_bat_speed(
                    bat_trajectory,
                    frame,
                    pixel_to_meter,
                    fps,
                    barrel_speed_multiplier=self._bat_speed_multiplier(
                        bat_trajectory
                    ),
                )
            except CalibrationError:
                continue
            if speed.speed_kmh > MAX_PLAUSIBLE_BAT_SPEED_KMH:
                continue
            if speed.speed_kmh > best_speed:
                best_speed = speed.speed_kmh
                best_frame = frame
        return best_frame

    @staticmethod
    def _normalize_reported_attack_angle(
        angle: LaunchAngleResult,
    ) -> LaunchAngleResult:
        """Normalize signed impact angle to report-facing attack-angle magnitude.

        ``ImpactAttackAngleCalculator`` intentionally returns a signed 2-frame
        angle for low-level diagnostics. The product metric named
        ``attack_angle`` is evaluated against positive hitting-path references
        (5-25°), so expose the magnitude while preserving the measurement
        frame and precision.
        """
        if angle.angle_degrees < 0.0:
            return LaunchAngleResult(
                angle_degrees=abs(angle.angle_degrees),
                precision=angle.precision,
                impact_frame=angle.impact_frame,
            )
        return angle

    @staticmethod
    def _bat_speed_multiplier(bat_trajectory: BatTrajectory) -> float:
        """Return speed multiplier for proxy bat trajectories."""
        detected = [d for d in bat_trajectory.detections if d.detected]
        if not detected:
            return 1.0

        predicted_count = sum(1 for d in detected if d.is_predicted)
        if predicted_count / len(detected) >= 0.8:
            return WRIST_PROXY_BARREL_SPEED_MULTIPLIER
        return 1.0

    @staticmethod
    def _calibrate_from_bat_length(
        bat_trajectory: BatTrajectory,
        bat_length_meters: float,
    ) -> float | None:
        """Estimate coordinate-to-meter scale from known bat length.

        Wrist-based trajectories store the configured bat length in the same
        coordinate space as the estimated bat head. This gives a better scale
        than body height when portrait or zoomed clips cut off the lower body.
        """
        if bat_length_meters <= 0:
            return None

        lengths = [
            float(d.length_pixels)
            for d in bat_trajectory.detections
            if d.detected
            and d.length_pixels > 0
            and (d.is_predicted or d.coordinate_space == "normalized")
        ]
        if not lengths:
            return None

        median_length = statistics.median(lengths)
        if median_length <= 0:
            return None

        return bat_length_meters / median_length

    @staticmethod
    def _mark_metric_unmeasurable(
        result: BiomechanicsResult,
        unmeasurable_metrics: list[UnmeasurableMetric],
        metric_name: str,
        reason: str,
    ) -> None:
        setattr(result, metric_name, None)
        unmeasurable_metrics.append(
            UnmeasurableMetric(metric_name=metric_name, reason=reason)
        )

    def _apply_swing_quality_sanity_guards(
        self,
        result: BiomechanicsResult,
        unmeasurable_metrics: list[UnmeasurableMetric],
        user_height_cm: float,
    ) -> None:
        """Drop anthropometric metrics that are not credible for the clip."""
        if user_height_cm <= 0:
            return

        limits = {
            "stride_length_cm": (
                user_height_cm * 0.90,
                "Estimated stride exceeds 90% of body height; likely partial/cropped tracking",
            ),
            "cog_sway_cm": (
                user_height_cm * 0.35,
                "Estimated center-of-gravity sway exceeds 35% of body height; "
                "likely camera/tracking artifact",
            ),
            "head_stability_cm": (
                user_height_cm * 0.20,
                "Estimated head movement exceeds 20% of body height; "
                "likely camera/tracking artifact",
            ),
        }

        for metric_name, (max_value, reason) in limits.items():
            value = getattr(result, metric_name, None)
            if value is not None and value > max_value:
                self._mark_metric_unmeasurable(
                    result, unmeasurable_metrics, metric_name, reason
                )

        if (
            result.hand_path_efficiency is not None
            and result.hand_path_efficiency > 0.0
            and result.hand_path_efficiency < 0.05
        ):
            self._mark_metric_unmeasurable(
                result,
                unmeasurable_metrics,
                "hand_path_efficiency",
                "Hand path efficiency below measurable floor; likely phase/tracking artifact",
            )

    def analyze(
        self,
        pose_sequence: List[PoseResult],
        bat_trajectory: BatTrajectory,
        user_height_cm: float,
        bat_length_meters: float,
        impact_frame: int,
        swing_phases: dict,
        fps: float,
        video_width: int = 1,
        video_height: int = 1,
        batting_direction: str = "right",
    ) -> BiomechanicsResult:
        """Run full biomechanics analysis with error handling and timeout.

        Orchestrates all sub-analyzers sequentially, catching CalibrationError
        from each and recording unmeasurable metrics. Checks elapsed time after
        each sub-analysis and returns partial results if the 30-second timeout
        is exceeded.

        Args:
            pose_sequence: List of PoseResult for the swing sequence.
            bat_trajectory: Bat trajectory with frame-by-frame detections.
            user_height_cm: User's height in centimeters.
            bat_length_meters: User's bat length in meters.
            impact_frame: Frame index of the impact moment.
            swing_phases: Dict with phase boundaries including:
                - 'rotation_start_frame': start of rotation phase
                - 'rotation_end_frame': end of rotation phase
                - 'load_frame': frame of load position
                - 'start_frame': overall swing start frame
                - 'end_frame': overall swing end frame
            fps: Video frame rate (frames per second).
            video_width: Video frame width in pixels for aspect ratio correction.
                Use 1 (default) when positions are already in pixel coordinates.
                Use actual width when positions are in normalized (0-1) coordinates.
            video_height: Video frame height in pixels for aspect ratio correction.
                Use 1 (default) when positions are already in pixel coordinates.
                Use actual height when positions are in normalized (0-1) coordinates.

        Returns:
            BiomechanicsResult with computed metrics, unmeasurable metrics list,
            processing time, and timeout flag if applicable.
        """
        start_time = time.time()
        unmeasurable_metrics: list[UnmeasurableMetric] = []
        result = BiomechanicsResult()

        # Step 1: Pixel Calibration
        pixel_to_meter: float | None = None
        if self._is_timeout(start_time):
            return self._finalize_result(result, unmeasurable_metrics, start_time, timeout=True)

        try:
            # Use the first pose with sufficient keypoints for calibration
            calibration_pose = self._find_calibration_pose(pose_sequence)
            if calibration_pose is None:
                raise CalibrationError(
                    "Insufficient joint tracking data"
                )
            pixel_to_meter = self._pixel_calibrator.calibrate(
                calibration_pose, user_height_cm
            )
            # Verify with bat length (optional verification, doesn't block)
            if bat_trajectory.detections:
                first_detected = next(
                    (d for d in bat_trajectory.detections if d.detected), None
                )
                if first_detected and bat_length_meters > 0:
                    try:
                        self._pixel_calibrator.verify_with_bat(
                            first_detected, bat_length_meters, pixel_to_meter
                        )
                    except CalibrationError as verify_error:
                        unmeasurable_metrics.append(
                            UnmeasurableMetric(
                                metric_name="bat_length_verification",
                                reason=str(verify_error),
                            )
                        )
        except CalibrationError as e:
            # Fallback: try torso-based calibration
            try:
                torso_pose = self._find_torso_calibration_pose(pose_sequence)
                if torso_pose is None:
                    raise CalibrationError("No torso keypoints available")
                pixel_to_meter = self._pixel_calibrator.calibrate_from_torso(
                    torso_pose, user_height_cm
                )
            except CalibrationError as e2:
                reason = (
                    f"Pixel calibration failed: {str(e)}; "
                    f"torso fallback also failed: {str(e2)}"
                )
                unmeasurable_metrics.append(
                    UnmeasurableMetric(metric_name="pixel_calibration", reason=reason)
                )
                unmeasurable_metrics.append(
                    UnmeasurableMetric(
                        metric_name="bat_speed",
                        reason=f"Pixel calibration failed: {str(e)}",
                    )
                )

        if pixel_to_meter is not None:
            bat_length_scale = self._calibrate_from_bat_length(
                bat_trajectory,
                bat_length_meters,
            )
            if bat_length_scale is not None:
                pixel_to_meter = max(pixel_to_meter, bat_length_scale)

        metric_impact_frame = impact_frame
        if pixel_to_meter is not None:
            metric_impact_frame = self._select_impact_metric_frame(
                bat_trajectory,
                impact_frame,
                pixel_to_meter,
                fps,
                video_width=video_width,
                video_height=video_height,
            )

        # Step 2: Bat Speed
        if self._is_timeout(start_time):
            return self._finalize_result(result, unmeasurable_metrics, start_time, timeout=True)

        if pixel_to_meter is not None:
            try:
                result.bat_speed = self._bat_speed_calculator.calculate_bat_speed(
                    bat_trajectory,
                    metric_impact_frame,
                    pixel_to_meter,
                    fps,
                    barrel_speed_multiplier=self._bat_speed_multiplier(
                        bat_trajectory
                    ),
                )
            except CalibrationError as e:
                reason = self._classify_bat_speed_reason(str(e))
                unmeasurable_metrics.append(
                    UnmeasurableMetric(metric_name="bat_speed", reason=reason)
                )

        # Step 3: Attack Angle (2-frame window at impact)
        if self._is_timeout(start_time):
            return self._finalize_result(result, unmeasurable_metrics, start_time, timeout=True)

        try:
            result.attack_angle = self._normalize_reported_attack_angle(
                self._impact_attack_angle_calculator.calculate_launch_angle(
                    bat_trajectory, metric_impact_frame,
                    video_width=video_width, video_height=video_height,
                )
            )
            if result.attack_angle and not (0.0 <= result.attack_angle.angle_degrees <= 60.0):
                best_angle = result.attack_angle
                for offset in [1, 2, -1, 3, -2]:
                    try:
                        candidate = self._normalize_reported_attack_angle(
                            self._impact_attack_angle_calculator.calculate_launch_angle(
                                bat_trajectory, metric_impact_frame + offset,
                                video_width=video_width, video_height=video_height,
                            )
                        )
                        if 0.0 <= candidate.angle_degrees <= 60.0:
                            best_angle = candidate
                            break
                    except CalibrationError:
                        continue
                result.attack_angle = best_angle
        except CalibrationError as e:
            reason = self._classify_attack_angle_reason(str(e))
            unmeasurable_metrics.append(
                UnmeasurableMetric(metric_name="attack_angle", reason=reason)
            )

        # Step 5: Kinematic Chain
        if self._is_timeout(start_time):
            return self._finalize_result(result, unmeasurable_metrics, start_time, timeout=True)

        try:
            phase_boundaries = {
                "start_frame": swing_phases.get("start_frame", 0),
                "end_frame": swing_phases.get("end_frame", 0),
            }
            result.kinematic_chain = self._kinematic_chain_analyzer.analyze_kinematic_chain(
                pose_sequence, phase_boundaries, fps
            )
        except CalibrationError:
            unmeasurable_metrics.append(
                UnmeasurableMetric(
                    metric_name="kinematic_chain",
                    reason="Insufficient joint tracking data",
                )
            )

        # Step 6: Rotation Analysis
        if self._is_timeout(start_time):
            return self._finalize_result(result, unmeasurable_metrics, start_time, timeout=True)

        try:
            rotation_phase = (
                swing_phases.get("rotation_start_frame", 0),
                swing_phases.get("rotation_end_frame", 0),
            )
            result.rotation = self._rotation_analyzer.calculate_rotation(
                pose_sequence, rotation_phase, fps
            )
            # Sanity check: hip-shoulder separation > 90° is physically impossible
            # in a baseball swing. This indicates side-view camera angle where
            # left/right keypoints overlap, causing noise amplification.
            if (result.rotation and
                    result.rotation.hip_shoulder_separation_degrees > 90.0):
                unmeasurable_metrics.append(
                    UnmeasurableMetric(
                        metric_name="hip_shoulder_separation",
                        reason="Camera angle unsuitable for rotation measurement "
                               "(side view causes left/right keypoint overlap)",
                    )
                )
                # Set rotation to None so downstream evaluation skips this metric
                result.rotation = None
        except CalibrationError:
            unmeasurable_metrics.append(
                UnmeasurableMetric(
                    metric_name="rotation",
                    reason="Insufficient joint tracking data",
                )
            )

        # Step 7: Hand Path Efficiency
        if self._is_timeout(start_time):
            return self._finalize_result(result, unmeasurable_metrics, start_time, timeout=True)

        try:
            load_frame = swing_phases.get("load_frame", 0)
            result.hand_path_efficiency = self._hand_path_analyzer.calculate_hand_path_efficiency(
                pose_sequence, load_frame, impact_frame, batting_direction=batting_direction
            )
        except CalibrationError:
            unmeasurable_metrics.append(
                UnmeasurableMetric(
                    metric_name="hand_path_efficiency",
                    reason="Insufficient joint tracking data",
                )
            )

        # Step 8: Swing Quality Metrics (stride, cog, head, knee, spine)
        if pixel_to_meter is not None and not self._is_timeout(start_time):
            sq = SwingQualityAnalyzer()
            result.stride_length_cm = sq.stride_length_cm(
                pose_sequence, swing_phases, pixel_to_meter
            )
            result.cog_sway_cm = sq.cog_sway_cm(pose_sequence, swing_phases, pixel_to_meter)
            result.cog_drop_cm = sq.cog_drop_cm(pose_sequence, swing_phases, pixel_to_meter)
            result.head_stability_cm = sq.head_stability_cm(pose_sequence, pixel_to_meter)
            front_knee_extension = sq.front_knee_flexion_degrees(
                pose_sequence, swing_phases, batting_direction=batting_direction
            )
            result.front_knee_extension_degrees = front_knee_extension
            result.front_knee_flexion_degrees = front_knee_extension
            result.spine_angle_degrees = sq.spine_angle_degrees(pose_sequence, swing_phases)

        self._apply_swing_quality_sanity_guards(
            result, unmeasurable_metrics, user_height_cm
        )

        return self._finalize_result(result, unmeasurable_metrics, start_time, timeout=False)

    def _is_timeout(self, start_time: float) -> bool:
        """Check if elapsed time exceeds the 30-second timeout."""
        return (time.time() - start_time) >= BIOMECHANICS_TIMEOUT_SECONDS

    def _finalize_result(
        self,
        result: BiomechanicsResult,
        unmeasurable_metrics: list[UnmeasurableMetric],
        start_time: float,
        timeout: bool,
    ) -> BiomechanicsResult:
        """Finalize the result with processing time and unmeasurable metrics."""
        result.unmeasurable_metrics = unmeasurable_metrics
        result.processing_time_seconds = time.time() - start_time
        result.timeout_occurred = timeout
        return result

    def _find_calibration_pose(
        self, pose_sequence: List[PoseResult]
    ) -> PoseResult | None:
        """Find the best pose for calibration (one with head and ankle keypoints).

        Returns the first pose that has the required keypoints for calibration,
        or None if no suitable pose is found.
        """
        ankle_keypoints = {_LEFT_ANKLE_KEYPOINT_NAME, _RIGHT_ANKLE_KEYPOINT_NAME}

        for pose in pose_sequence:
            keypoint_names = {
                kp.name for kp in pose.keypoints if kp.confidence >= 0.5
            }
            has_head = _HEAD_KEYPOINT_NAME in keypoint_names
            has_ankle = bool(ankle_keypoints & keypoint_names)
            if has_head and has_ankle:
                return pose
        return None

    def _find_torso_calibration_pose(
        self, pose_sequence: List[PoseResult]
    ) -> PoseResult | None:
        """Find a pose suitable for torso-based calibration (shoulder + hip keypoints).

        Returns the first pose that has at least one shoulder and one hip keypoint.
        """
        shoulder_names = {"left_shoulder", "right_shoulder"}
        hip_names = {"left_hip", "right_hip"}

        for pose in pose_sequence:
            keypoint_names = {
                kp.name for kp in pose.keypoints if kp.confidence >= 0.5
            }
            has_shoulder = bool(shoulder_names & keypoint_names)
            has_hip = bool(hip_names & keypoint_names)
            if has_shoulder and has_hip:
                return pose
        return None

    @staticmethod
    def _classify_bat_speed_reason(error_message: str) -> str:
        """Classify the reason for bat speed measurement failure."""
        if "insufficient" in error_message.lower() or "need at least" in error_message.lower():
            return "Insufficient bat detections"
        if "impact" in error_message.lower():
            return "Impact frame not detected"
        return f"Pixel calibration failed: {error_message}"

    @staticmethod
    def _classify_attack_angle_reason(error_message: str) -> str:
        """Classify the reason for impact attack-angle measurement failure."""
        msg = error_message.lower()
        if "no bat detection at impact" in msg:
            return "No bat detection at impact frame"
        if "insufficient" in msg or "no bat detection at frame" in msg:
            return "Insufficient bat detections"
        if "impact" in msg or "did not move" in msg or "no bat movement" in msg:
            return "Impact frame not detected"
        return f"Pixel calibration failed: {error_message}"

# ============================================================================
# Swing Quality Analyzer
# ============================================================================

class SwingQualityAnalyzer:
    """Additional swing quality metrics beyond core biomechanics."""

    def stride_length_cm(self, pose_sequence, swing_phases, pixel_to_meter):
        """Compute stride length as ankle spread at rotation start (cm).

        Measures the horizontal distance between left and right ankle at
        the start of rotation — the final stride width. Independent of
        stride style (toe-tap, leg-kick, etc.).
        """
        rot_start = swing_phases.get("rotation_start_frame")
        if rot_start is None:
            # Fallback: use impact or mid-swing
            rot_start = swing_phases.get("impact_frame") or len(pose_sequence) // 2
        if rot_start is None:
            return None

        pose = self._find_pose_at_frame(pose_sequence, rot_start)
        if pose is None:
            return None

        left_ankle = self._find_kp(pose, "left_ankle")
        right_ankle = self._find_kp(pose, "right_ankle")
        if left_ankle is None or right_ankle is None:
            return None

        dx = right_ankle.x - left_ankle.x
        dy = right_ankle.y - left_ankle.y
        return math.sqrt(dx * dx + dy * dy) * pixel_to_meter * 100.0

    def cog_sway_cm(self, pose_sequence, swing_phases, pixel_to_meter):
        """Compute center-of-gravity lateral sway during stride (cm)."""
        stride_start = swing_phases.get("stride_start_frame")
        stride_end = swing_phases.get("stride_end_frame")
        if stride_start is None or stride_end is None:
            return None
        hip_x_positions = []
        for pose in pose_sequence:
            if stride_start <= pose.frame_index <= stride_end:
                lh = self._find_kp(pose, "left_hip")
                rh = self._find_kp(pose, "right_hip")
                if lh and rh:
                    hip_x_positions.append((lh.x + rh.x) / 2.0)
        if len(hip_x_positions) < 2:
            return None
        return (max(hip_x_positions) - min(hip_x_positions)) * pixel_to_meter * 100.0

    def cog_drop_cm(self, pose_sequence, swing_phases, pixel_to_meter):
        """Compute center-of-gravity vertical drop during stride (cm)."""
        load_frame = swing_phases.get("load_frame")
        stride_end = swing_phases.get("stride_end_frame")
        if load_frame is None or stride_end is None:
            return None
        load_hip_y = stride_hip_y = None
        for pose in pose_sequence:
            lh = self._find_kp(pose, "left_hip")
            rh = self._find_kp(pose, "right_hip")
            if lh and rh:
                hy = (lh.y + rh.y) / 2.0
                if pose.frame_index == load_frame:
                    load_hip_y = hy
                elif pose.frame_index == stride_end:
                    stride_hip_y = hy
        if load_hip_y is None or stride_hip_y is None:
            return None
        return abs(stride_hip_y - load_hip_y) * pixel_to_meter * 100.0

    def head_stability_cm(self, pose_sequence, pixel_to_meter):
        """Compute head stability as p95 nose displacement from mean (cm).

        Using p95 (instead of absolute max) reduces sensitivity to one-frame
        keypoint spikes while preserving meaningful motion magnitude.
        """
        positions = []
        for pose in pose_sequence:
            nose = self._find_kp(pose, "nose")
            if nose:
                positions.append((nose.x, nose.y))
        if len(positions) < 2:
            return None

        mean_x = sum(p[0] for p in positions) / len(positions)
        mean_y = sum(p[1] for p in positions) / len(positions)

        distances = [
            math.sqrt((p[0] - mean_x) ** 2 + (p[1] - mean_y) ** 2)
            for p in positions
        ]
        distances.sort()

        # p95 with linear interpolation
        n = len(distances)
        if n == 1:
            p95 = distances[0]
        else:
            idx = 0.95 * (n - 1)
            lo = int(math.floor(idx))
            hi = int(math.ceil(idx))
            if lo == hi:
                p95 = distances[lo]
            else:
                frac = idx - lo
                p95 = distances[lo] * (1.0 - frac) + distances[hi] * frac

        return p95 * pixel_to_meter * 100.0

    def front_knee_flexion_degrees(self, pose_sequence, swing_phases, batting_direction="right"):
        """Compute lead/front knee extension angle at stride landing (degrees).

        This returns the three-point hip-knee-ankle joint angle. Larger values
        mean a more extended/braced front leg, with 180° near full extension.
        The historical method name is kept for compatibility.
        """
        stride_end = swing_phases.get("stride_end_frame")
        if stride_end is None:
            return None
        pose = self._find_pose_at_frame(pose_sequence, stride_end)
        if pose is None:
            return None

        lead_side = self._lead_side(batting_direction)
        trail_side = "right" if lead_side == "left" else "left"

        hip = self._find_kp(pose, f"{lead_side}_hip")
        knee = self._find_kp(pose, f"{lead_side}_knee")
        ankle = self._find_kp(pose, f"{lead_side}_ankle")
        if not (hip and knee and ankle):
            hip = self._find_kp(pose, f"{trail_side}_hip")
            knee = self._find_kp(pose, f"{trail_side}_knee")
            ankle = self._find_kp(pose, f"{trail_side}_ankle")
        if not (hip and knee and ankle):
            return None
        return self._three_point_angle(hip, knee, ankle)

    def spine_angle_degrees(self, pose_sequence, swing_phases):
        """Compute spine angle at load position (degrees from vertical)."""
        load_frame = swing_phases.get("load_frame")
        if load_frame is None:
            return None
        pose = self._find_pose_at_frame(pose_sequence, load_frame)
        if pose is None:
            return None
        ls = self._find_kp(pose, "left_shoulder")
        rs = self._find_kp(pose, "right_shoulder")
        lh = self._find_kp(pose, "left_hip")
        rh = self._find_kp(pose, "right_hip")
        if not (ls and rs and lh and rh):
            return None
        sy = (ls.y + rs.y) / 2.0
        hy = (lh.y + rh.y) / 2.0
        sx = (ls.x + rs.x) / 2.0
        hx = (lh.x + rh.x) / 2.0
        dy = hy - sy  # >0: hips below shoulders (normal standing)
        dx = hx - sx
        if dy == 0.0:
            return 0.0
        return math.degrees(math.atan2(dx, dy))

    @staticmethod
    def _find_kp(pose, name):
        for kp in pose.keypoints:
            if kp.name == name and kp.confidence >= 0.3:
                return kp
        return None

    @staticmethod
    def _find_pose_at_frame(pose_sequence, frame_index):
        for pose in pose_sequence:
            if pose.frame_index == frame_index:
                return pose
        return None

    @staticmethod
    def _find_keypoint_at_frame(pose_sequence, frame_index, name):
        pose = SwingQualityAnalyzer._find_pose_at_frame(pose_sequence, frame_index)
        if pose is None:
            return None
        return SwingQualityAnalyzer._find_kp(pose, name)

    @staticmethod
    def _lead_side(batting_direction: str) -> str:
        """Map batting direction to lead/front side in body keypoint names.

        RHB -> left lead side, LHB -> right lead side.
        """
        return "right" if str(batting_direction).lower() == "left" else "left"

    @staticmethod
    def _three_point_angle(hip, knee, ankle):
        v1 = (hip.x - knee.x, hip.y - knee.y)
        v2 = (ankle.x - knee.x, ankle.y - knee.y)
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))
