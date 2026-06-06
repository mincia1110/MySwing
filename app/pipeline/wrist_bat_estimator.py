"""Wrist-based bat position estimator.

Estimates bat position from MediaPipe wrist/elbow keypoints by extending
the elbow→wrist direction vector by the bat length. This is the production
bat trajectory source.
"""

from __future__ import annotations

import math
from typing import List

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.pose import Keypoint, PoseResult


# Default bat length as fraction of frame height (~34 inches for full-body shot)
DEFAULT_BAT_LENGTH_NORMALIZED = 0.25


class WristBatEstimator:
    """Estimates bat position from wrist/elbow keypoints.

    Uses the elbow→wrist vector direction to estimate where the bat head is,
    assuming the bat extends from the hands in the forearm direction.
    """

    def __init__(
        self,
        bat_length_normalized: float = DEFAULT_BAT_LENGTH_NORMALIZED,
        dominant_hand: str = "right",
        min_keypoint_confidence: float = 0.3,
    ) -> None:
        """Initialize WristBatEstimator.

        Args:
            bat_length_normalized: Bat length as fraction of frame height
                (default 0.25 ≈ 34 inches for average person in frame).
            dominant_hand: "right" or "left" for the power hand.
            min_keypoint_confidence: Minimum confidence to use a keypoint.
        """
        if not 0.0 < bat_length_normalized <= 1.0:
            raise ValueError("bat_length_normalized must be between 0.0 and 1.0")
        if dominant_hand not in ("right", "left"):
            raise ValueError("dominant_hand must be 'right' or 'left'")

        self.bat_length_normalized = bat_length_normalized
        self.dominant_hand = dominant_hand
        self.min_keypoint_confidence = min_keypoint_confidence

    def estimate_from_pose(
        self, pose_result: PoseResult, frame_index: int,
        video_width: int = 1, video_height: int = 1,
    ) -> BatDetectionResult:
        """Estimate bat position from pose keypoints.

        Strategy:
        1. Find the power hand wrist (right_wrist for right-handed, left for left)
        2. Find the corresponding elbow
        3. Calculate elbow→wrist direction vector
        4. Extend by bat_length to get bat head position
        5. Calculate orientation angle from the vector
        6. Return BatDetectionResult with estimated position

        Falls back to using wrist position alone if elbow is not available.

        Args:
            pose_result: Pose estimation result with keypoints.
            frame_index: Frame index for the result.
            video_width: Video frame width in pixels for aspect ratio correction
                of orientation angle. Use 1 (default) to skip correction.
                Use actual width when positions are in normalized (0-1) coordinates.
            video_height: Video frame height in pixels for aspect ratio correction
                of orientation angle. Use 1 (default) to skip correction.
                Use actual height when positions are in normalized (0-1) coordinates.

        Returns:
            BatDetectionResult with estimated bat position (is_predicted=True),
            or a no-detection result if keypoints are insufficient.
        """
        if not pose_result or not pose_result.keypoints:
            return self._create_no_detection(frame_index)

        # Get keypoints for dominant hand
        wrist_name = f"{self.dominant_hand}_wrist"
        elbow_name = f"{self.dominant_hand}_elbow"

        wrist = self._find_keypoint(pose_result.keypoints, wrist_name)
        elbow = self._find_keypoint(pose_result.keypoints, elbow_name)

        # Try non-dominant hand if dominant hand wrist not available
        if wrist is None:
            alt_hand = "left" if self.dominant_hand == "right" else "right"
            wrist_name = f"{alt_hand}_wrist"
            elbow_name = f"{alt_hand}_elbow"
            wrist = self._find_keypoint(pose_result.keypoints, wrist_name)
            elbow = self._find_keypoint(pose_result.keypoints, elbow_name)

        if wrist is None:
            return self._create_no_detection(frame_index)

        # If both wrist and elbow are available, use direction vector
        if elbow is not None:
            return self._estimate_with_direction(
                wrist, elbow, frame_index,
                video_width=video_width, video_height=video_height,
            )

        # Fallback: use wrist position only (less accurate)
        return self._estimate_from_wrist_only(wrist, frame_index)

    def estimate_trajectory(
        self, pose_sequence: List[PoseResult],
        video_width: int = 1, video_height: int = 1,
    ) -> BatTrajectory:
        """Build a BatTrajectory from pose-based estimates across all frames.

        Args:
            pose_sequence: List of PoseResult for each frame.
            video_width: Video frame width in pixels for aspect ratio correction.
                Use 1 (default) to skip correction.
                Use actual width when positions are in normalized (0-1) coordinates.
            video_height: Video frame height in pixels for aspect ratio correction.
                Use 1 (default) to skip correction.
                Use actual height when positions are in normalized (0-1) coordinates.

        Returns:
            BatTrajectory with estimated detections and speeds.
        """
        if not pose_sequence:
            return BatTrajectory(
                detections=[],
                bat_speed_pixels_per_frame=[],
                tracking_accuracy=0.0,
                tracking_failures=[],
            )

        detections: List[BatDetectionResult] = []
        for pose in pose_sequence:
            detection = self.estimate_from_pose(
                pose, pose.frame_index,
                video_width=video_width, video_height=video_height,
            )
            detections.append(detection)

        # Calculate speeds between consecutive detected frames
        speeds: List[float] = []
        for i in range(1, len(detections)):
            speed = self._calculate_speed(detections[i - 1], detections[i])
            speeds.append(speed)

        # Identify tracking failures
        tracking_failures = self._identify_tracking_failures(detections)

        # Calculate tracking accuracy
        detected_count = sum(1 for d in detections if d.detected)
        tracking_accuracy = detected_count / len(detections) if detections else 0.0

        return BatTrajectory(
            detections=detections,
            bat_speed_pixels_per_frame=speeds,
            tracking_accuracy=tracking_accuracy,
            tracking_failures=tracking_failures,
        )

    def _estimate_with_direction(
        self, wrist: Keypoint, elbow: Keypoint, frame_index: int,
        video_width: int = 1, video_height: int = 1,
    ) -> BatDetectionResult:
        """Estimate bat position using elbow→wrist direction vector.

        The bat extends from the wrist in the direction of the forearm
        (elbow→wrist vector), extended by the bat length.

        When positions are in normalized coordinates (0-1), the aspect ratio
        distorts the orientation angle. We correct by scaling the direction
        vector by video dimensions before computing the angle. Pass
        video_width=1 and video_height=1 (defaults) to skip correction.

        Args:
            wrist: Wrist keypoint (normalized coordinates).
            elbow: Elbow keypoint (normalized coordinates).
            frame_index: Frame index.
            video_width: Video frame width for aspect ratio correction (default 1).
            video_height: Video frame height for aspect ratio correction (default 1).

        Returns:
            BatDetectionResult with estimated position.
        """
        # Direction vector: elbow -> wrist in normalized coordinates.
        dx = wrist.x - elbow.x
        dy = wrist.y - elbow.y

        width = max(float(video_width), 1.0)
        height = max(float(video_height), 1.0)

        # Normalize in frame-pixel space. bat_length_normalized is defined as
        # a fraction of frame height, so using normalized x/y directly would
        # over-extend horizontal bats on wide video.
        dx_physical = dx * width
        dy_physical = dy * height
        magnitude = math.sqrt(dx_physical * dx_physical + dy_physical * dy_physical)
        if magnitude < 1e-6:
            return self._estimate_from_wrist_only(wrist, frame_index)

        dir_x = dx_physical / magnitude
        dir_y = dy_physical / magnitude

        bat_length_pixels = self.bat_length_normalized * height
        bat_head_x = wrist.x + (dir_x * bat_length_pixels / width)
        bat_head_y = wrist.y + (dir_y * bat_length_pixels / height)

        # Bat center: midpoint between wrist and bat head
        center_x = (wrist.x + bat_head_x) / 2.0
        center_y = (wrist.y + bat_head_y) / 2.0

        # Orientation angle from direction vector (degrees, 0-360)
        # Scale by video dimensions to correct for aspect ratio distortion
        # in normalized coordinates
        angle_rad = math.atan2(dy_physical, dx_physical)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360.0

        # Length in normalized coordinates (convert to approximate pixels
        # assuming a standard frame; actual pixel conversion happens downstream)
        length = self.bat_length_normalized

        # Confidence based on keypoint quality
        confidence = min(wrist.confidence, elbow.confidence) * 0.7

        return BatDetectionResult(
            frame_index=frame_index,
            detected=True,
            position=(center_x, center_y),
            orientation_angle=angle_deg,
            length_pixels=length,
            confidence=confidence,
            is_predicted=True,
            coordinate_space="normalized",
            bat_head_position=(bat_head_x, bat_head_y),
        )

    def _estimate_from_wrist_only(
        self, wrist: Keypoint, frame_index: int
    ) -> BatDetectionResult:
        """Estimate bat position from wrist position only (less accurate).

        When only the wrist is available (no elbow), use the wrist position
        as the bat center with reduced confidence.

        Args:
            wrist: Wrist keypoint.
            frame_index: Frame index.

        Returns:
            BatDetectionResult with wrist as center position.
        """
        return BatDetectionResult(
            frame_index=frame_index,
            detected=True,
            position=(wrist.x, wrist.y),
            orientation_angle=0.0,  # Unknown orientation
            length_pixels=self.bat_length_normalized,
            confidence=wrist.confidence * 0.4,  # Lower confidence
            is_predicted=True,
            coordinate_space="normalized",
            bat_head_position=None,
        )

    def _find_keypoint(
        self, keypoints: List[Keypoint], name: str
    ) -> Keypoint | None:
        """Find a keypoint by name with minimum confidence threshold.

        Args:
            keypoints: List of keypoints from pose estimation.
            name: Keypoint name to find (e.g., "right_wrist").

        Returns:
            Keypoint if found with sufficient confidence, None otherwise.
        """
        for kp in keypoints:
            if kp.name == name and kp.confidence >= self.min_keypoint_confidence:
                return kp
        return None

    def _create_no_detection(self, frame_index: int) -> BatDetectionResult:
        """Create a no-detection result for frames with insufficient keypoints."""
        return BatDetectionResult(
            frame_index=frame_index,
            detected=False,
            position=(0.0, 0.0),
            orientation_angle=0.0,
            length_pixels=0.0,
            confidence=0.0,
            is_predicted=True,
            coordinate_space="normalized",
        )

    @staticmethod
    def _calculate_speed(
        det1: BatDetectionResult, det2: BatDetectionResult
    ) -> float:
        """Calculate speed between two consecutive detections.

        Args:
            det1: Earlier detection.
            det2: Later detection.

        Returns:
            Speed in coordinate units per frame.
        """
        if not det1.detected or not det2.detected:
            return 0.0

        frame_gap = abs(det2.frame_index - det1.frame_index)
        if frame_gap == 0:
            return 0.0

        point1 = WristBatEstimator._motion_point(det1)
        point2 = WristBatEstimator._motion_point(det2)

        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        distance = math.sqrt(dx * dx + dy * dy)

        return distance / frame_gap

    @staticmethod
    def _motion_point(det: BatDetectionResult) -> tuple[float, float]:
        """Use the barrel/head point for motion when available."""
        if det.bat_head_position is not None:
            return det.bat_head_position
        return det.position

    @staticmethod
    def _identify_tracking_failures(
        detections: List[BatDetectionResult],
    ) -> List[tuple[int, int]]:
        """Identify intervals of consecutive tracking failures.

        Args:
            detections: List of BatDetectionResult sorted by frame_index.

        Returns:
            List of (start_frame, end_frame) tuples for failure intervals.
        """
        failures: List[tuple[int, int]] = []
        failure_start: int | None = None

        for det in detections:
            if not det.detected:
                if failure_start is None:
                    failure_start = det.frame_index
            else:
                if failure_start is not None:
                    failures.append((failure_start, det.frame_index - 1))
                    failure_start = None

        # Handle failure extending to end
        if failure_start is not None and detections:
            failures.append((failure_start, detections[-1].frame_index))

        return failures
