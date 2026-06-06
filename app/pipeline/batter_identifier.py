"""Primary batter identification and keypoint role assignment module.

This module provides:
- BatterIdentifier: Selects the primary batter from multiple detected persons
  based on proximity to the batting zone center (Requirement 3.6).
- KeypointRoleAssigner: Assigns keypoint roles (lead arm, power arm, front foot,
  back foot) based on the user's batting direction (Requirement 3.7).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from app.models.enums import BattingDirection
from app.models.pose import Keypoint, PoseResult


class BatterIdentifier:
    """Identifies the primary batter from multiple detected persons.

    When multiple persons are detected in a frame, the primary batter is
    defined as the person whose center of mass is closest to the center
    of the predefined batting zone region within the frame.
    """

    def identify_primary_batter(
        self,
        multi_pose_results: List[PoseResult],
        batting_zone_center: Tuple[float, float],
    ) -> PoseResult:
        """Select the person closest to the batting zone center.

        Args:
            multi_pose_results: List of PoseResult for each detected person.
            batting_zone_center: (x, y) center of the batting zone in
                normalized coordinates (0-1).

        Returns:
            The PoseResult of the person closest to the batting zone center.

        Raises:
            ValueError: If multi_pose_results is empty.
        """
        if not multi_pose_results:
            raise ValueError("No pose results provided for batter identification.")

        if len(multi_pose_results) == 1:
            return multi_pose_results[0]

        closest_person: PoseResult | None = None
        min_distance = float("inf")

        for pose_result in multi_pose_results:
            center_of_mass = self._calculate_center_of_mass(pose_result)
            distance = self._distance_to_zone(center_of_mass, batting_zone_center)

            if distance < min_distance:
                min_distance = distance
                closest_person = pose_result

        # This should never be None given the non-empty check above
        assert closest_person is not None
        return closest_person

    def _calculate_center_of_mass(
        self, pose_result: PoseResult
    ) -> Tuple[float, float]:
        """Calculate the center of mass as the average of all keypoint positions.

        Args:
            pose_result: PoseResult containing detected keypoints.

        Returns:
            (x, y) tuple representing the center of mass in normalized coordinates.

        Raises:
            ValueError: If pose_result has no keypoints.
        """
        if not pose_result.keypoints:
            raise ValueError("PoseResult has no keypoints for center of mass calculation.")

        total_x = sum(kp.x for kp in pose_result.keypoints)
        total_y = sum(kp.y for kp in pose_result.keypoints)
        n = len(pose_result.keypoints)

        return (total_x / n, total_y / n)

    def _distance_to_zone(
        self,
        center_of_mass: Tuple[float, float],
        zone_center: Tuple[float, float],
    ) -> float:
        """Calculate Euclidean distance between center of mass and zone center.

        Args:
            center_of_mass: (x, y) position of the person's center of mass.
            zone_center: (x, y) position of the batting zone center.

        Returns:
            Euclidean distance between the two points.
        """
        dx = center_of_mass[0] - zone_center[0]
        dy = center_of_mass[1] - zone_center[1]
        return math.sqrt(dx * dx + dy * dy)


class KeypointRoleAssigner:
    """Assigns keypoint roles based on batting direction.

    Maps body keypoints to functional roles (lead arm, power arm, front foot,
    back foot) according to the batter's handedness:
    - RIGHT-handed: lead_arm=left_arm, power_arm=right_arm,
                    front_foot=left_foot, back_foot=right_foot
    - LEFT-handed:  lead_arm=right_arm, power_arm=left_arm,
                    front_foot=right_foot, back_foot=left_foot
    """

    # Mapping from batting direction to role assignments
    _ROLE_MAPPINGS: Dict[BattingDirection, Dict[str, str]] = {
        BattingDirection.RIGHT: {
            "lead_arm": "left",
            "power_arm": "right",
            "front_foot": "left",
            "back_foot": "right",
        },
        BattingDirection.LEFT: {
            "lead_arm": "right",
            "power_arm": "left",
            "front_foot": "right",
            "back_foot": "left",
        },
    }

    def assign_roles(
        self,
        pose_result: PoseResult,
        batting_direction: BattingDirection,
    ) -> Dict[str, str]:
        """Map keypoints to functional roles based on batting direction.

        Args:
            pose_result: PoseResult containing detected keypoints.
            batting_direction: The batter's handedness (LEFT or RIGHT).

        Returns:
            Dictionary mapping role names to the body side:
            - "lead_arm": "left" or "right"
            - "power_arm": "left" or "right"
            - "front_foot": "left" or "right"
            - "back_foot": "left" or "right"
        """
        return dict(self._ROLE_MAPPINGS[batting_direction])
