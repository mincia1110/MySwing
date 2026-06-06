"""Unit tests for BatterIdentifier and KeypointRoleAssigner.

Tests cover:
- Primary batter selection (closest to zone center)
- Single person trivial case
- Multiple persons at different distances
- Center of mass calculation
- Keypoint role assignment for right-handed batter
- Keypoint role assignment for left-handed batter
"""

import math

import pytest

from app.models.enums import BattingDirection
from app.models.pose import Keypoint, PoseResult
from app.pipeline.batter_identifier import BatterIdentifier, KeypointRoleAssigner


# --- Helpers ---


def _make_keypoint(x: float, y: float, name: str = "test") -> Keypoint:
    """Create a Keypoint with default confidence and z values."""
    return Keypoint(x=x, y=y, z=0.0, confidence=0.9, name=name)


def _make_pose_result(
    keypoints: list[Keypoint],
    person_id: int = 0,
    frame_index: int = 0,
) -> PoseResult:
    """Create a PoseResult with given keypoints."""
    return PoseResult(
        frame_index=frame_index,
        keypoints=keypoints,
        person_id=person_id,
        is_primary_batter=False,
        overall_confidence=0.9,
        is_low_confidence=False,
    )


# --- BatterIdentifier Tests ---


class TestBatterIdentifier:
    """Tests for BatterIdentifier.identify_primary_batter."""

    def setup_method(self) -> None:
        self.identifier = BatterIdentifier()

    def test_single_person_returns_that_person(self) -> None:
        """With a single person, that person is always the primary batter."""
        keypoints = [_make_keypoint(0.5, 0.5, "nose")]
        pose = _make_pose_result(keypoints, person_id=1)

        result = self.identifier.identify_primary_batter(
            [pose], batting_zone_center=(0.5, 0.5)
        )

        assert result is pose
        assert result.person_id == 1

    def test_closest_person_selected_from_multiple(self) -> None:
        """The person closest to the batting zone center is selected."""
        # Person A: center of mass at (0.2, 0.2)
        person_a = _make_pose_result(
            [_make_keypoint(0.2, 0.2)], person_id=1
        )
        # Person B: center of mass at (0.5, 0.5) - closest to zone center
        person_b = _make_pose_result(
            [_make_keypoint(0.5, 0.5)], person_id=2
        )
        # Person C: center of mass at (0.8, 0.8)
        person_c = _make_pose_result(
            [_make_keypoint(0.8, 0.8)], person_id=3
        )

        zone_center = (0.5, 0.5)
        result = self.identifier.identify_primary_batter(
            [person_a, person_b, person_c], zone_center
        )

        assert result.person_id == 2

    def test_multiple_persons_different_distances(self) -> None:
        """Correctly identifies the closest person among varied distances."""
        # Person at (0.3, 0.3) - distance to (0.5, 0.5) = sqrt(0.04+0.04) ≈ 0.283
        person_near = _make_pose_result(
            [_make_keypoint(0.3, 0.3)], person_id=10
        )
        # Person at (0.9, 0.1) - distance to (0.5, 0.5) = sqrt(0.16+0.16) ≈ 0.566
        person_far = _make_pose_result(
            [_make_keypoint(0.9, 0.1)], person_id=20
        )

        zone_center = (0.5, 0.5)
        result = self.identifier.identify_primary_batter(
            [person_far, person_near], zone_center
        )

        assert result.person_id == 10

    def test_empty_list_raises_value_error(self) -> None:
        """An empty list of pose results raises ValueError."""
        with pytest.raises(ValueError, match="No pose results provided"):
            self.identifier.identify_primary_batter([], (0.5, 0.5))

    def test_center_of_mass_with_multiple_keypoints(self) -> None:
        """Center of mass is the average of all keypoint positions."""
        # Person with keypoints at (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)
        # Center of mass = (0.4, 0.6)
        keypoints = [
            _make_keypoint(0.2, 0.4, "left_hip"),
            _make_keypoint(0.4, 0.6, "right_hip"),
            _make_keypoint(0.6, 0.8, "nose"),
        ]
        person = _make_pose_result(keypoints, person_id=1)

        # Zone center at (0.4, 0.6) - exactly at center of mass
        zone_center = (0.4, 0.6)
        result = self.identifier.identify_primary_batter([person], zone_center)

        assert result.person_id == 1


class TestCenterOfMass:
    """Tests for BatterIdentifier._calculate_center_of_mass."""

    def setup_method(self) -> None:
        self.identifier = BatterIdentifier()

    def test_single_keypoint(self) -> None:
        """Center of mass of a single keypoint is that keypoint's position."""
        keypoints = [_make_keypoint(0.3, 0.7)]
        pose = _make_pose_result(keypoints)

        com = self.identifier._calculate_center_of_mass(pose)

        assert com == pytest.approx((0.3, 0.7))

    def test_multiple_keypoints_average(self) -> None:
        """Center of mass is the average of all keypoint positions."""
        keypoints = [
            _make_keypoint(0.0, 0.0),
            _make_keypoint(1.0, 1.0),
        ]
        pose = _make_pose_result(keypoints)

        com = self.identifier._calculate_center_of_mass(pose)

        assert com == pytest.approx((0.5, 0.5))

    def test_three_keypoints(self) -> None:
        """Center of mass with three keypoints."""
        keypoints = [
            _make_keypoint(0.1, 0.2),
            _make_keypoint(0.3, 0.4),
            _make_keypoint(0.5, 0.6),
        ]
        pose = _make_pose_result(keypoints)

        com = self.identifier._calculate_center_of_mass(pose)

        assert com == pytest.approx((0.3, 0.4))

    def test_empty_keypoints_raises_value_error(self) -> None:
        """Empty keypoints list raises ValueError."""
        pose = _make_pose_result([])

        with pytest.raises(ValueError, match="no keypoints"):
            self.identifier._calculate_center_of_mass(pose)


class TestDistanceToZone:
    """Tests for BatterIdentifier._distance_to_zone."""

    def setup_method(self) -> None:
        self.identifier = BatterIdentifier()

    def test_same_point_zero_distance(self) -> None:
        """Distance between identical points is zero."""
        distance = self.identifier._distance_to_zone((0.5, 0.5), (0.5, 0.5))
        assert distance == pytest.approx(0.0)

    def test_known_distance(self) -> None:
        """Euclidean distance for a 3-4-5 triangle."""
        # (0, 0) to (3, 4) = 5
        distance = self.identifier._distance_to_zone((0.0, 0.0), (3.0, 4.0))
        assert distance == pytest.approx(5.0)

    def test_unit_distance(self) -> None:
        """Distance along one axis."""
        distance = self.identifier._distance_to_zone((0.0, 0.0), (1.0, 0.0))
        assert distance == pytest.approx(1.0)


# --- KeypointRoleAssigner Tests ---


class TestKeypointRoleAssigner:
    """Tests for KeypointRoleAssigner.assign_roles."""

    def setup_method(self) -> None:
        self.assigner = KeypointRoleAssigner()

    def test_right_handed_batter_roles(self) -> None:
        """Right-handed batter: lead_arm=left, power_arm=right, front_foot=left, back_foot=right."""
        keypoints = [
            _make_keypoint(0.3, 0.5, "left_wrist"),
            _make_keypoint(0.7, 0.5, "right_wrist"),
            _make_keypoint(0.3, 0.9, "left_ankle"),
            _make_keypoint(0.7, 0.9, "right_ankle"),
        ]
        pose = _make_pose_result(keypoints)

        roles = self.assigner.assign_roles(pose, BattingDirection.RIGHT)

        assert roles["lead_arm"] == "left"
        assert roles["power_arm"] == "right"
        assert roles["front_foot"] == "left"
        assert roles["back_foot"] == "right"

    def test_left_handed_batter_roles(self) -> None:
        """Left-handed batter: lead_arm=right, power_arm=left, front_foot=right, back_foot=left."""
        keypoints = [
            _make_keypoint(0.3, 0.5, "left_wrist"),
            _make_keypoint(0.7, 0.5, "right_wrist"),
            _make_keypoint(0.3, 0.9, "left_ankle"),
            _make_keypoint(0.7, 0.9, "right_ankle"),
        ]
        pose = _make_pose_result(keypoints)

        roles = self.assigner.assign_roles(pose, BattingDirection.LEFT)

        assert roles["lead_arm"] == "right"
        assert roles["power_arm"] == "left"
        assert roles["front_foot"] == "right"
        assert roles["back_foot"] == "left"

    def test_roles_contain_all_four_keys(self) -> None:
        """Role assignment always returns all four role keys."""
        keypoints = [_make_keypoint(0.5, 0.5, "nose")]
        pose = _make_pose_result(keypoints)

        for direction in BattingDirection:
            roles = self.assigner.assign_roles(pose, direction)
            assert set(roles.keys()) == {"lead_arm", "power_arm", "front_foot", "back_foot"}

    def test_right_and_left_are_opposite(self) -> None:
        """Right-handed and left-handed assignments are mirror images."""
        keypoints = [_make_keypoint(0.5, 0.5, "nose")]
        pose = _make_pose_result(keypoints)

        right_roles = self.assigner.assign_roles(pose, BattingDirection.RIGHT)
        left_roles = self.assigner.assign_roles(pose, BattingDirection.LEFT)

        # Each role should have opposite side assignments
        for role in ["lead_arm", "power_arm", "front_foot", "back_foot"]:
            assert right_roles[role] != left_roles[role]
