"""Unit tests for portrait (vertical) video support.

Tests that angle calculations produce correct results when video dimensions
are provided, particularly for portrait mode (width < height) videos where
normalized coordinates have different physical scales for x and y.
"""

import math

import pytest

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.pose import Keypoint, PoseResult
from app.pipeline.biomechanics_analyzer import (
    AttackAngleCalculator,
    LaunchAngleCalculator,
)
from app.pipeline.wrist_bat_estimator import WristBatEstimator


# --- Helper functions ---


def _make_detection(
    frame_index: int,
    position: tuple[float, float] = (0.5, 0.5),
    detected: bool = True,
) -> BatDetectionResult:
    """Create a BatDetectionResult with given parameters."""
    return BatDetectionResult(
        frame_index=frame_index,
        detected=detected,
        position=position,
        orientation_angle=0.0,
        length_pixels=0.25,
        confidence=0.9,
        is_predicted=True,
    )


def _make_keypoint(name: str, x: float, y: float, confidence: float = 0.9) -> Keypoint:
    """Create a Keypoint with given parameters."""
    return Keypoint(name=name, x=x, y=y, z=0.0, confidence=confidence)


def _make_pose_result(frame_index: int, keypoints: list[Keypoint]) -> PoseResult:
    """Create a PoseResult with given keypoints."""
    return PoseResult(
        frame_index=frame_index,
        keypoints=keypoints,
        person_id=0,
        is_primary_batter=True,
        overall_confidence=0.9,
        is_low_confidence=False,
    )


# --- LaunchAngleCalculator portrait video tests ---


class TestLaunchAnglePortraitVideo:
    """Test launch angle calculations with portrait video dimensions."""

    def test_horizontal_movement_portrait_normalized_coords(self):
        """Horizontal movement in portrait video (1080x1920) with normalized coords.

        A movement of (0.1, 0) in normalized coords maps to:
        - dx_physical = 0.1 * 1080 = 108 pixels
        - dy_physical = 0 * 1920 = 0 pixels
        - angle = atan2(0, 108) = 0° (correct)
        """
        calculator = LaunchAngleCalculator()
        detections = [
            _make_detection(9, position=(0.4, 0.5)),
            _make_detection(10, position=(0.5, 0.5)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(
            trajectory, impact_frame=10,
            video_width=1080, video_height=1920,
        )

        assert abs(result.angle_degrees - 0.0) < 0.01

    def test_diagonal_movement_portrait_vs_landscape(self):
        """Same normalized displacement gives different angles in portrait vs landscape.

        Movement of (0.1, -0.1) in normalized coords:
        - Landscape (1920x1080): dx=192, dy=-108 → angle = atan2(108, 192) ≈ 29.4°
        - Portrait (1080x1920): dx=108, dy=-192 → angle = atan2(192, 108) ≈ 60.6°
        - Square (1:1): dx=0.1, dy=-0.1 → angle = atan2(0.1, 0.1) = 45°
        """
        calculator = LaunchAngleCalculator()
        detections = [
            _make_detection(9, position=(0.4, 0.6)),
            _make_detection(10, position=(0.5, 0.5)),  # moved right and up
        ]
        trajectory = BatTrajectory(detections=detections)

        # Landscape: wider frame means horizontal movement is physically larger
        result_landscape = calculator.calculate_launch_angle(
            trajectory, impact_frame=10,
            video_width=1920, video_height=1080,
        )

        # Portrait: taller frame means vertical movement is physically larger
        result_portrait = calculator.calculate_launch_angle(
            trajectory, impact_frame=10,
            video_width=1080, video_height=1920,
        )

        # No correction (default): treats as square
        result_no_correction = calculator.calculate_launch_angle(
            trajectory, impact_frame=10,
        )

        # Landscape should give a shallower angle (more horizontal)
        assert result_landscape.angle_degrees < 45.0
        assert abs(result_landscape.angle_degrees - math.degrees(math.atan2(108, 192))) < 0.01

        # Portrait should give a steeper angle (more vertical)
        assert result_portrait.angle_degrees > 45.0
        assert abs(result_portrait.angle_degrees - math.degrees(math.atan2(192, 108))) < 0.01

        # No correction gives 45° (equal dx and dy)
        assert abs(result_no_correction.angle_degrees - 45.0) < 0.01

    def test_portrait_horizontal_swing_not_distorted(self):
        """A purely horizontal swing in portrait mode should still be ~0°.

        This is the key test: in portrait video, a rightward horizontal
        movement should produce ~0° regardless of aspect ratio.
        """
        calculator = LaunchAngleCalculator()
        # Pure horizontal movement in normalized coords
        detections = [
            _make_detection(9, position=(0.3, 0.5)),
            _make_detection(10, position=(0.5, 0.5)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(
            trajectory, impact_frame=10,
            video_width=1080, video_height=1920,
        )

        assert abs(result.angle_degrees - 0.0) < 0.01

    def test_portrait_typical_launch_angle(self):
        """Typical 10° launch angle in portrait video.

        In portrait (1080x1920), for a 10° angle:
        - We need atan2(dy_physical, dx_physical) = 10°
        - tan(10°) = dy_physical / dx_physical
        - If dx_norm = 0.1, then dx_physical = 0.1 * 1080 = 108
        - dy_physical = 108 * tan(10°) ≈ 19.04
        - dy_norm = 19.04 / 1920 ≈ 0.00992
        - In image coords, upward = negative dy, so dy_norm = -0.00992
        """
        calculator = LaunchAngleCalculator()
        target_angle = 10.0
        dx_norm = 0.1
        # Calculate required dy_norm for portrait video
        dx_physical = dx_norm * 1080
        dy_physical = dx_physical * math.tan(math.radians(target_angle))
        dy_norm = dy_physical / 1920  # Convert back to normalized
        # In image coords, upward = negative dy
        detections = [
            _make_detection(9, position=(0.4, 0.5)),
            _make_detection(10, position=(0.4 + dx_norm, 0.5 - dy_norm)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(
            trajectory, impact_frame=10,
            video_width=1080, video_height=1920,
        )

        assert abs(result.angle_degrees - target_angle) < 0.5

    def test_default_no_correction_preserves_pixel_behavior(self):
        """Default (no video dimensions) preserves pixel-coordinate behavior.

        When positions are in pixel coordinates, the default video_width=1,
        video_height=1 means no scaling is applied.
        """
        calculator = LaunchAngleCalculator()
        # Pixel coordinates: 45° movement
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(150.0, 150.0)),  # dx=50, dy=-50
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=10)

        # atan2(-(-50), 50) = atan2(50, 50) = 45°
        assert abs(result.angle_degrees - 45.0) < 0.01


# --- AttackAngleCalculator portrait video tests ---


class TestAttackAnglePortraitVideo:
    """Test attack angle calculations with portrait video dimensions."""

    def test_constant_horizontal_angle_portrait(self):
        """Constant horizontal movement in portrait video gives 0° attack angle."""
        calculator = AttackAngleCalculator()
        # fps=120, 150ms = 18 frames, zone: frames 82-100
        detections = []
        for i in range(82, 101):
            x = 0.3 + (i - 82) * 0.01  # horizontal movement in normalized coords
            y = 0.5  # no vertical movement
            detections.append(_make_detection(i, position=(x, y)))

        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_attack_angle(
            trajectory, impact_frame=100, fps=120.0,
            video_width=1080, video_height=1920,
        )

        assert abs(result.angle_degrees - 0.0) < 0.5

    def test_portrait_vs_landscape_attack_angle(self):
        """Same normalized movement gives different attack angles in portrait vs landscape."""
        calculator = AttackAngleCalculator()
        # Diagonal movement: dx_norm=0.01, dy_norm=-0.01 per frame (upward-right)
        detections = []
        for i in range(82, 101):
            x = 0.3 + (i - 82) * 0.01
            y = 0.5 - (i - 82) * 0.01  # upward in image coords
            detections.append(_make_detection(i, position=(x, y)))

        trajectory = BatTrajectory(detections=detections)

        result_portrait = calculator.calculate_attack_angle(
            trajectory, impact_frame=100, fps=120.0,
            video_width=1080, video_height=1920,
        )

        result_landscape = calculator.calculate_attack_angle(
            trajectory, impact_frame=100, fps=120.0,
            video_width=1920, video_height=1080,
        )

        # Portrait: vertical movement is physically larger → steeper angle
        assert result_portrait.angle_degrees > 45.0

        # Landscape: horizontal movement is physically larger → shallower angle
        assert result_landscape.angle_degrees < 45.0


# --- WristBatEstimator portrait video tests ---


class TestWristBatEstimatorPortraitVideo:
    """Test wrist bat estimator orientation angle with portrait video dimensions."""

    def test_horizontal_direction_portrait(self):
        """Horizontal forearm direction in portrait video gives ~0° orientation."""
        estimator = WristBatEstimator(bat_length_normalized=0.25)
        keypoints = [
            _make_keypoint("right_elbow", x=0.3, y=0.5),
            _make_keypoint("right_wrist", x=0.5, y=0.5),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(
            pose, frame_index=0,
            video_width=1080, video_height=1920,
        )

        # Pure horizontal direction → 0°
        assert abs(result.orientation_angle - 0.0) < 1.0

    def test_diagonal_direction_portrait_vs_landscape(self):
        """Diagonal forearm gives different orientation in portrait vs landscape."""
        estimator = WristBatEstimator(bat_length_normalized=0.25)
        keypoints = [
            _make_keypoint("right_elbow", x=0.4, y=0.6),
            _make_keypoint("right_wrist", x=0.6, y=0.4),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result_portrait = estimator.estimate_from_pose(
            pose, frame_index=0,
            video_width=1080, video_height=1920,
        )

        result_landscape = estimator.estimate_from_pose(
            pose, frame_index=0,
            video_width=1920, video_height=1080,
        )

        # In portrait, vertical component is physically larger → steeper angle
        # In landscape, horizontal component is physically larger → shallower angle
        # Direction is (0.2, -0.2) in normalized coords
        # Portrait: atan2(-0.2*1920, 0.2*1080) = atan2(-384, 216) → negative angle → +360
        # Landscape: atan2(-0.2*1080, 0.2*1920) = atan2(-216, 384) → negative angle → +360
        assert result_portrait.orientation_angle != result_landscape.orientation_angle

    def test_estimate_trajectory_with_portrait_dimensions(self):
        """Trajectory estimation passes video dimensions through correctly."""
        estimator = WristBatEstimator(bat_length_normalized=0.25)
        poses = [
            _make_pose_result(
                frame_index=i,
                keypoints=[
                    _make_keypoint("right_elbow", x=0.3, y=0.5),
                    _make_keypoint("right_wrist", x=0.5, y=0.5),
                ],
            )
            for i in range(3)
        ]

        trajectory = estimator.estimate_trajectory(
            poses, video_width=1080, video_height=1920,
        )

        # All detections should have horizontal orientation (~0°)
        for det in trajectory.detections:
            assert det.detected
            assert abs(det.orientation_angle - 0.0) < 1.0


# --- Integration test: full pipeline scenario ---


class TestPortraitVideoIntegration:
    """Integration tests simulating the full portrait video scenario."""

    def test_wrist_estimated_portrait_launch_angle_is_reasonable(self):
        """In portrait video with wrist estimation, launch angle should be reasonable.

        This is the key scenario that was broken: portrait video (1080x1920)
        with wrist-estimated bat positions producing nonsensical angles like -149°.
        With aspect ratio correction, a slight upward swing should produce
        a small positive angle.
        """
        # Simulate wrist-estimated positions (normalized coords)
        # Bat moving slightly upward-right (typical swing)
        # In normalized coords: dx=0.05, dy=-0.01 (slight upward)
        detections = [
            _make_detection(9, position=(0.45, 0.51)),
            _make_detection(10, position=(0.50, 0.50)),
        ]
        trajectory = BatTrajectory(detections=detections)

        calculator = LaunchAngleCalculator()

        # Portrait video dimensions
        result = calculator.calculate_launch_angle(
            trajectory, impact_frame=10,
            video_width=1080, video_height=1920,
        )

        # Should be a small positive angle (slight upward swing)
        # dx_physical = 0.05 * 1080 = 54
        # dy_physical = -0.01 * 1920 = -19.2
        # angle = atan2(-(-19.2), 54) = atan2(19.2, 54) ≈ 19.6°
        expected = math.degrees(math.atan2(19.2, 54))
        assert abs(result.angle_degrees - expected) < 0.5
        # Most importantly: it's a reasonable angle, not -149°
        assert -30 < result.angle_degrees < 60

    def test_without_correction_portrait_gives_wrong_angle(self):
        """Without aspect ratio correction, portrait video gives distorted angles.

        This demonstrates the bug: without correction (video_width=1, video_height=1),
        the same movement gives a different (incorrect for portrait) angle.
        """
        detections = [
            _make_detection(9, position=(0.45, 0.51)),
            _make_detection(10, position=(0.50, 0.50)),
        ]
        trajectory = BatTrajectory(detections=detections)

        calculator = LaunchAngleCalculator()

        # Without correction (default)
        result_uncorrected = calculator.calculate_launch_angle(
            trajectory, impact_frame=10,
        )

        # With portrait correction
        result_corrected = calculator.calculate_launch_angle(
            trajectory, impact_frame=10,
            video_width=1080, video_height=1920,
        )

        # Uncorrected: atan2(-(-0.01), 0.05) = atan2(0.01, 0.05) ≈ 11.3°
        # Corrected: atan2(19.2, 54) ≈ 19.6°
        # The corrected angle accounts for the portrait aspect ratio
        assert result_corrected.angle_degrees > result_uncorrected.angle_degrees
