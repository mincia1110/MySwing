"""Unit tests for the wrist-based bat position estimator.

Tests estimation with valid wrist+elbow keypoints, fallback when only
wrist is available, trajectory building, orientation calculation,
and handling of empty/insufficient keypoints.
"""

import math

import pytest

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.pose import Keypoint, PoseResult
from app.pipeline.wrist_bat_estimator import WristBatEstimator


def _make_keypoint(
    name: str,
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
    confidence: float = 0.9,
) -> Keypoint:
    """Helper to create a Keypoint instance."""
    return Keypoint(x=x, y=y, z=z, confidence=confidence, name=name)


def _make_pose_result(
    frame_index: int,
    keypoints: list[Keypoint] | None = None,
    overall_confidence: float = 0.9,
) -> PoseResult:
    """Helper to create a PoseResult instance."""
    return PoseResult(
        frame_index=frame_index,
        keypoints=keypoints or [],
        person_id=0,
        is_primary_batter=True,
        overall_confidence=overall_confidence,
        is_low_confidence=False,
    )


class TestWristBatEstimatorInit:
    """Test WristBatEstimator initialization and validation."""

    def test_default_initialization(self):
        """Default parameters are set correctly."""
        estimator = WristBatEstimator()
        assert estimator.bat_length_normalized == 0.25
        assert estimator.dominant_hand == "right"
        assert estimator.min_keypoint_confidence == 0.3

    def test_custom_parameters(self):
        """Custom parameters are accepted."""
        estimator = WristBatEstimator(
            bat_length_normalized=0.3,
            dominant_hand="left",
            min_keypoint_confidence=0.5,
        )
        assert estimator.bat_length_normalized == 0.3
        assert estimator.dominant_hand == "left"
        assert estimator.min_keypoint_confidence == 0.5

    def test_invalid_bat_length_zero(self):
        """bat_length_normalized=0 raises ValueError."""
        with pytest.raises(ValueError, match="bat_length_normalized"):
            WristBatEstimator(bat_length_normalized=0.0)

    def test_invalid_bat_length_negative(self):
        """Negative bat_length_normalized raises ValueError."""
        with pytest.raises(ValueError, match="bat_length_normalized"):
            WristBatEstimator(bat_length_normalized=-0.1)

    def test_invalid_bat_length_too_large(self):
        """bat_length_normalized > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="bat_length_normalized"):
            WristBatEstimator(bat_length_normalized=1.5)

    def test_invalid_dominant_hand(self):
        """Invalid dominant_hand raises ValueError."""
        with pytest.raises(ValueError, match="dominant_hand"):
            WristBatEstimator(dominant_hand="both")


class TestEstimateFromPose:
    """Test estimate_from_pose with various keypoint configurations."""

    def test_with_wrist_and_elbow(self):
        """Estimation with both wrist and elbow produces valid result."""
        estimator = WristBatEstimator(bat_length_normalized=0.25)
        keypoints = [
            _make_keypoint("right_elbow", x=0.4, y=0.5),
            _make_keypoint("right_wrist", x=0.5, y=0.5),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(pose, frame_index=0)

        assert result.detected is True
        assert result.is_predicted is True
        assert result.coordinate_space == "normalized"
        assert result.bat_head_position is not None
        assert result.frame_index == 0
        assert result.confidence > 0.0

        # Bat center should be between wrist and bat head
        # Elbow at 0.4, wrist at 0.5 -> dx=0.1 rightward
        # Bat extends from the wrist in the forearm direction -> head at 0.75
        # Center at (0.5 + 0.75) / 2 = 0.625
        assert abs(result.position[0] - 0.625) < 0.01
        assert abs(result.position[1] - 0.5) < 0.01

    def test_orientation_horizontal_right(self):
        """Horizontal rightward direction gives ~0 degrees orientation."""
        estimator = WristBatEstimator(bat_length_normalized=0.25)
        keypoints = [
            _make_keypoint("right_elbow", x=0.3, y=0.5),
            _make_keypoint("right_wrist", x=0.5, y=0.5),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(pose, frame_index=0)

        # Direction is purely rightward → angle should be ~0 degrees
        assert abs(result.orientation_angle - 0.0) < 1.0

    def test_orientation_downward(self):
        """Downward direction gives ~90 degrees orientation."""
        estimator = WristBatEstimator(bat_length_normalized=0.25)
        keypoints = [
            _make_keypoint("right_elbow", x=0.5, y=0.3),
            _make_keypoint("right_wrist", x=0.5, y=0.5),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(pose, frame_index=0)

        # Direction is downward → angle should be ~90 degrees
        assert abs(result.orientation_angle - 90.0) < 1.0

    def test_orientation_upward_left(self):
        """Upward-left direction gives correct angle (180-270 range)."""
        estimator = WristBatEstimator(bat_length_normalized=0.25)
        keypoints = [
            _make_keypoint("right_elbow", x=0.6, y=0.6),
            _make_keypoint("right_wrist", x=0.4, y=0.4),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(pose, frame_index=0)

        # Direction is up-left → angle should be ~225 degrees (180+45)
        assert abs(result.orientation_angle - 225.0) < 1.0

    def test_wrist_only_fallback(self):
        """When only wrist is available, uses wrist position with lower confidence."""
        estimator = WristBatEstimator(bat_length_normalized=0.25)
        keypoints = [
            _make_keypoint("right_wrist", x=0.6, y=0.4, confidence=0.8),
        ]
        pose = _make_pose_result(frame_index=5, keypoints=keypoints)

        result = estimator.estimate_from_pose(pose, frame_index=5)

        assert result.detected is True
        assert result.is_predicted is True
        assert result.frame_index == 5
        # Position should be the wrist position
        assert abs(result.position[0] - 0.6) < 0.01
        assert abs(result.position[1] - 0.4) < 0.01
        # Orientation unknown
        assert result.orientation_angle == 0.0
        # Lower confidence than full estimation
        assert result.confidence == pytest.approx(0.8 * 0.4, abs=0.01)

    def test_no_keypoints_returns_no_detection(self):
        """Empty keypoints returns no-detection result."""
        estimator = WristBatEstimator()
        pose = _make_pose_result(frame_index=3, keypoints=[])

        result = estimator.estimate_from_pose(pose, frame_index=3)

        assert result.detected is False
        assert result.is_predicted is True
        assert result.frame_index == 3
        assert result.position == (0.0, 0.0)
        assert result.confidence == 0.0

    def test_low_confidence_keypoints_ignored(self):
        """Keypoints below min_confidence are treated as unavailable."""
        estimator = WristBatEstimator(min_keypoint_confidence=0.5)
        keypoints = [
            _make_keypoint("right_wrist", x=0.5, y=0.5, confidence=0.2),
            _make_keypoint("right_elbow", x=0.4, y=0.5, confidence=0.1),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(pose, frame_index=0)

        assert result.detected is False

    def test_left_handed_uses_left_wrist(self):
        """Left-handed estimator uses left_wrist and left_elbow."""
        estimator = WristBatEstimator(
            bat_length_normalized=0.25, dominant_hand="left"
        )
        keypoints = [
            _make_keypoint("left_elbow", x=0.6, y=0.5),
            _make_keypoint("left_wrist", x=0.5, y=0.5),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(pose, frame_index=0)

        assert result.detected is True
        assert result.is_predicted is True
        # Direction is leftward (elbow at 0.6, wrist at 0.5) -> dx=-0.1.
        # Bat extends from the wrist in the forearm direction.
        assert abs(result.position[0] - 0.375) < 0.01

    def test_position_uses_frame_aspect_ratio_for_bat_length(self):
        """Horizontal extension uses height-based bat length converted through width."""
        estimator = WristBatEstimator(bat_length_normalized=0.25)
        keypoints = [
            _make_keypoint("right_elbow", x=0.4, y=0.5),
            _make_keypoint("right_wrist", x=0.5, y=0.5),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(
            pose, frame_index=0, video_width=1920, video_height=1080
        )

        expected_head_x = 0.5 + (0.25 * 1080 / 1920)
        expected_center_x = (0.5 + expected_head_x) / 2
        assert result.bat_head_position is not None
        assert result.bat_head_position[0] == pytest.approx(expected_head_x)
        assert result.position[0] == pytest.approx(expected_center_x)

    def test_mirrored_left_handed_estimate_matches_after_unmirror(self):
        """A mirrored pose should produce the mirrored bat head with matching geometry."""
        estimator_right = WristBatEstimator(
            bat_length_normalized=0.25, dominant_hand="right"
        )
        estimator_left = WristBatEstimator(
            bat_length_normalized=0.25, dominant_hand="left"
        )
        right_pose = _make_pose_result(
            frame_index=0,
            keypoints=[
                _make_keypoint("right_elbow", x=0.4, y=0.45),
                _make_keypoint("right_wrist", x=0.5, y=0.50),
            ],
        )
        mirrored_left_pose = _make_pose_result(
            frame_index=0,
            keypoints=[
                _make_keypoint("left_elbow", x=0.6, y=0.45),
                _make_keypoint("left_wrist", x=0.5, y=0.50),
            ],
        )

        right = estimator_right.estimate_from_pose(
            right_pose, frame_index=0, video_width=1920, video_height=1080
        )
        mirrored = estimator_left.estimate_from_pose(
            mirrored_left_pose, frame_index=0, video_width=1920, video_height=1080
        )

        assert right.bat_head_position is not None
        assert mirrored.bat_head_position is not None
        unmirrored_head = (1.0 - mirrored.bat_head_position[0], mirrored.bat_head_position[1])
        unmirrored_center = (1.0 - mirrored.position[0], mirrored.position[1])
        assert unmirrored_head == pytest.approx(right.bat_head_position)
        assert unmirrored_center == pytest.approx(right.position)

    def test_falls_back_to_non_dominant_hand(self):
        """Falls back to non-dominant hand when dominant wrist unavailable."""
        estimator = WristBatEstimator(dominant_hand="right")
        keypoints = [
            # No right_wrist, but left_wrist available
            _make_keypoint("left_elbow", x=0.4, y=0.5),
            _make_keypoint("left_wrist", x=0.5, y=0.5),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(pose, frame_index=0)

        assert result.detected is True
        assert result.is_predicted is True

    def test_none_pose_result_returns_no_detection(self):
        """None pose_result returns no-detection."""
        estimator = WristBatEstimator()

        result = estimator.estimate_from_pose(None, frame_index=0)

        assert result.detected is False

    def test_confidence_calculation_with_direction(self):
        """Confidence is min(wrist, elbow) * 0.7 when using direction."""
        estimator = WristBatEstimator()
        keypoints = [
            _make_keypoint("right_elbow", x=0.4, y=0.5, confidence=0.8),
            _make_keypoint("right_wrist", x=0.5, y=0.5, confidence=0.9),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(pose, frame_index=0)

        # min(0.8, 0.9) * 0.7 = 0.56
        assert result.confidence == pytest.approx(0.56, abs=0.01)


class TestEstimateTrajectory:
    """Test trajectory building from pose sequences."""

    def test_empty_sequence(self):
        """Empty pose sequence produces empty trajectory."""
        estimator = WristBatEstimator()
        trajectory = estimator.estimate_trajectory([])

        assert trajectory.detections == []
        assert trajectory.bat_speed_pixels_per_frame == []
        assert trajectory.tracking_accuracy == 0.0
        assert trajectory.tracking_failures == []

    def test_single_frame(self):
        """Single frame produces trajectory with one detection and no speeds."""
        estimator = WristBatEstimator()
        keypoints = [
            _make_keypoint("right_elbow", x=0.4, y=0.5),
            _make_keypoint("right_wrist", x=0.5, y=0.5),
        ]
        pose_sequence = [_make_pose_result(frame_index=0, keypoints=keypoints)]

        trajectory = estimator.estimate_trajectory(pose_sequence)

        assert len(trajectory.detections) == 1
        assert trajectory.detections[0].detected is True
        assert trajectory.bat_speed_pixels_per_frame == []
        assert trajectory.tracking_accuracy == 1.0

    def test_multiple_frames_with_movement(self):
        """Multiple frames produce trajectory with speeds calculated."""
        estimator = WristBatEstimator(bat_length_normalized=0.25)
        pose_sequence = []

        # Simulate wrist moving rightward across frames
        for i in range(5):
            keypoints = [
                _make_keypoint("right_elbow", x=0.3 + i * 0.02, y=0.5),
                _make_keypoint("right_wrist", x=0.4 + i * 0.02, y=0.5),
            ]
            pose_sequence.append(_make_pose_result(frame_index=i, keypoints=keypoints))

        trajectory = estimator.estimate_trajectory(pose_sequence)

        assert len(trajectory.detections) == 5
        assert len(trajectory.bat_speed_pixels_per_frame) == 4
        assert trajectory.tracking_accuracy == 1.0
        assert trajectory.tracking_failures == []
        # All speeds should be positive (bat is moving)
        for speed in trajectory.bat_speed_pixels_per_frame:
            assert speed > 0.0

    def test_mixed_detection_and_failure(self):
        """Frames with missing keypoints produce tracking failures."""
        estimator = WristBatEstimator()
        pose_sequence = [
            # Frame 0: good keypoints
            _make_pose_result(
                frame_index=0,
                keypoints=[
                    _make_keypoint("right_elbow", x=0.4, y=0.5),
                    _make_keypoint("right_wrist", x=0.5, y=0.5),
                ],
            ),
            # Frame 1: no keypoints
            _make_pose_result(frame_index=1, keypoints=[]),
            # Frame 2: no keypoints
            _make_pose_result(frame_index=2, keypoints=[]),
            # Frame 3: good keypoints
            _make_pose_result(
                frame_index=3,
                keypoints=[
                    _make_keypoint("right_elbow", x=0.5, y=0.5),
                    _make_keypoint("right_wrist", x=0.6, y=0.5),
                ],
            ),
        ]

        trajectory = estimator.estimate_trajectory(pose_sequence)

        assert len(trajectory.detections) == 4
        assert trajectory.detections[0].detected is True
        assert trajectory.detections[1].detected is False
        assert trajectory.detections[2].detected is False
        assert trajectory.detections[3].detected is True
        # Tracking accuracy: 2/4 = 0.5
        assert trajectory.tracking_accuracy == pytest.approx(0.5)
        # Tracking failure from frame 1 to 2
        assert (1, 2) in trajectory.tracking_failures

    def test_all_frames_failed(self):
        """All frames with no keypoints gives 0% accuracy."""
        estimator = WristBatEstimator()
        pose_sequence = [
            _make_pose_result(frame_index=0, keypoints=[]),
            _make_pose_result(frame_index=1, keypoints=[]),
            _make_pose_result(frame_index=2, keypoints=[]),
        ]

        trajectory = estimator.estimate_trajectory(pose_sequence)

        assert len(trajectory.detections) == 3
        assert trajectory.tracking_accuracy == 0.0
        assert all(not d.detected for d in trajectory.detections)

    def test_all_results_marked_as_predicted(self):
        """All results from wrist estimation are marked is_predicted=True."""
        estimator = WristBatEstimator()
        pose_sequence = [
            _make_pose_result(
                frame_index=i,
                keypoints=[
                    _make_keypoint("right_elbow", x=0.4, y=0.5),
                    _make_keypoint("right_wrist", x=0.5, y=0.5),
                ],
            )
            for i in range(3)
        ]

        trajectory = estimator.estimate_trajectory(pose_sequence)

        for detection in trajectory.detections:
            assert detection.is_predicted is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_coincident_wrist_and_elbow(self):
        """When wrist and elbow are at same position, falls back to wrist-only."""
        estimator = WristBatEstimator()
        keypoints = [
            _make_keypoint("right_elbow", x=0.5, y=0.5),
            _make_keypoint("right_wrist", x=0.5, y=0.5),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(pose, frame_index=0)

        # Should fall back to wrist-only (direction magnitude is 0)
        assert result.detected is True
        assert result.position == (0.5, 0.5)

    def test_bat_length_at_boundary(self):
        """bat_length_normalized=1.0 is valid (full frame height)."""
        estimator = WristBatEstimator(bat_length_normalized=1.0)
        keypoints = [
            _make_keypoint("right_elbow", x=0.4, y=0.5),
            _make_keypoint("right_wrist", x=0.5, y=0.5),
        ]
        pose = _make_pose_result(frame_index=0, keypoints=keypoints)

        result = estimator.estimate_from_pose(pose, frame_index=0)

        assert result.detected is True
        # Elbow at 0.4, wrist at 0.5, bat_length=1.0.
        # Bat extends in the elbow->wrist direction -> head at 1.5.
        assert abs(result.position[0] - 1.0) < 0.01

    def test_speed_calculation_with_undetected_frames(self):
        """Speed is 0 when either frame has no detection."""
        estimator = WristBatEstimator()
        det1 = BatDetectionResult(
            frame_index=0, detected=True, position=(0.5, 0.5),
            orientation_angle=0.0, length_pixels=0.25,
            confidence=0.5, is_predicted=True,
        )
        det2 = BatDetectionResult(
            frame_index=1, detected=False, position=(0.0, 0.0),
            orientation_angle=0.0, length_pixels=0.0,
            confidence=0.0, is_predicted=True,
        )

        speed = WristBatEstimator._calculate_speed(det1, det2)
        assert speed == 0.0

    def test_speed_prefers_bat_head_position(self):
        """Trajectory speed uses the barrel/head point when it is available."""
        det1 = BatDetectionResult(
            frame_index=0, detected=True, position=(0.5, 0.5),
            orientation_angle=0.0, length_pixels=0.25,
            confidence=0.5, is_predicted=True,
            bat_head_position=(0.6, 0.5),
        )
        det2 = BatDetectionResult(
            frame_index=2, detected=True, position=(0.5, 0.5),
            orientation_angle=0.0, length_pixels=0.25,
            confidence=0.5, is_predicted=True,
            bat_head_position=(0.8, 0.5),
        )

        speed = WristBatEstimator._calculate_speed(det1, det2)

        assert speed == pytest.approx(0.1)
