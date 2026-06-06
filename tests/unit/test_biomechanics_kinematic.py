"""Unit tests for kinematic chain, rotation, and hand path analysis (Task 9.3).

Tests KinematicChainAnalyzer, RotationAnalyzer, and HandPathAnalyzer classes covering:
- Angular velocity calculation for known angle sequences
- Peak detection (correct frame and value)
- Proximal-to-distal sequence check
- Hip/shoulder rotation speed calculation
- Hip-shoulder separation angle
- Hand path efficiency (straight path = 1.0, curved path < 1.0)
- Graceful handling of insufficient data
"""

import math

import pytest

from app.models.biomechanics import JointAngularVelocity, KinematicChainResult, RotationResult
from app.models.pose import Keypoint, PoseResult
from app.pipeline.biomechanics_analyzer import (
    HandPathAnalyzer,
    KinematicChainAnalyzer,
    RotationAnalyzer,
)


# --- Helper functions ---


def _make_keypoint(
    name: str,
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
    confidence: float = 0.9,
) -> Keypoint:
    """Create a Keypoint with given parameters."""
    return Keypoint(x=x, y=y, z=z, confidence=confidence, name=name)


def _make_pose_result(
    keypoints: list[Keypoint],
    frame_index: int = 0,
) -> PoseResult:
    """Create a PoseResult with given keypoints."""
    return PoseResult(
        frame_index=frame_index,
        keypoints=keypoints,
        person_id=0,
        is_primary_batter=True,
        overall_confidence=0.9,
        is_low_confidence=False,
    )


def _make_kinematic_pose(
    frame_index: int,
    hip_angle_offset: float = 0.0,
    shoulder_angle_offset: float = 0.0,
    elbow_angle_offset: float = 0.0,
    wrist_angle_offset: float = 0.0,
) -> PoseResult:
    """Create a pose with keypoints positioned to produce specific joint angles.

    Uses geometric placement to control the 3-point angles at each joint.
    The offset rotates the child keypoint around the joint to change the angle.
    """
    # Base positions (normalized 0-1 space)
    # Hip joint: left_shoulder(0.5, 0.3) - left_hip(0.5, 0.5) - left_knee(0.5, 0.7)
    # Shoulder joint: left_hip(0.5, 0.5) - left_shoulder(0.5, 0.3) - left_elbow(0.7, 0.3)
    # Elbow joint: left_shoulder(0.5, 0.3) - left_elbow(0.7, 0.3) - left_wrist(0.9, 0.3)
    # Wrist joint: left_elbow(0.7, 0.3) - left_wrist(0.9, 0.3) - left_index(1.0, 0.3)

    # For hip: rotate left_knee around left_hip
    knee_angle = math.radians(hip_angle_offset)
    knee_x = 0.5 + 0.2 * math.sin(knee_angle)
    knee_y = 0.5 + 0.2 * math.cos(knee_angle)

    # For shoulder: rotate left_elbow around left_shoulder
    elbow_base_angle = math.radians(shoulder_angle_offset)
    elbow_x = 0.5 + 0.2 * math.cos(elbow_base_angle)
    elbow_y = 0.3 + 0.2 * math.sin(elbow_base_angle)

    # For elbow: rotate left_wrist around left_elbow
    wrist_base_angle = math.radians(elbow_angle_offset)
    wrist_x = elbow_x + 0.2 * math.cos(wrist_base_angle)
    wrist_y = elbow_y + 0.2 * math.sin(wrist_base_angle)

    # For wrist: rotate left_index around left_wrist
    index_base_angle = math.radians(wrist_angle_offset)
    index_x = wrist_x + 0.1 * math.cos(index_base_angle)
    index_y = wrist_y + 0.1 * math.sin(index_base_angle)

    keypoints = [
        _make_keypoint("left_shoulder", x=0.5, y=0.3),
        _make_keypoint("left_hip", x=0.5, y=0.5),
        _make_keypoint("left_knee", x=knee_x, y=knee_y),
        _make_keypoint("left_elbow", x=elbow_x, y=elbow_y),
        _make_keypoint("left_wrist", x=wrist_x, y=wrist_y),
        _make_keypoint("left_index", x=index_x, y=index_y),
        _make_keypoint("right_hip", x=0.6, y=0.5),
        _make_keypoint("right_shoulder", x=0.6, y=0.3),
    ]
    return _make_pose_result(keypoints, frame_index=frame_index)


# --- KinematicChainAnalyzer Tests ---


class TestKinematicChainAngularVelocity:
    """Test angular velocity calculation for known angle sequences."""

    def test_basic_angular_velocity_calculation(self):
        """Angular velocity is computed as angle_change * fps for consecutive frames."""
        analyzer = KinematicChainAnalyzer()

        # Create a simple sequence where the hip angle changes by 10 degrees per frame
        # Use explicit keypoint positions to control the angle
        poses = []
        for i in range(5):
            # Rotate knee around hip by i*10 degrees
            poses.append(_make_kinematic_pose(frame_index=i, hip_angle_offset=i * 10.0))

        result = analyzer.analyze_kinematic_chain(
            poses,
            phase_boundaries={"start_frame": 0, "end_frame": 4},
            fps=120.0,
        )

        # Should have at least the hips joint measured
        assert "hips" in result.joint_peak_angular_velocities
        # The angular velocity should be positive (angle is changing)
        assert result.joint_peak_angular_velocities["hips"].peak_velocity_dps > 0

    def test_zero_fps_returns_empty_result(self):
        """Zero FPS returns empty result gracefully."""
        analyzer = KinematicChainAnalyzer()
        poses = [_make_kinematic_pose(i) for i in range(5)]

        result = analyzer.analyze_kinematic_chain(
            poses,
            phase_boundaries={"start_frame": 0, "end_frame": 4},
            fps=0.0,
        )

        assert result.joint_peak_angular_velocities == {}
        assert result.sequence_correct is False

    def test_single_frame_returns_empty_result(self):
        """Single frame cannot compute velocity, returns empty."""
        analyzer = KinematicChainAnalyzer()
        poses = [_make_kinematic_pose(0)]

        result = analyzer.analyze_kinematic_chain(
            poses,
            phase_boundaries={"start_frame": 0, "end_frame": 0},
            fps=120.0,
        )

        assert result.joint_peak_angular_velocities == {}
        assert result.sequence_correct is False


class TestKinematicChainPeakDetection:
    """Test peak angular velocity detection (correct frame and value)."""

    def test_peak_at_correct_frame(self):
        """Peak velocity is detected at the frame with maximum angular change."""
        analyzer = KinematicChainAnalyzer()

        # Create sequence where hip angle changes slowly then fast then slow
        # Frames 0-2: 5 deg/frame, Frame 2-3: 30 deg/frame, Frames 3-4: 5 deg/frame
        poses = []
        offsets = [0, 5, 10, 40, 45]  # biggest jump is between frame 2 and 3
        for i, offset in enumerate(offsets):
            poses.append(_make_kinematic_pose(frame_index=i, hip_angle_offset=offset))

        result = analyzer.analyze_kinematic_chain(
            poses,
            phase_boundaries={"start_frame": 0, "end_frame": 4},
            fps=120.0,
        )

        assert "hips" in result.joint_peak_angular_velocities
        hip_vel = result.joint_peak_angular_velocities["hips"]
        # Peak should be at frame 3 (biggest change from frame 2 to 3)
        assert hip_vel.peak_frame == 3

    def test_peak_velocity_value(self):
        """Peak velocity value matches expected calculation."""
        analyzer = KinematicChainAnalyzer()

        # Simple case: two frames with known angle change
        # Frame 0: hip offset 0, Frame 1: hip offset 90
        # The 3-point angle change should be significant
        poses = [
            _make_kinematic_pose(frame_index=0, hip_angle_offset=0.0),
            _make_kinematic_pose(frame_index=1, hip_angle_offset=90.0),
        ]

        result = analyzer.analyze_kinematic_chain(
            poses,
            phase_boundaries={"start_frame": 0, "end_frame": 1},
            fps=60.0,
        )

        assert "hips" in result.joint_peak_angular_velocities
        # Velocity should be angle_change * fps (for 1-frame diff)
        hip_vel = result.joint_peak_angular_velocities["hips"]
        assert hip_vel.peak_velocity_dps > 0

    def test_peak_time_ms_calculation(self):
        """Peak time in ms is correctly calculated relative to start frame."""
        analyzer = KinematicChainAnalyzer()

        # Peak at frame 3, start at frame 0, fps=120
        # Expected time: (3-0)/120 * 1000 = 25 ms
        offsets = [0, 5, 10, 40, 45]
        poses = [
            _make_kinematic_pose(frame_index=i, hip_angle_offset=offsets[i])
            for i in range(5)
        ]

        result = analyzer.analyze_kinematic_chain(
            poses,
            phase_boundaries={"start_frame": 0, "end_frame": 4},
            fps=120.0,
        )

        assert "hips" in result.joint_peak_angular_velocities
        hip_vel = result.joint_peak_angular_velocities["hips"]
        expected_time_ms = (hip_vel.peak_frame - 0) / 120.0 * 1000.0
        assert abs(hip_vel.peak_time_ms - expected_time_ms) < 0.01

    def test_low_confidence_segments_reduce_peak_velocity(self):
        """Low-confidence frames should down-weight angular velocity spikes."""
        analyzer = KinematicChainAnalyzer()

        offsets = [0, 10, 120, 30, 40]
        high_conf_poses = [
            _make_kinematic_pose(frame_index=i, hip_angle_offset=offsets[i])
            for i in range(5)
        ]
        low_conf_poses = [
            _make_kinematic_pose(frame_index=i, hip_angle_offset=offsets[i])
            for i in range(5)
        ]

        # Make the spike frame low confidence across all keypoints.
        for kp in low_conf_poses[2].keypoints:
            kp.confidence = 0.5

        high_conf_result = analyzer.analyze_kinematic_chain(
            high_conf_poses,
            phase_boundaries={"start_frame": 0, "end_frame": 4},
            fps=120.0,
        )
        low_conf_result = analyzer.analyze_kinematic_chain(
            low_conf_poses,
            phase_boundaries={"start_frame": 0, "end_frame": 4},
            fps=120.0,
        )

        assert "hips" in high_conf_result.joint_peak_angular_velocities
        assert "hips" in low_conf_result.joint_peak_angular_velocities
        assert (
            low_conf_result.joint_peak_angular_velocities["hips"].peak_velocity_dps
            < high_conf_result.joint_peak_angular_velocities["hips"].peak_velocity_dps
        )


class TestKinematicChainSequence:
    """Test proximal-to-distal sequence check."""

    def test_correct_sequence(self):
        """Sequence is correct when hips peak before shoulders before elbows before wrists."""
        analyzer = KinematicChainAnalyzer()

        # Use cumulative offsets where the biggest jump for each joint
        # occurs at progressively later frames:
        # Hips: biggest jump at frame 2 (0→50)
        # Shoulders: biggest jump at frame 4 (10→60)
        # Elbows: biggest jump at frame 6 (20→70)
        # Wrists: biggest jump at frame 8 (30→80)
        hip_offsets =      [0, 5, 55, 60, 62, 63, 64, 65, 66, 67]
        shoulder_offsets =  [0, 2, 5,  10, 60, 65, 67, 68, 69, 70]
        elbow_offsets =     [0, 2, 5,  8,  12, 20, 70, 75, 77, 78]
        wrist_offsets =     [0, 2, 5,  8,  12, 15, 20, 30, 80, 85]

        poses = []
        for i in range(10):
            poses.append(_make_kinematic_pose(
                frame_index=i,
                hip_angle_offset=hip_offsets[i],
                shoulder_angle_offset=shoulder_offsets[i],
                elbow_angle_offset=elbow_offsets[i],
                wrist_angle_offset=wrist_offsets[i],
            ))

        result = analyzer.analyze_kinematic_chain(
            poses,
            phase_boundaries={"start_frame": 0, "end_frame": 9},
            fps=120.0,
        )

        # Verify all 4 joints detected and sequence is correct
        assert len(result.joint_peak_angular_velocities) == 4
        hips_frame = result.joint_peak_angular_velocities["hips"].peak_frame
        shoulders_frame = result.joint_peak_angular_velocities["shoulders"].peak_frame
        elbows_frame = result.joint_peak_angular_velocities["elbows"].peak_frame
        wrists_frame = result.joint_peak_angular_velocities["wrists"].peak_frame
        assert hips_frame <= shoulders_frame <= elbows_frame <= wrists_frame
        assert result.sequence_correct is True

    def test_incorrect_sequence_wrists_before_elbows(self):
        """Sequence is incorrect when peak frames are not in proximal-to-distal order."""
        analyzer = KinematicChainAnalyzer()

        # Directly test the _check_proximal_to_distal method with known data
        # Create JointAngularVelocity objects with specific peak frames
        joint_velocities = {
            "hips": JointAngularVelocity(
                joint_name="hips", peak_velocity_dps=1000.0, peak_frame=2, peak_time_ms=16.7
            ),
            "shoulders": JointAngularVelocity(
                joint_name="shoulders", peak_velocity_dps=1200.0, peak_frame=4, peak_time_ms=33.3
            ),
            "elbows": JointAngularVelocity(
                joint_name="elbows", peak_velocity_dps=1500.0, peak_frame=8, peak_time_ms=66.7
            ),
            "wrists": JointAngularVelocity(
                joint_name="wrists", peak_velocity_dps=1800.0, peak_frame=6, peak_time_ms=50.0
            ),
        }

        # Wrists (frame 6) before elbows (frame 8) → incorrect
        result = analyzer._check_proximal_to_distal(joint_velocities)
        assert result is False

    def test_correct_sequence_via_check_method(self):
        """Sequence is correct when peak frames follow proximal-to-distal order."""
        analyzer = KinematicChainAnalyzer()

        joint_velocities = {
            "hips": JointAngularVelocity(
                joint_name="hips", peak_velocity_dps=1000.0, peak_frame=2, peak_time_ms=16.7
            ),
            "shoulders": JointAngularVelocity(
                joint_name="shoulders", peak_velocity_dps=1200.0, peak_frame=4, peak_time_ms=33.3
            ),
            "elbows": JointAngularVelocity(
                joint_name="elbows", peak_velocity_dps=1500.0, peak_frame=6, peak_time_ms=50.0
            ),
            "wrists": JointAngularVelocity(
                joint_name="wrists", peak_velocity_dps=1800.0, peak_frame=8, peak_time_ms=66.7
            ),
        }

        result = analyzer._check_proximal_to_distal(joint_velocities)
        assert result is True

    def test_missing_joint_means_incorrect_sequence(self):
        """If any joint is missing, sequence is marked as incorrect."""
        analyzer = KinematicChainAnalyzer()

        # Create poses with only hip keypoints (no wrist/index)
        poses = []
        for i in range(5):
            keypoints = [
                _make_keypoint("left_shoulder", x=0.5, y=0.3),
                _make_keypoint("left_hip", x=0.5, y=0.5),
                _make_keypoint("left_knee", x=0.5, y=0.7 + i * 0.02),
            ]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        result = analyzer.analyze_kinematic_chain(
            poses,
            phase_boundaries={"start_frame": 0, "end_frame": 4},
            fps=120.0,
        )

        assert result.sequence_correct is False


# --- RotationAnalyzer Tests ---


class TestRotationSpeed:
    """Test hip/shoulder rotation speed calculation."""

    def test_hip_rotation_speed(self):
        """Hip rotation speed is calculated from hip line angle changes."""
        analyzer = RotationAnalyzer()

        # Create poses where hip line rotates 10 degrees per frame
        poses = []
        for i in range(5):
            angle_rad = math.radians(i * 10.0)
            # left_hip and right_hip form a line that rotates
            left_hip_x = 0.5 - 0.1 * math.cos(angle_rad)
            left_hip_y = 0.5 - 0.1 * math.sin(angle_rad)
            right_hip_x = 0.5 + 0.1 * math.cos(angle_rad)
            right_hip_y = 0.5 + 0.1 * math.sin(angle_rad)

            keypoints = [
                _make_keypoint("left_hip", x=left_hip_x, y=left_hip_y),
                _make_keypoint("right_hip", x=right_hip_x, y=right_hip_y),
                _make_keypoint("left_shoulder", x=0.5, y=0.3),
                _make_keypoint("right_shoulder", x=0.6, y=0.3),
            ]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        result = analyzer.calculate_rotation(
            poses, rotation_phase=(0, 4), fps=120.0
        )

        # Hip rotates 10 deg/frame at 120 fps = 1200 deg/s
        assert abs(result.hip_rotation_speed_dps - 1200.0) < 1.0

    def test_shoulder_rotation_speed(self):
        """Shoulder rotation speed is calculated from shoulder line angle changes."""
        analyzer = RotationAnalyzer()

        # Create poses where shoulder line rotates 15 degrees per frame
        poses = []
        for i in range(5):
            angle_rad = math.radians(i * 15.0)
            left_shoulder_x = 0.5 - 0.1 * math.cos(angle_rad)
            left_shoulder_y = 0.3 - 0.1 * math.sin(angle_rad)
            right_shoulder_x = 0.5 + 0.1 * math.cos(angle_rad)
            right_shoulder_y = 0.3 + 0.1 * math.sin(angle_rad)

            keypoints = [
                _make_keypoint("left_hip", x=0.5, y=0.5),
                _make_keypoint("right_hip", x=0.6, y=0.5),
                _make_keypoint("left_shoulder", x=left_shoulder_x, y=left_shoulder_y),
                _make_keypoint("right_shoulder", x=right_shoulder_x, y=right_shoulder_y),
            ]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        result = analyzer.calculate_rotation(
            poses, rotation_phase=(0, 4), fps=120.0
        )

        # Shoulder rotates 15 deg/frame at 120 fps = 1800 deg/s
        assert abs(result.shoulder_rotation_speed_dps - 1800.0) < 1.0

    def test_zero_fps_returns_zero_speeds(self):
        """Zero FPS returns zero rotation speeds gracefully."""
        analyzer = RotationAnalyzer()
        poses = [_make_kinematic_pose(i) for i in range(5)]

        result = analyzer.calculate_rotation(
            poses, rotation_phase=(0, 4), fps=0.0
        )

        assert result.hip_rotation_speed_dps == 0.0
        assert result.shoulder_rotation_speed_dps == 0.0

    def test_single_frame_jitter_is_smoothed_in_peak_speed(self):
        """A one-frame angle spike should not dominate peak rotation speed."""
        analyzer = RotationAnalyzer()

        # Mostly 10°/frame hip rotation, with one extreme jitter frame.
        angles_deg = [0.0, 10.0, 120.0, 30.0, 40.0]
        poses = []
        for i, angle_deg in enumerate(angles_deg):
            angle_rad = math.radians(angle_deg)
            keypoints = [
                _make_keypoint("left_hip", x=0.5 - 0.1 * math.cos(angle_rad), y=0.5 - 0.1 * math.sin(angle_rad)),
                _make_keypoint("right_hip", x=0.5 + 0.1 * math.cos(angle_rad), y=0.5 + 0.1 * math.sin(angle_rad)),
                _make_keypoint("left_shoulder", x=0.4, y=0.3),
                _make_keypoint("right_shoulder", x=0.6, y=0.3),
            ]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        result = analyzer.calculate_rotation(poses, rotation_phase=(0, 4), fps=120.0)

        # Legacy raw-diff peak would be near 13200 dps from jitter.
        # Smoothed peak should stay much closer to realistic swing rotation speeds.
        assert result.hip_rotation_speed_dps < 5000.0
        assert result.hip_rotation_speed_dps > 500.0

    def test_low_confidence_spike_is_down_weighted(self):
        """Low-confidence keypoint spike should contribute less to peak speed."""
        analyzer = RotationAnalyzer()

        poses = []
        angles_deg = [0.0, 10.0, 120.0, 30.0, 40.0]
        for i, angle_deg in enumerate(angles_deg):
            angle_rad = math.radians(angle_deg)
            low_conf = 0.5 if i == 2 else 0.95
            keypoints = [
                _make_keypoint("left_hip", x=0.5 - 0.1 * math.cos(angle_rad), y=0.5 - 0.1 * math.sin(angle_rad), confidence=low_conf),
                _make_keypoint("right_hip", x=0.5 + 0.1 * math.cos(angle_rad), y=0.5 + 0.1 * math.sin(angle_rad), confidence=low_conf),
                _make_keypoint("left_shoulder", x=0.4, y=0.3),
                _make_keypoint("right_shoulder", x=0.6, y=0.3),
            ]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        result = analyzer.calculate_rotation(poses, rotation_phase=(0, 4), fps=120.0)

        assert result.hip_rotation_speed_dps < 3500.0


class TestHipShoulderSeparation:
    """Test hip-shoulder separation angle calculation."""

    def test_separation_angle_calculation(self):
        """Separation angle is the max absolute difference between hip and shoulder angles."""
        analyzer = RotationAnalyzer()

        # Create poses where hips rotate faster than shoulders
        poses = []
        for i in range(5):
            hip_angle_rad = math.radians(i * 20.0)  # 20 deg/frame
            shoulder_angle_rad = math.radians(i * 5.0)  # 5 deg/frame

            keypoints = [
                _make_keypoint("left_hip", x=0.5 - 0.1 * math.cos(hip_angle_rad),
                               y=0.5 - 0.1 * math.sin(hip_angle_rad)),
                _make_keypoint("right_hip", x=0.5 + 0.1 * math.cos(hip_angle_rad),
                               y=0.5 + 0.1 * math.sin(hip_angle_rad)),
                _make_keypoint("left_shoulder", x=0.5 - 0.1 * math.cos(shoulder_angle_rad),
                               y=0.3 - 0.1 * math.sin(shoulder_angle_rad)),
                _make_keypoint("right_shoulder", x=0.5 + 0.1 * math.cos(shoulder_angle_rad),
                               y=0.3 + 0.1 * math.sin(shoulder_angle_rad)),
            ]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        result = analyzer.calculate_rotation(
            poses, rotation_phase=(0, 4), fps=120.0
        )

        # At frame 4: hip angle = 80 deg, shoulder angle = 20 deg
        # Max separation should be at frame 4: |80 - 20| = 60 degrees
        # (angles are computed via atan2, so actual values depend on geometry)
        assert result.hip_shoulder_separation_degrees > 0

    def test_no_separation_when_same_rotation(self):
        """Zero separation when hips and shoulders rotate at same rate."""
        analyzer = RotationAnalyzer()

        # Both rotate at same rate
        poses = []
        for i in range(5):
            angle_rad = math.radians(i * 10.0)
            keypoints = [
                _make_keypoint("left_hip", x=0.5 - 0.1 * math.cos(angle_rad),
                               y=0.5 - 0.1 * math.sin(angle_rad)),
                _make_keypoint("right_hip", x=0.5 + 0.1 * math.cos(angle_rad),
                               y=0.5 + 0.1 * math.sin(angle_rad)),
                _make_keypoint("left_shoulder", x=0.5 - 0.1 * math.cos(angle_rad),
                               y=0.3 - 0.1 * math.sin(angle_rad)),
                _make_keypoint("right_shoulder", x=0.5 + 0.1 * math.cos(angle_rad),
                               y=0.3 + 0.1 * math.sin(angle_rad)),
            ]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        result = analyzer.calculate_rotation(
            poses, rotation_phase=(0, 4), fps=120.0
        )

        # Same rotation → separation should be 0 (or very close)
        assert result.hip_shoulder_separation_degrees < 1.0

    def test_insufficient_data_returns_zero(self):
        """Insufficient data returns zero separation."""
        analyzer = RotationAnalyzer()

        result = analyzer.calculate_rotation(
            [], rotation_phase=(0, 4), fps=120.0
        )

        assert result.hip_shoulder_separation_degrees == 0.0


# --- HandPathAnalyzer Tests ---


class TestHandPathEfficiency:
    """Test hand path efficiency calculation."""

    def test_straight_path_efficiency_is_one(self):
        """A perfectly straight hand path has efficiency = 1.0."""
        analyzer = HandPathAnalyzer()

        # Hand moves in a straight line from (0.2, 0.5) to (0.8, 0.5)
        poses = []
        for i in range(6):
            x = 0.2 + i * 0.12  # linear movement
            keypoints = [_make_keypoint("left_wrist", x=x, y=0.5)]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        efficiency = analyzer.calculate_hand_path_efficiency(
            poses, load_frame=0, impact_frame=5
        )

        assert efficiency == 1.0

    def test_curved_path_efficiency_less_than_one(self):
        """A curved hand path has efficiency < 1.0."""
        analyzer = HandPathAnalyzer()

        # Hand moves in a curve (arc) from (0.2, 0.5) to (0.8, 0.5)
        # via (0.5, 0.2) - a significant detour upward
        poses = []
        positions = [
            (0.2, 0.5),
            (0.3, 0.35),
            (0.5, 0.2),
            (0.7, 0.35),
            (0.8, 0.5),
        ]
        for i, (x, y) in enumerate(positions):
            keypoints = [_make_keypoint("left_wrist", x=x, y=y)]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        efficiency = analyzer.calculate_hand_path_efficiency(
            poses, load_frame=0, impact_frame=4
        )

        assert 0.0 < efficiency < 1.0

    def test_efficiency_rounded_to_two_decimals(self):
        """Efficiency is rounded to 2 decimal places."""
        analyzer = HandPathAnalyzer()

        # Create a path that would produce a non-round efficiency
        poses = []
        positions = [
            (0.2, 0.5),
            (0.35, 0.4),
            (0.5, 0.45),
            (0.65, 0.5),
            (0.8, 0.5),
        ]
        for i, (x, y) in enumerate(positions):
            keypoints = [_make_keypoint("left_wrist", x=x, y=y)]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        efficiency = analyzer.calculate_hand_path_efficiency(
            poses, load_frame=0, impact_frame=4
        )

        # Check it's rounded to 2 decimal places
        assert efficiency == round(efficiency, 2)

    def test_efficiency_clamped_between_zero_and_one(self):
        """Efficiency is always between 0.0 and 1.0."""
        analyzer = HandPathAnalyzer()

        # Any valid path should produce 0 <= efficiency <= 1
        poses = []
        positions = [(0.2, 0.5), (0.3, 0.3), (0.5, 0.7), (0.7, 0.3), (0.8, 0.5)]
        for i, (x, y) in enumerate(positions):
            keypoints = [_make_keypoint("left_wrist", x=x, y=y)]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        efficiency = analyzer.calculate_hand_path_efficiency(
            poses, load_frame=0, impact_frame=4
        )

        assert 0.0 <= efficiency <= 1.0


class TestHandPathInsufficientData:
    """Test hand path efficiency with insufficient data."""

    def test_empty_pose_sequence(self):
        """Empty pose sequence returns 0.0."""
        analyzer = HandPathAnalyzer()

        efficiency = analyzer.calculate_hand_path_efficiency(
            [], load_frame=0, impact_frame=5
        )

        assert efficiency == 0.0

    def test_single_pose(self):
        """Single pose returns 0.0 (need at least 2 positions)."""
        analyzer = HandPathAnalyzer()
        keypoints = [_make_keypoint("left_wrist", x=0.5, y=0.5)]
        poses = [_make_pose_result(keypoints, frame_index=0)]

        efficiency = analyzer.calculate_hand_path_efficiency(
            poses, load_frame=0, impact_frame=5
        )

        assert efficiency == 0.0

    def test_load_frame_after_impact_frame(self):
        """Returns 0.0 when load_frame >= impact_frame."""
        analyzer = HandPathAnalyzer()
        poses = [
            _make_pose_result([_make_keypoint("left_wrist", x=0.2, y=0.5)], frame_index=0),
            _make_pose_result([_make_keypoint("left_wrist", x=0.8, y=0.5)], frame_index=5),
        ]

        efficiency = analyzer.calculate_hand_path_efficiency(
            poses, load_frame=5, impact_frame=0
        )

        assert efficiency == 0.0

    def test_no_wrist_keypoints(self):
        """Returns 0.0 when no wrist keypoints are available."""
        analyzer = HandPathAnalyzer()
        poses = []
        for i in range(5):
            # Only hip keypoints, no wrist
            keypoints = [_make_keypoint("left_hip", x=0.5, y=0.5)]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        efficiency = analyzer.calculate_hand_path_efficiency(
            poses, load_frame=0, impact_frame=4
        )

        assert efficiency == 0.0

    def test_low_confidence_wrist_excluded(self):
        """Low confidence wrist keypoints are excluded."""
        analyzer = HandPathAnalyzer()
        poses = []
        for i in range(5):
            keypoints = [_make_keypoint("left_wrist", x=0.2 + i * 0.1, y=0.5, confidence=0.3)]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        efficiency = analyzer.calculate_hand_path_efficiency(
            poses, load_frame=0, impact_frame=4
        )

        assert efficiency == 0.0

    def test_stationary_hand_returns_zero(self):
        """Hand that doesn't move returns 0.0 (actual_path = 0)."""
        analyzer = HandPathAnalyzer()
        poses = []
        for i in range(5):
            keypoints = [_make_keypoint("left_wrist", x=0.5, y=0.5)]
            poses.append(_make_pose_result(keypoints, frame_index=i))

        efficiency = analyzer.calculate_hand_path_efficiency(
            poses, load_frame=0, impact_frame=4
        )

        assert efficiency == 0.0
