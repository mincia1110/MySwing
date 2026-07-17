"""Unit tests for the swing phase classifier module (Task 7.1).

Tests phase classification with synthetic pose/bat data, transition frame
identification, duration calculation formula, correct phase ordering,
and graceful handling of incomplete data.
"""

import pytest

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.enums import SwingPhase
from app.models.pose import Keypoint, PoseResult
from app.models.swing import SwingPhaseResult, TransitionBoundary
from app.pipeline.swing_classifier import SwingPhaseClassifier

# ---------------------------------------------------------------------------
# Helper functions for creating synthetic test data
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
    keypoints: list[Keypoint] | None = None,
    overall_confidence: float = 0.9,
) -> PoseResult:
    """Create a PoseResult with default or custom keypoints."""
    if keypoints is None:
        keypoints = _default_keypoints()
    return PoseResult(
        frame_index=frame_index,
        keypoints=keypoints,
        person_id=1,
        is_primary_batter=True,
        overall_confidence=overall_confidence,
        is_low_confidence=False,
    )


def _default_keypoints(
    left_hip_x: float = 0.4,
    right_hip_x: float = 0.6,
    left_ankle_x: float = 0.4,
    left_ankle_y: float = 0.9,
    right_wrist_x: float = 0.55,
) -> list[Keypoint]:
    """Create a default set of keypoints for testing."""
    return [
        _make_keypoint("head", x=0.5, y=0.1),
        _make_keypoint("left_shoulder", x=0.4, y=0.3),
        _make_keypoint("right_shoulder", x=0.6, y=0.3),
        _make_keypoint("left_elbow", x=0.35, y=0.45),
        _make_keypoint("right_elbow", x=0.65, y=0.45),
        _make_keypoint("left_wrist", x=0.3, y=0.55),
        _make_keypoint("right_wrist", x=right_wrist_x, y=0.55),
        _make_keypoint("left_hip", x=left_hip_x, y=0.55),
        _make_keypoint("right_hip", x=right_hip_x, y=0.55),
        _make_keypoint("left_knee", x=0.4, y=0.7),
        _make_keypoint("right_knee", x=0.6, y=0.7),
        _make_keypoint("left_ankle", x=left_ankle_x, y=left_ankle_y),
        _make_keypoint("right_ankle", x=0.6, y=0.9),
        _make_keypoint("spine", x=0.5, y=0.3),
    ]


def _make_bat_detection(
    frame_index: int,
    position: tuple[float, float] = (320.0, 240.0),
    detected: bool = True,
    confidence: float = 0.95,
    is_predicted: bool = False,
    coordinate_space: str = "pixel",
) -> BatDetectionResult:
    """Create a BatDetectionResult."""
    return BatDetectionResult(
        frame_index=frame_index,
        detected=detected,
        position=position,
        orientation_angle=45.0,
        length_pixels=150.0,
        confidence=confidence,
        is_predicted=is_predicted,
        coordinate_space=coordinate_space,
    )


def _create_full_swing_sequence(num_frames: int = 60, fps: float = 30.0):
    """Create a synthetic full swing sequence with all 6 phases.

    Simulates a realistic swing with clear phase transitions:
    - Frames 0-9: Stance (minimal movement)
    - Frames 10-19: Load (hip/hand backward movement)
    - Frames 20-29: Stride (front ankle lifts)
    - Frames 30-39: Rotation (hip rotation, ankle plants)
    - Frames 40-44: Impact (peak bat speed)
    - Frames 45-59: Follow-through (bat speed decreases)
    """
    pose_sequence: list[PoseResult] = []
    bat_detections: list[BatDetectionResult] = []
    bat_speeds: list[float] = []

    for i in range(num_frames):
        # Default positions
        left_hip_x = 0.4
        right_hip_x = 0.6
        left_ankle_y = 0.9
        right_wrist_x = 0.55

        if i < 10:
            # STANCE: minimal movement
            pass
        elif i < 20:
            # LOAD: hip moves backward, hands move back
            progress = (i - 10) / 10.0
            right_hip_x = 0.6 + progress * 0.1  # hip moves back significantly
            right_wrist_x = 0.55 + progress * 0.12  # hands move back
        elif i < 30:
            # STRIDE: front ankle lifts
            progress = (i - 20) / 10.0
            left_ankle_y = 0.9 - progress * 0.2  # ankle lifts up dramatically
            right_hip_x = 0.7  # hip stays back from load
            right_wrist_x = 0.67  # hands stay back
        elif i < 40:
            # ROTATION: ankle plants, hip rotates
            progress = (i - 30) / 10.0
            left_ankle_y = 0.9  # ankle back down (planted)
            left_hip_x = 0.4 + progress * 0.08  # hips rotate significantly
            right_hip_x = 0.7 - progress * 0.08  # hips rotate
        elif i < 45:
            # IMPACT: peak bat speed zone
            left_hip_x = 0.48
            right_hip_x = 0.62
        else:
            # FOLLOW-THROUGH: deceleration
            left_hip_x = 0.48
            right_hip_x = 0.62

        keypoints = _default_keypoints(
            left_hip_x=left_hip_x,
            right_hip_x=right_hip_x,
            left_ankle_y=left_ankle_y,
            right_wrist_x=right_wrist_x,
        )
        pose_sequence.append(_make_pose(frame_index=i, keypoints=keypoints))

        # Bat detections with increasing then decreasing speed
        bat_x = 320.0 + i * 5.0
        bat_detections.append(_make_bat_detection(i, position=(bat_x, 240.0)))

    # Build bat speeds simulating acceleration then deceleration
    for i in range(num_frames - 1):
        if i < 30:
            speed = i * 2.0  # gradual acceleration
        elif i < 42:
            speed = 60.0 + (i - 30) * 10.0  # rapid acceleration
        elif i < 45:
            speed = 180.0  # peak speed
        else:
            speed = max(0.0, 180.0 - (i - 45) * 20.0)  # deceleration
        bat_speeds.append(speed)

    bat_trajectory = BatTrajectory(
        detections=bat_detections,
        bat_speed_pixels_per_frame=bat_speeds,
        tracking_accuracy=0.95,
        tracking_failures=[],
    )

    return pose_sequence, bat_trajectory, fps


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestClassifyPhases:
    """Test classify_phases with synthetic pose/bat data."""

    def test_full_swing_classifies_all_six_phases(self):
        """A complete swing sequence should identify all 6 phases."""
        pose_sequence, bat_trajectory, fps = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        result = classifier.classify_phases(pose_sequence, bat_trajectory, fps)

        # All 6 phases should be present
        assert SwingPhase.STANCE in result.phases
        assert SwingPhase.LOAD in result.phases
        assert SwingPhase.STRIDE in result.phases
        assert SwingPhase.ROTATION in result.phases
        assert SwingPhase.IMPACT in result.phases
        assert SwingPhase.FOLLOW_THROUGH in result.phases

    def test_predicted_wrist_proxy_cannot_create_contact_phases(self):
        """Pose-derived proxy points are not evidence of impact or follow-through."""
        pose_sequence, bat_trajectory, fps = _create_full_swing_sequence()
        for detection in bat_trajectory.detections:
            detection.is_predicted = True

        result = SwingPhaseClassifier().classify_phases(
            pose_sequence,
            bat_trajectory,
            fps,
        )

        assert SwingPhase.IMPACT not in result.phases
        assert SwingPhase.FOLLOW_THROUGH not in result.phases

    def test_mixed_proxy_speed_cannot_override_sparse_observed_lines(self):
        """Three early observations cannot unlock a later predicted contact peak."""
        pose_sequence, bat_trajectory, fps = _create_full_swing_sequence()
        for detection in bat_trajectory.detections[3:]:
            detection.is_predicted = True

        result = SwingPhaseClassifier().classify_phases(
            pose_sequence,
            bat_trajectory,
            fps,
        )

        assert SwingPhase.IMPACT not in result.phases
        assert SwingPhase.FOLLOW_THROUGH not in result.phases

    def test_phases_are_in_correct_order(self):
        """Phase start frames should be in chronological order."""
        pose_sequence, bat_trajectory, fps = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        result = classifier.classify_phases(pose_sequence, bat_trajectory, fps)

        expected_order = [
            SwingPhase.STANCE,
            SwingPhase.LOAD,
            SwingPhase.STRIDE,
            SwingPhase.ROTATION,
            SwingPhase.IMPACT,
            SwingPhase.FOLLOW_THROUGH,
        ]

        # Verify phases appear in correct order by start frame
        phase_starts = []
        for phase in expected_order:
            if phase in result.phases:
                phase_starts.append(result.phases[phase][0])

        assert phase_starts == sorted(phase_starts)

    def test_transitions_are_chronological(self):
        """Transition frame indices should be strictly increasing."""
        pose_sequence, bat_trajectory, fps = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        result = classifier.classify_phases(pose_sequence, bat_trajectory, fps)

        frame_indices = [t.frame_index for t in result.transitions]
        for i in range(1, len(frame_indices)):
            assert frame_indices[i] > frame_indices[i - 1]

    def test_phase_durations_are_positive(self):
        """All phase durations should be positive values."""
        pose_sequence, bat_trajectory, fps = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        result = classifier.classify_phases(pose_sequence, bat_trajectory, fps)

        for phase, duration in result.phase_durations_ms.items():
            assert duration >= 0.0, f"Phase {phase} has negative duration: {duration}"

    def test_empty_pose_sequence_returns_empty_result(self):
        """Empty pose sequence should return empty SwingPhaseResult."""
        classifier = SwingPhaseClassifier()
        bat_trajectory = BatTrajectory()

        result = classifier.classify_phases([], bat_trajectory, 30.0)

        assert result.phases == {}
        assert result.transitions == []
        assert result.phase_durations_ms == {}

    def test_pipeline_adds_only_observed_contact_to_partial_phases(self, monkeypatch):
        """Observed bat speed may add contact but cannot invent body phases."""
        from app.models.swing import SwingPhaseResult, TransitionBoundary
        from app.tasks import pipeline

        pose_sequence, bat_trajectory, fps = _create_full_swing_sequence(num_frames=60)

        partial_result = SwingPhaseResult(
            phases={
                SwingPhase.STANCE: (0, 2),
                SwingPhase.LOAD: (2, 40),
            },
            transitions=[
                TransitionBoundary(SwingPhase.STANCE, SwingPhase.LOAD, 2, 0.8),
            ],
        )

        def fake_classify_phases(self, pose_sequence_arg, bat_trajectory_arg, fps_arg):
            return partial_result

        monkeypatch.setattr(
            SwingPhaseClassifier,
            "classify_phases",
            fake_classify_phases,
        )

        result = pipeline._run_swing_classification(
            "analysis-id",
            {"pose_sequence": pipeline._serialize_dataclass(pose_sequence)},
            {"bat_trajectory": pipeline._serialize_dataclass(bat_trajectory)},
            fps,
        )

        assert set(result["phases"]) == {"stance", "load", "impact"}
        assert result["phases"]["impact"][0] == result["phases"]["impact"][1]
        assert "stride" not in result["phase_durations_ms"]
        assert "rotation" not in result["phase_durations_ms"]
        assert result["transitions"] == [
            {
                "from_phase": "stance",
                "to_phase": "load",
                "frame_index": 2,
                "confidence": 0.8,
            }
        ]
        assert result["phase_source"] == "mixed_pose_and_observed_bat_contact"
        assert result["phase_evidence"]["impact_method"] == "smoothed_peak_speed"
        assert result["phase_evidence"]["missing_phases_after_fallback"] == [
            "follow_through",
            "rotation",
            "stride",
        ]

    def test_speed_fallback_keeps_phase_ranges_monotonic_when_peak_is_early(self):
        """Speed fallback should not create inverted impact ranges for early peaks."""
        from app.tasks import pipeline

        pose_sequence, bat_trajectory, fps = _create_full_swing_sequence(num_frames=115)
        bat_trajectory.bat_speed_pixels_per_frame = [1.0] * 114
        bat_trajectory.bat_speed_pixels_per_frame[56] = 100.0

        phases, durations = pipeline._estimate_phases_from_speed(
            bat_trajectory, pose_sequence, fps
        )

        previous_end = -1
        for phase_name in [
            "stance",
            "load",
            "stride",
            "rotation",
            "impact",
            "follow_through",
        ]:
            start, end = phases[phase_name]
            assert start >= previous_end
            assert end >= start, phase_name
            previous_end = end
        assert all(duration >= 0.0 for duration in durations.values())

    def test_zero_fps_returns_empty_result(self):
        """Zero fps should return empty SwingPhaseResult."""
        pose_sequence, bat_trajectory, _ = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        result = classifier.classify_phases(pose_sequence, bat_trajectory, 0.0)

        assert result.phases == {}
        assert result.transitions == []

    def test_negative_fps_returns_empty_result(self):
        """Negative fps should return empty SwingPhaseResult."""
        pose_sequence, bat_trajectory, _ = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        result = classifier.classify_phases(pose_sequence, bat_trajectory, -30.0)

        assert result.phases == {}


class TestFindTransitionFrames:
    """Test transition frame identification."""

    def test_full_swing_has_five_transitions(self):
        """A complete swing should have 5 transitions (between 6 phases)."""
        pose_sequence, bat_trajectory, _ = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        transitions = classifier.find_transition_frames(pose_sequence, bat_trajectory)

        assert len(transitions) == 5

    def test_transition_phases_are_consecutive(self):
        """Each transition should connect consecutive phases."""
        pose_sequence, bat_trajectory, _ = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        transitions = classifier.find_transition_frames(pose_sequence, bat_trajectory)

        expected_pairs = [
            (SwingPhase.STANCE, SwingPhase.LOAD),
            (SwingPhase.LOAD, SwingPhase.STRIDE),
            (SwingPhase.STRIDE, SwingPhase.ROTATION),
            (SwingPhase.ROTATION, SwingPhase.IMPACT),
            (SwingPhase.IMPACT, SwingPhase.FOLLOW_THROUGH),
        ]

        for i, trans in enumerate(transitions):
            assert trans.from_phase == expected_pairs[i][0]
            assert trans.to_phase == expected_pairs[i][1]

    def test_transition_confidence_is_valid(self):
        """Transition confidence should be between 0 and 1."""
        pose_sequence, bat_trajectory, _ = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        transitions = classifier.find_transition_frames(pose_sequence, bat_trajectory)

        for trans in transitions:
            assert 0.0 <= trans.confidence <= 1.0

    def test_too_few_frames_returns_no_transitions(self):
        """Fewer than MIN_SWING_FRAMES should return empty transitions."""
        classifier = SwingPhaseClassifier()
        # Only 5 frames (below MIN_SWING_FRAMES=10)
        pose_sequence = [_make_pose(i) for i in range(5)]
        bat_trajectory = BatTrajectory()

        transitions = classifier.find_transition_frames(pose_sequence, bat_trajectory)

        assert transitions == []

    def test_stance_to_load_detected_by_hip_movement(self):
        """Stance→Load transition detected when hip moves backward."""
        classifier = SwingPhaseClassifier()

        # Create sequence where hip moves backward at frame 5
        pose_sequence = []
        for i in range(15):
            if i < 5:
                kps = _default_keypoints(right_hip_x=0.6)
            else:
                kps = _default_keypoints(right_hip_x=0.6 + 0.05)
            pose_sequence.append(_make_pose(i, keypoints=kps))

        bat_trajectory = BatTrajectory(
            detections=[_make_bat_detection(i) for i in range(15)],
            bat_speed_pixels_per_frame=[5.0] * 14,
            tracking_accuracy=0.95,
            tracking_failures=[],
        )

        transitions = classifier.find_transition_frames(pose_sequence, bat_trajectory)

        # Should detect at least the Stance→Load transition
        stance_to_load = [
            t for t in transitions if t.from_phase == SwingPhase.STANCE
        ]
        assert len(stance_to_load) >= 1
        # Transition should be around frame 5 (±2 frame tolerance)
        assert abs(stance_to_load[0].frame_index - 5) <= 2

    def test_stance_to_load_detected_by_hand_movement(self):
        """Stance→Load transition detected when hands move backward."""
        classifier = SwingPhaseClassifier()

        pose_sequence = []
        for i in range(15):
            if i < 5:
                kps = _default_keypoints(right_wrist_x=0.55)
            else:
                kps = _default_keypoints(right_wrist_x=0.55 + 0.05)
            pose_sequence.append(_make_pose(i, keypoints=kps))

        bat_trajectory = BatTrajectory(
            detections=[_make_bat_detection(i) for i in range(15)],
            bat_speed_pixels_per_frame=[5.0] * 14,
            tracking_accuracy=0.95,
            tracking_failures=[],
        )

        transitions = classifier.find_transition_frames(pose_sequence, bat_trajectory)

        stance_to_load = [
            t for t in transitions if t.from_phase == SwingPhase.STANCE
        ]
        assert len(stance_to_load) >= 1

    def test_impact_prefers_first_substantial_peak_before_follow_through_peak(self):
        """A later, slightly larger wrist peak must not replace contact."""
        classifier = SwingPhaseClassifier()
        speeds = [0.0] * 50
        # Contact-adjacent cluster centered at offset 20.
        speeds[19:22] = [82.0, 90.0, 84.0]
        # Follow-through cluster is slightly faster but later.
        speeds[34:37] = [88.0, 100.0, 90.0]

        impact = classifier._find_rotation_to_impact(
            speeds,
            start_frame=100,
            stride_to_rotation=110,
        )

        assert impact == 120

    def test_impact_is_unavailable_without_rotation_anchor(self):
        """A pose-proxy speed peak alone must not be labeled as contact."""
        classifier = SwingPhaseClassifier()

        impact = classifier._find_rotation_to_impact(
            [0.0, 2.0, 8.0, 3.0, 1.0],
            start_frame=40,
            stride_to_rotation=None,
        )

        assert impact is None

    def test_follow_through_uses_selected_impact_peak_not_later_global_peak(self):
        """Follow-through threshold is local to the chosen contact event."""
        classifier = SwingPhaseClassifier()
        speeds = [0.0] * 45
        speeds[9:12] = [72.0, 90.0, 80.0]
        speeds[12:15] = [55.0, 50.0, 45.0]
        speeds[30:33] = [95.0, 110.0, 100.0]

        follow = classifier._find_impact_to_follow_through(
            speeds,
            start_frame=200,
            rotation_to_impact=210,
        )

        assert follow == 212

    def test_follow_through_is_unavailable_without_impact_anchor(self):
        """Post-contact phase cannot precede an unobserved contact event."""
        classifier = SwingPhaseClassifier()

        follow = classifier._find_impact_to_follow_through(
            [0.0, 5.0, 2.0, 1.0],
            start_frame=100,
            rotation_to_impact=None,
        )

        assert follow is None

    def test_follow_through_is_unavailable_without_sustained_deceleration(self):
        """A fixed post-contact offset is not evidence of follow-through."""
        classifier = SwingPhaseClassifier()
        speeds = [0.0] * 20
        speeds[8:11] = [80.0, 90.0, 85.0]
        speeds[11:] = [75.0] * 9

        follow = classifier._find_impact_to_follow_through(
            speeds,
            start_frame=100,
            rotation_to_impact=109,
        )

        assert follow is None


class TestValidityAwarePoseSignals:
    """Test validity and coordinate scaling of pose-derived transitions."""

    def test_missing_front_ankle_never_triggers_foot_plant(self):
        """Unavailable ankle pairs must not look like a planted foot."""
        pose_sequence = []
        for frame_index in range(10):
            keypoints = _default_keypoints(
                right_hip_x=0.6 if frame_index == 0 else 0.65,
                left_ankle_y=0.84 if frame_index == 3 else 0.9,
            )
            if frame_index >= 4:
                keypoints = [kp for kp in keypoints if kp.name != "left_ankle"]
            pose_sequence.append(_make_pose(frame_index, keypoints=keypoints))

        transitions = SwingPhaseClassifier().find_transition_frames(
            pose_sequence, BatTrajectory()
        )

        transition_targets = {transition.to_phase for transition in transitions}
        assert SwingPhase.LOAD in transition_targets
        assert SwingPhase.STRIDE in transition_targets
        assert SwingPhase.ROTATION not in transition_targets

    def test_pose_gaps_do_not_count_as_consecutive_foot_plant_samples(self):
        """Adjacent list entries across dropped frames are not consecutive evidence."""
        classifier = SwingPhaseClassifier()
        pose_sequence = [
            _make_pose(frame_index)
            for frame_index in (0, 1, 2, 5, 6)
        ]

        transition = classifier._find_stride_to_rotation(
            pose_sequence,
            ankle_stable=[1.0, 1.0, 0.0, 0.0],
            hip_rotation=[0.0, 0.0, 0.02, 0.02],
            load_to_stride=1,
        )

        assert transition is None

    def test_one_valid_stable_sample_is_insufficient_for_foot_plant(self):
        """A single low-motion pair cannot confirm ankle stabilization."""
        pose_sequence = [_make_pose(frame_index) for frame_index in range(9)]
        ankle_stable = [0.02] * 8
        ankle_stable[4] = 0.001
        hip_rotation = [0.0] * 8
        hip_rotation[4] = 0.02

        transition = SwingPhaseClassifier()._find_stride_to_rotation(
            pose_sequence,
            ankle_stable,
            hip_rotation,
            load_to_stride=2,
        )

        assert transition is None

    @pytest.mark.parametrize(
        ("rotation_idx", "expected_frame"),
        [
            (3, 6),  # Immediately before the planted pair.
            (4, 6),  # On the first stable sample.
            (5, 6),  # On the second stable sample.
            (6, 7),  # Immediately after the planted pair.
        ],
    )
    def test_two_stable_samples_accept_adjacent_rotation(
        self, rotation_idx, expected_frame
    ):
        """Rotation may coincide with or immediately border a stable pair."""
        frame_indices = list(range(9))
        pose_sequence = [_make_pose(index) for index in frame_indices]
        ankle_stable = [0.02] * 8
        ankle_stable[4:6] = [0.001, 0.002]
        hip_rotation = [0.0] * 8
        hip_rotation[rotation_idx] = 0.02

        transition = SwingPhaseClassifier()._find_stride_to_rotation(
            pose_sequence,
            ankle_stable,
            hip_rotation,
            load_to_stride=2,
        )

        assert transition == expected_frame

    def test_two_stable_samples_support_no_rotation_fallback_after_stride(self):
        """Confirmed stability can fall back after a real stride and lead-in."""
        frame_indices = list(range(9))
        pose_sequence = [_make_pose(index) for index in frame_indices]
        ankle_stable = [0.02] * 8
        ankle_stable[4:6] = [0.001, 0.002]

        classifier = SwingPhaseClassifier()
        transition = classifier._find_stride_to_rotation(
            pose_sequence,
            ankle_stable,
            [0.0] * 8,
            load_to_stride=2,
        )
        without_stride = classifier._find_stride_to_rotation(
            pose_sequence,
            ankle_stable,
            [0.0] * 8,
            load_to_stride=None,
        )

        assert transition == 6
        assert without_stride is None

    def test_missing_hip_and_wrist_cues_do_not_create_load_transition(self):
        """Missing movement cues remain unavailable rather than numeric motion."""
        keypoints = [
            kp
            for kp in _default_keypoints()
            if kp.name not in {"right_hip", "right_wrist"}
        ]
        pose_sequence = [
            _make_pose(3, keypoints=keypoints),
            _make_pose(17, keypoints=keypoints),
        ]
        classifier = SwingPhaseClassifier()

        hip_movement = classifier._compute_hip_backward_signal(pose_sequence)
        hand_movement = classifier._compute_hand_backward_signal(pose_sequence)

        assert hip_movement == [None]
        assert hand_movement == [None]
        assert (
            classifier._find_stance_to_load(
                pose_sequence, hip_movement, hand_movement
            )
            is None
        )

    def test_portrait_and_landscape_use_same_height_normalized_motion(self):
        """Equivalent pixel x motion produces the same signals and event."""
        results = []
        for width, height in [(200.0, 100.0), (50.0, 100.0)]:
            normalized_dx = 2.0 / width
            pose_sequence = [
                _make_pose(
                    11,
                    keypoints=_default_keypoints(
                        right_hip_x=0.6,
                        right_wrist_x=0.55,
                        left_ankle_x=0.4,
                    ),
                ),
                _make_pose(
                    12,
                    keypoints=_default_keypoints(
                        right_hip_x=0.6 + normalized_dx,
                        right_wrist_x=0.55 + normalized_dx,
                        left_ankle_x=0.4 + normalized_dx,
                    ),
                ),
            ]
            classifier = SwingPhaseClassifier(
                video_width=width,
                video_height=height,
            )
            hip_movement = classifier._compute_hip_backward_signal(pose_sequence)
            hand_movement = classifier._compute_hand_backward_signal(pose_sequence)
            ankle_movement = classifier._compute_ankle_stabilize_signal(
                pose_sequence
            )
            hip_rotation = classifier._compute_hip_rotation_signal(pose_sequence)
            transition = classifier._find_stance_to_load(
                pose_sequence, hip_movement, hand_movement
            )
            results.append(
                (
                    hip_movement[0],
                    hand_movement[0],
                    ankle_movement[0],
                    hip_rotation[0],
                    transition,
                )
            )

        assert results[0][:-1] == pytest.approx([0.02, 0.02, 0.02, 0.02])
        assert results[1][:-1] == pytest.approx(results[0][:-1])
        assert results[0][-1] == results[1][-1] == 12

    def test_default_dimensions_preserve_square_normalized_behavior(self):
        """The default 1x1 scale retains prior normalized x displacement."""
        pose_sequence = [
            _make_pose(101, keypoints=_default_keypoints(right_hip_x=0.6)),
            _make_pose(102, keypoints=_default_keypoints(right_hip_x=0.61)),
        ]
        default_classifier = SwingPhaseClassifier()
        square_classifier = SwingPhaseClassifier(
            video_width=640.0,
            video_height=640.0,
        )

        default_signal = default_classifier._compute_hip_backward_signal(
            pose_sequence
        )
        square_signal = square_classifier._compute_hip_backward_signal(
            pose_sequence
        )

        assert default_signal == pytest.approx([0.01])
        assert square_signal == pytest.approx(default_signal)
        assert (
            default_classifier._find_stance_to_load(
                pose_sequence, default_signal, [None]
            )
            == 102
        )

    def test_observed_normalized_bat_speed_is_aspect_invariant(self):
        """Equivalent pixel barrel motion has the same peak in any aspect ratio."""
        physical_points = [(0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (40.0, 60.0)]
        results = []
        for width, height in ((200.0, 100.0), (50.0, 100.0)):
            trajectory = BatTrajectory(
                detections=[
                    _make_bat_detection(
                        frame_index,
                        position=(x / width, y / height),
                        coordinate_space="normalized",
                    )
                    for frame_index, (x, y) in enumerate(physical_points)
                ]
            )
            classifier = SwingPhaseClassifier(
                video_width=width,
                video_height=height,
            )
            results.append(
                classifier._get_observed_bat_speeds(
                    trajectory,
                    start_frame=0,
                    end_frame=3,
                )
            )

        assert results[0] == pytest.approx([0.0, 40.0, 30.0, 30.0])
        assert results[1] == pytest.approx(results[0])

    @pytest.mark.parametrize(
        "dimensions",
        [
            {"video_width": 0.0},
            {"video_width": -1.0},
            {"video_width": float("nan")},
            {"video_height": 0.0},
            {"video_height": float("inf")},
            {"video_height": None},
        ],
    )
    def test_video_dimensions_must_be_positive_and_finite(self, dimensions):
        """Invalid dimensions are rejected before deriving the aspect scale."""
        with pytest.raises(ValueError, match="positive finite"):
            SwingPhaseClassifier(**dimensions)


class TestCalculatePhaseDurations:
    """Test duration calculation formula."""

    def test_basic_duration_calculation(self):
        """Duration = (end_frame - start_frame) / fps * 1000."""
        classifier = SwingPhaseClassifier()
        transitions = [
            TransitionBoundary(
                from_phase=SwingPhase.STANCE,
                to_phase=SwingPhase.LOAD,
                frame_index=10,
                confidence=0.9,
            ),
            TransitionBoundary(
                from_phase=SwingPhase.LOAD,
                to_phase=SwingPhase.STRIDE,
                frame_index=20,
                confidence=0.9,
            ),
            TransitionBoundary(
                from_phase=SwingPhase.STRIDE,
                to_phase=SwingPhase.ROTATION,
                frame_index=30,
                confidence=0.9,
            ),
        ]

        durations = classifier.calculate_phase_durations(transitions, fps=30.0)

        # LOAD: (20 - 10) / 30 * 1000 = 333.33 ms
        assert abs(durations[SwingPhase.LOAD] - 333.33) < 0.1
        # STRIDE: (30 - 20) / 30 * 1000 = 333.33 ms
        assert abs(durations[SwingPhase.STRIDE] - 333.33) < 0.1

    def test_duration_with_60fps(self):
        """Duration calculation at 60fps."""
        classifier = SwingPhaseClassifier()
        transitions = [
            TransitionBoundary(
                from_phase=SwingPhase.STANCE,
                to_phase=SwingPhase.LOAD,
                frame_index=0,
                confidence=0.9,
            ),
            TransitionBoundary(
                from_phase=SwingPhase.LOAD,
                to_phase=SwingPhase.STRIDE,
                frame_index=6,
                confidence=0.9,
            ),
        ]

        durations = classifier.calculate_phase_durations(transitions, fps=60.0)

        # LOAD: (6 - 0) / 60 * 1000 = 100 ms
        assert abs(durations[SwingPhase.LOAD] - 100.0) < 0.01

    def test_duration_with_120fps(self):
        """Duration calculation at 120fps (high-speed camera)."""
        classifier = SwingPhaseClassifier()
        transitions = [
            TransitionBoundary(
                from_phase=SwingPhase.ROTATION,
                to_phase=SwingPhase.IMPACT,
                frame_index=100,
                confidence=0.9,
            ),
            TransitionBoundary(
                from_phase=SwingPhase.IMPACT,
                to_phase=SwingPhase.FOLLOW_THROUGH,
                frame_index=112,
                confidence=0.9,
            ),
        ]

        durations = classifier.calculate_phase_durations(transitions, fps=120.0)

        # IMPACT: (112 - 100) / 120 * 1000 = 100 ms
        assert abs(durations[SwingPhase.IMPACT] - 100.0) < 0.01

    def test_empty_transitions_returns_empty_dict(self):
        """Empty transitions list returns empty durations."""
        classifier = SwingPhaseClassifier()

        durations = classifier.calculate_phase_durations([], fps=30.0)

        assert durations == {}

    def test_single_transition_returns_empty_dict(self):
        """Single transition has no duration to calculate between pairs."""
        classifier = SwingPhaseClassifier()
        transitions = [
            TransitionBoundary(
                from_phase=SwingPhase.STANCE,
                to_phase=SwingPhase.LOAD,
                frame_index=10,
                confidence=0.9,
            ),
        ]

        durations = classifier.calculate_phase_durations(transitions, fps=30.0)

        assert durations == {}

    def test_zero_fps_returns_empty_dict(self):
        """Zero fps returns empty durations."""
        classifier = SwingPhaseClassifier()
        transitions = [
            TransitionBoundary(
                from_phase=SwingPhase.STANCE,
                to_phase=SwingPhase.LOAD,
                frame_index=10,
                confidence=0.9,
            ),
            TransitionBoundary(
                from_phase=SwingPhase.LOAD,
                to_phase=SwingPhase.STRIDE,
                frame_index=20,
                confidence=0.9,
            ),
        ]

        durations = classifier.calculate_phase_durations(transitions, fps=0.0)

        assert durations == {}

    def test_all_five_transitions_produce_four_durations(self):
        """Five transitions produce durations for 4 middle phases."""
        classifier = SwingPhaseClassifier()
        transitions = [
            TransitionBoundary(SwingPhase.STANCE, SwingPhase.LOAD, 10, 0.9),
            TransitionBoundary(SwingPhase.LOAD, SwingPhase.STRIDE, 20, 0.9),
            TransitionBoundary(SwingPhase.STRIDE, SwingPhase.ROTATION, 30, 0.9),
            TransitionBoundary(SwingPhase.ROTATION, SwingPhase.IMPACT, 40, 0.9),
            TransitionBoundary(SwingPhase.IMPACT, SwingPhase.FOLLOW_THROUGH, 45, 0.9),
        ]

        durations = classifier.calculate_phase_durations(transitions, fps=30.0)

        assert SwingPhase.LOAD in durations
        assert SwingPhase.STRIDE in durations
        assert SwingPhase.ROTATION in durations
        assert SwingPhase.IMPACT in durations
        assert len(durations) == 4


class TestPhaseOrdering:
    """Test that all 6 phases are identified in correct order."""

    def test_phase_start_frames_are_non_decreasing(self):
        """Phase start frames should be non-decreasing."""
        pose_sequence, bat_trajectory, fps = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        result = classifier.classify_phases(pose_sequence, bat_trajectory, fps)

        phase_order = [
            SwingPhase.STANCE,
            SwingPhase.LOAD,
            SwingPhase.STRIDE,
            SwingPhase.ROTATION,
            SwingPhase.IMPACT,
            SwingPhase.FOLLOW_THROUGH,
        ]

        prev_start = -1
        for phase in phase_order:
            if phase in result.phases:
                start, end = result.phases[phase]
                assert start >= prev_start, (
                    f"Phase {phase} starts at {start} but previous ended at {prev_start}"
                )
                prev_start = start

    def test_no_phase_gaps(self):
        """Each phase should start where the previous one ends."""
        pose_sequence, bat_trajectory, fps = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        result = classifier.classify_phases(pose_sequence, bat_trajectory, fps)

        phase_order = [
            SwingPhase.STANCE,
            SwingPhase.LOAD,
            SwingPhase.STRIDE,
            SwingPhase.ROTATION,
            SwingPhase.IMPACT,
            SwingPhase.FOLLOW_THROUGH,
        ]

        present_phases = [p for p in phase_order if p in result.phases]

        for i in range(1, len(present_phases)):
            prev_end = result.phases[present_phases[i - 1]][1]
            curr_start = result.phases[present_phases[i]][0]
            assert curr_start == prev_end, (
                f"Gap between {present_phases[i-1]} (end={prev_end}) "
                f"and {present_phases[i]} (start={curr_start})"
            )


class TestIncompleteData:
    """Test graceful handling of incomplete data."""

    def test_no_bat_trajectory_still_classifies_pose_phases(self):
        """Missing bat trajectory should still classify early phases."""
        pose_sequence, _, fps = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()
        empty_trajectory = BatTrajectory()

        result = classifier.classify_phases(pose_sequence, empty_trajectory, fps)

        # Should at least detect Stance and Load from pose data
        assert SwingPhase.STANCE in result.phases

    def test_low_confidence_keypoints_handled_gracefully(self):
        """Low confidence keypoints should not crash the classifier."""
        classifier = SwingPhaseClassifier()

        pose_sequence = []
        for i in range(20):
            keypoints = [
                _make_keypoint(name, confidence=0.2)  # Below threshold
                for name in [
                    "head", "left_shoulder", "right_shoulder",
                    "left_elbow", "right_elbow", "left_wrist",
                    "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle",
                    "right_ankle", "spine",
                ]
            ]
            pose_sequence.append(_make_pose(i, keypoints=keypoints))

        bat_trajectory = BatTrajectory(
            detections=[_make_bat_detection(i) for i in range(20)],
            bat_speed_pixels_per_frame=[5.0] * 19,
            tracking_accuracy=0.95,
            tracking_failures=[],
        )

        # Should not raise an exception
        result = classifier.classify_phases(pose_sequence, bat_trajectory, 30.0)
        assert isinstance(result, SwingPhaseResult)

    def test_single_frame_returns_minimal_result(self):
        """Single frame should return empty result (below MIN_SWING_FRAMES)."""
        classifier = SwingPhaseClassifier()
        pose_sequence = [_make_pose(0)]
        bat_trajectory = BatTrajectory()

        result = classifier.classify_phases(pose_sequence, bat_trajectory, 30.0)

        # Single frame is below MIN_SWING_FRAMES, so no transitions
        # but classify_phases still returns a result with the frame as STANCE
        assert isinstance(result, SwingPhaseResult)

    def test_missing_keypoints_in_some_frames(self):
        """Frames with missing keypoints should be handled gracefully."""
        classifier = SwingPhaseClassifier()

        pose_sequence = []
        for i in range(20):
            if i % 3 == 0:
                # Every 3rd frame has minimal keypoints
                keypoints = [_make_keypoint("head", x=0.5, y=0.1)]
            else:
                keypoints = _default_keypoints()
            pose_sequence.append(_make_pose(i, keypoints=keypoints))

        bat_trajectory = BatTrajectory(
            detections=[_make_bat_detection(i) for i in range(20)],
            bat_speed_pixels_per_frame=[10.0] * 19,
            tracking_accuracy=0.9,
            tracking_failures=[],
        )

        # Should not raise an exception
        result = classifier.classify_phases(pose_sequence, bat_trajectory, 30.0)
        assert isinstance(result, SwingPhaseResult)

    def test_all_bat_detections_missing(self):
        """All bat detections missing should still produce a result."""
        pose_sequence, _, fps = _create_full_swing_sequence()
        classifier = SwingPhaseClassifier()

        bat_trajectory = BatTrajectory(
            detections=[
                _make_bat_detection(i, detected=False) for i in range(60)
            ],
            bat_speed_pixels_per_frame=[0.0] * 59,
            tracking_accuracy=0.0,
            tracking_failures=[(0, 59)],
        )

        result = classifier.classify_phases(pose_sequence, bat_trajectory, fps)

        # Should still detect pose-based phases
        assert isinstance(result, SwingPhaseResult)
        assert SwingPhase.STANCE in result.phases
