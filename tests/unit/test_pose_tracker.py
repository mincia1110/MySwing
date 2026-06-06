"""Unit tests for the PoseTracker module.

Tests multi-frame tracking, identity maintenance, interpolation,
and low-confidence flagging logic.

Requirements validated:
- 3.2: Track keypoints across frames, maintain identity with ≤3 frame gap
- 3.3: Interpolate occluded keypoints (≤5 frames, <40% occluded)
- 3.4: Flag low-confidence frames (>5 frames or ≥40% occluded)
"""

import pytest

from app.models.pose import Keypoint, PoseResult
from app.pipeline.pose_tracker import (
    MAX_IDENTITY_GAP_FRAMES,
    MAX_INTERPOLATION_GAP_FRAMES,
    MAX_OCCLUSION_RATIO,
    PoseTracker,
)


# --- Helper functions ---


def make_keypoints(names: list[str], confidence: float = 0.9) -> list[Keypoint]:
    """Create a list of keypoints with given names and uniform confidence."""
    return [
        Keypoint(
            x=0.5 + i * 0.01,
            y=0.5 + i * 0.01,
            z=0.0,
            confidence=confidence,
            name=name,
        )
        for i, name in enumerate(names)
    ]


def make_pose_result(
    frame_index: int,
    keypoint_names: list[str] | None = None,
    person_id: int = 0,
    confidence: float = 0.9,
) -> PoseResult:
    """Create a PoseResult with specified parameters."""
    if keypoint_names is None:
        keypoint_names = [
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "spine",
            "neck",
            "left_ear",
            "right_ear",
        ]
    keypoints = make_keypoints(keypoint_names, confidence)
    return PoseResult(
        frame_index=frame_index,
        keypoints=keypoints,
        person_id=person_id,
        is_primary_batter=True,
        overall_confidence=confidence,
        is_low_confidence=False,
    )


ALL_KEYPOINT_NAMES = [
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "spine",
    "neck",
    "left_ear",
    "right_ear",
]


# --- Tests for identity tracking ---


class TestIdentityTracking:
    """Tests for person identity maintenance across frames."""

    def test_consecutive_frames_maintain_identity(self):
        """Consecutive frames (no gap) should maintain the same person_id."""
        tracker = PoseTracker()
        results = [
            make_pose_result(frame_index=0),
            make_pose_result(frame_index=1),
            make_pose_result(frame_index=2),
        ]

        tracked = tracker.track_across_frames(results)

        # All should have the same person_id
        assert tracked[0].person_id == tracked[1].person_id
        assert tracked[1].person_id == tracked[2].person_id

    def test_identity_maintained_within_1_frame_gap(self):
        """Identity should be maintained with 1 frame gap (≤3 threshold)."""
        tracker = PoseTracker()
        # Frames 0, 2 (gap of 1 frame at index 1)
        results = [
            make_pose_result(frame_index=0),
            make_pose_result(frame_index=2),
        ]

        tracked = tracker.track_across_frames(results)

        assert tracked[0].person_id == tracked[1].person_id

    def test_identity_maintained_within_2_frame_gap(self):
        """Identity should be maintained with 2 frame gap (≤3 threshold)."""
        tracker = PoseTracker()
        # Frames 0, 3 (gap of 2 frames at indices 1, 2)
        results = [
            make_pose_result(frame_index=0),
            make_pose_result(frame_index=3),
        ]

        tracked = tracker.track_across_frames(results)

        assert tracked[0].person_id == tracked[1].person_id

    def test_identity_maintained_within_3_frame_gap(self):
        """Identity should be maintained with exactly 3 frame gap (boundary)."""
        tracker = PoseTracker()
        # Frames 0, 4 (gap of 3 frames at indices 1, 2, 3)
        results = [
            make_pose_result(frame_index=0),
            make_pose_result(frame_index=4),
        ]

        tracked = tracker.track_across_frames(results)

        assert tracked[0].person_id == tracked[1].person_id

    def test_identity_reset_after_4_frame_gap(self):
        """Identity should be reset with 4 frame gap (>3 threshold)."""
        tracker = PoseTracker()
        # Frames 0, 5 (gap of 4 frames at indices 1, 2, 3, 4)
        results = [
            make_pose_result(frame_index=0),
            make_pose_result(frame_index=5),
        ]

        tracked = tracker.track_across_frames(results)

        assert tracked[0].person_id != tracked[1].person_id

    def test_identity_reset_after_large_gap(self):
        """Identity should be reset with a large gap (>>3 frames)."""
        tracker = PoseTracker()
        # Frames 0, 20 (gap of 19 frames)
        results = [
            make_pose_result(frame_index=0),
            make_pose_result(frame_index=20),
        ]

        tracked = tracker.track_across_frames(results)

        assert tracked[0].person_id != tracked[1].person_id

    def test_multiple_identity_resets(self):
        """Multiple large gaps should produce multiple different person_ids."""
        tracker = PoseTracker()
        # Three segments with large gaps between them
        results = [
            make_pose_result(frame_index=0),
            make_pose_result(frame_index=1),
            make_pose_result(frame_index=10),  # gap of 8 → reset
            make_pose_result(frame_index=11),
            make_pose_result(frame_index=20),  # gap of 8 → reset
        ]

        tracked = tracker.track_across_frames(results)

        # First two should share ID
        assert tracked[0].person_id == tracked[1].person_id
        # Third and fourth should share a different ID
        assert tracked[2].person_id == tracked[3].person_id
        assert tracked[2].person_id != tracked[0].person_id
        # Fifth should have yet another ID
        assert tracked[4].person_id != tracked[2].person_id
        assert tracked[4].person_id != tracked[0].person_id

    def test_empty_input_returns_empty(self):
        """Empty input should return empty list."""
        tracker = PoseTracker()
        assert tracker.track_across_frames([]) == []

    def test_single_frame_gets_person_id(self):
        """Single frame should get a valid person_id."""
        tracker = PoseTracker()
        results = [make_pose_result(frame_index=0)]

        tracked = tracker.track_across_frames(results)

        assert len(tracked) == 1
        assert isinstance(tracked[0].person_id, int)


# --- Tests for interpolation ---


class TestInterpolation:
    """Tests for occluded keypoint interpolation."""

    def test_interpolation_within_5_frames_and_below_40_percent(self):
        """Keypoints should be interpolated when ≤5 frames and <40% occluded."""
        tracker = PoseTracker()

        # Frame 0: all keypoints present
        # Frame 1: missing 2 keypoints (2/17 ≈ 11.8% < 40%)
        # Frame 2: all keypoints present
        full_names = ALL_KEYPOINT_NAMES
        partial_names = [n for n in full_names if n not in ("left_wrist", "right_wrist")]

        results = [
            make_pose_result(frame_index=0, keypoint_names=full_names),
            make_pose_result(frame_index=1, keypoint_names=partial_names),
            make_pose_result(frame_index=2, keypoint_names=full_names),
        ]

        tracked = tracker.track_across_frames(results)

        # Frame 1 should now have interpolated left_wrist and right_wrist
        frame1_names = {kp.name for kp in tracked[1].keypoints}
        assert "left_wrist" in frame1_names
        assert "right_wrist" in frame1_names

    def test_interpolation_linear_values(self):
        """Interpolated keypoints should have linearly interpolated coordinates."""
        tracker = PoseTracker()

        # Use 5 keypoints total so missing 1 = 20% < 40% threshold
        common_kps = [
            Keypoint(x=0.0, y=0.0, z=0.0, confidence=0.9, name="head"),
            Keypoint(x=0.1, y=0.1, z=0.0, confidence=0.9, name="left_shoulder"),
            Keypoint(x=0.2, y=0.2, z=0.0, confidence=0.9, name="right_shoulder"),
            Keypoint(x=0.3, y=0.3, z=0.0, confidence=0.9, name="left_hip"),
        ]

        kp_frame0 = common_kps + [
            Keypoint(x=0.2, y=0.4, z=0.0, confidence=0.8, name="left_wrist"),
        ]
        kp_frame1 = list(common_kps)  # left_wrist missing (1/5 = 20% < 40%)
        kp_frame2 = common_kps + [
            Keypoint(x=0.6, y=0.8, z=0.0, confidence=0.8, name="left_wrist"),
        ]

        results = [
            PoseResult(
                frame_index=0, keypoints=kp_frame0, person_id=0,
                is_primary_batter=True, overall_confidence=0.85,
                is_low_confidence=False,
            ),
            PoseResult(
                frame_index=1, keypoints=kp_frame1, person_id=0,
                is_primary_batter=True, overall_confidence=0.9,
                is_low_confidence=False,
            ),
            PoseResult(
                frame_index=2, keypoints=kp_frame2, person_id=0,
                is_primary_batter=True, overall_confidence=0.85,
                is_low_confidence=False,
            ),
        ]

        tracked = tracker.track_across_frames(results)

        # Find interpolated left_wrist in frame 1
        frame1_wrist = next(
            (kp for kp in tracked[1].keypoints if kp.name == "left_wrist"), None
        )
        assert frame1_wrist is not None
        # Linear interpolation: midpoint between (0.2, 0.4) and (0.6, 0.8)
        assert abs(frame1_wrist.x - 0.4) < 1e-6
        assert abs(frame1_wrist.y - 0.6) < 1e-6

    def test_no_interpolation_when_occlusion_exceeds_40_percent(self):
        """No interpolation when ≥40% of keypoints are occluded."""
        tracker = PoseTracker()

        full_names = ALL_KEYPOINT_NAMES  # 17 keypoints
        # Remove 7 keypoints (7/17 ≈ 41.2% > 40%)
        partial_names = full_names[:10]

        results = [
            make_pose_result(frame_index=0, keypoint_names=full_names),
            make_pose_result(frame_index=1, keypoint_names=partial_names),
            make_pose_result(frame_index=2, keypoint_names=full_names),
        ]

        tracked = tracker.track_across_frames(results)

        # Frame 1 should NOT have interpolated keypoints (occlusion too high)
        frame1_names = {kp.name for kp in tracked[1].keypoints}
        # The missing keypoints should still be missing
        assert len(frame1_names) == len(partial_names)

    def test_no_interpolation_when_gap_exceeds_5_frames(self):
        """No interpolation when occlusion persists beyond 5 consecutive frames."""
        tracker = PoseTracker()

        full_names = ALL_KEYPOINT_NAMES
        # Missing 1 keypoint (well below 40%)
        partial_names = [n for n in full_names if n != "left_wrist"]

        # 7 consecutive frames with left_wrist missing (gap > 5)
        results = [make_pose_result(frame_index=0, keypoint_names=full_names)]
        for i in range(1, 8):
            results.append(make_pose_result(frame_index=i, keypoint_names=partial_names))
        results.append(make_pose_result(frame_index=8, keypoint_names=full_names))

        tracked = tracker.track_across_frames(results)

        # Middle frames should NOT have interpolated left_wrist (gap > 5)
        for i in range(1, 8):
            frame_names = {kp.name for kp in tracked[i].keypoints}
            assert "left_wrist" not in frame_names

    def test_interpolation_at_boundary_5_frames(self):
        """Interpolation should work at exactly 5 consecutive occluded frames."""
        tracker = PoseTracker()

        full_names = ALL_KEYPOINT_NAMES
        # Missing 1 keypoint (well below 40%)
        partial_names = [n for n in full_names if n != "left_wrist"]

        # Exactly 5 consecutive frames with left_wrist missing
        results = [make_pose_result(frame_index=0, keypoint_names=full_names)]
        for i in range(1, 6):
            results.append(make_pose_result(frame_index=i, keypoint_names=partial_names))
        results.append(make_pose_result(frame_index=6, keypoint_names=full_names))

        tracked = tracker.track_across_frames(results)

        # Middle frames should have interpolated left_wrist (gap = 5, within threshold)
        for i in range(1, 6):
            frame_names = {kp.name for kp in tracked[i].keypoints}
            assert "left_wrist" in frame_names

    def test_interpolation_not_applied_across_different_person_ids(self):
        """Interpolation should not cross person_id boundaries."""
        tracker = PoseTracker()

        full_names = ALL_KEYPOINT_NAMES
        partial_names = [n for n in full_names if n != "left_wrist"]

        # Large gap between frame 0 and frame 10 → different person_ids
        results = [
            make_pose_result(frame_index=0, keypoint_names=full_names),
            make_pose_result(frame_index=10, keypoint_names=partial_names),
            make_pose_result(frame_index=11, keypoint_names=full_names),
        ]

        tracked = tracker.track_across_frames(results)

        # Frame at index 10 should have a different person_id
        assert tracked[0].person_id != tracked[1].person_id
        # Interpolation within the second person's group should still work
        frame10_names = {kp.name for kp in tracked[1].keypoints}
        assert "left_wrist" in frame10_names  # interpolated from frame 11


# --- Tests for low-confidence flagging ---


class TestLowConfidenceFlagging:
    """Tests for low-confidence frame flagging."""

    def test_flag_when_occlusion_exceeds_40_percent(self):
        """Frames with ≥40% keypoints occluded should be flagged low-confidence."""
        tracker = PoseTracker()

        full_names = ALL_KEYPOINT_NAMES  # 17 keypoints
        # Remove 7 keypoints (7/17 ≈ 41.2% ≥ 40%)
        partial_names = full_names[:10]

        results = [
            make_pose_result(frame_index=0, keypoint_names=full_names),
            make_pose_result(frame_index=1, keypoint_names=partial_names),
            make_pose_result(frame_index=2, keypoint_names=full_names),
        ]

        tracked = tracker.track_across_frames(results)

        assert tracked[0].is_low_confidence is False
        assert tracked[1].is_low_confidence is True
        assert tracked[2].is_low_confidence is False

    def test_flag_when_occlusion_exceeds_5_frames(self):
        """Frames with >5 consecutive frames of occlusion should be flagged."""
        tracker = PoseTracker()

        full_names = ALL_KEYPOINT_NAMES
        # Missing 1 keypoint (below 40% ratio)
        partial_names = [n for n in full_names if n != "left_wrist"]

        # 7 consecutive frames with left_wrist missing (> 5 frame threshold)
        results = [make_pose_result(frame_index=0, keypoint_names=full_names)]
        for i in range(1, 8):
            results.append(make_pose_result(frame_index=i, keypoint_names=partial_names))
        results.append(make_pose_result(frame_index=8, keypoint_names=full_names))

        tracked = tracker.track_across_frames(results)

        # Frames 1-7 should be flagged as low-confidence (gap=7 > 5)
        assert tracked[0].is_low_confidence is False
        for i in range(1, 8):
            assert tracked[i].is_low_confidence is True
        assert tracked[8].is_low_confidence is False

    def test_no_flag_within_5_frames_and_below_40_percent(self):
        """Frames within thresholds should NOT be flagged low-confidence."""
        tracker = PoseTracker()

        full_names = ALL_KEYPOINT_NAMES
        # Missing 2 keypoints (2/17 ≈ 11.8% < 40%)
        partial_names = [n for n in full_names if n not in ("left_wrist", "right_wrist")]

        # 3 consecutive frames with missing keypoints (≤ 5)
        results = [
            make_pose_result(frame_index=0, keypoint_names=full_names),
            make_pose_result(frame_index=1, keypoint_names=partial_names),
            make_pose_result(frame_index=2, keypoint_names=partial_names),
            make_pose_result(frame_index=3, keypoint_names=partial_names),
            make_pose_result(frame_index=4, keypoint_names=full_names),
        ]

        tracked = tracker.track_across_frames(results)

        # None should be flagged (gap=3 ≤ 5, ratio < 40%)
        for result in tracked:
            assert result.is_low_confidence is False

    def test_flag_at_exactly_40_percent_occlusion(self):
        """Frames with exactly 40% occlusion should be flagged (≥40% threshold)."""
        tracker = PoseTracker()

        # Use 10 keypoints total, remove 4 (4/10 = 40% exactly)
        full_names = ALL_KEYPOINT_NAMES[:10]
        partial_names = full_names[:6]  # 4 missing out of 10 = 40%

        results = [
            make_pose_result(frame_index=0, keypoint_names=full_names),
            make_pose_result(frame_index=1, keypoint_names=partial_names),
            make_pose_result(frame_index=2, keypoint_names=full_names),
        ]

        tracked = tracker.track_across_frames(results)

        assert tracked[1].is_low_confidence is True


# --- Tests for mixed scenarios ---


class TestMixedScenarios:
    """Tests for combined tracking, interpolation, and flagging scenarios."""

    def test_tracking_with_gap_then_interpolation(self):
        """Identity reset followed by interpolation within new identity."""
        tracker = PoseTracker()

        full_names = ALL_KEYPOINT_NAMES
        partial_names = [n for n in full_names if n != "left_wrist"]

        results = [
            make_pose_result(frame_index=0, keypoint_names=full_names),
            # Large gap → identity reset
            make_pose_result(frame_index=10, keypoint_names=full_names),
            make_pose_result(frame_index=11, keypoint_names=partial_names),
            make_pose_result(frame_index=12, keypoint_names=full_names),
        ]

        tracked = tracker.track_across_frames(results)

        # First frame has different person_id
        assert tracked[0].person_id != tracked[1].person_id
        # Frames 10-12 share same person_id
        assert tracked[1].person_id == tracked[2].person_id == tracked[3].person_id
        # Frame 11 should have interpolated left_wrist
        frame11_names = {kp.name for kp in tracked[2].keypoints}
        assert "left_wrist" in frame11_names

    def test_partial_occlusion_then_full_occlusion(self):
        """Partial occlusion (interpolated) followed by heavy occlusion (flagged)."""
        tracker = PoseTracker()

        full_names = ALL_KEYPOINT_NAMES  # 17 keypoints
        # Light occlusion: 2 missing (11.8% < 40%)
        light_partial = [n for n in full_names if n not in ("left_wrist", "right_wrist")]
        # Heavy occlusion: 8 missing (8/17 ≈ 47% ≥ 40%)
        heavy_partial = full_names[:9]

        results = [
            make_pose_result(frame_index=0, keypoint_names=full_names),
            make_pose_result(frame_index=1, keypoint_names=light_partial),
            make_pose_result(frame_index=2, keypoint_names=full_names),
            make_pose_result(frame_index=3, keypoint_names=heavy_partial),
            make_pose_result(frame_index=4, keypoint_names=full_names),
        ]

        tracked = tracker.track_across_frames(results)

        # Frame 1: interpolated (light occlusion)
        frame1_names = {kp.name for kp in tracked[1].keypoints}
        assert "left_wrist" in frame1_names
        assert tracked[1].is_low_confidence is False

        # Frame 3: flagged (heavy occlusion)
        assert tracked[3].is_low_confidence is True

    def test_unsorted_input_is_handled(self):
        """Input frames in non-sequential order should be sorted and processed."""
        tracker = PoseTracker()

        results = [
            make_pose_result(frame_index=2),
            make_pose_result(frame_index=0),
            make_pose_result(frame_index=1),
        ]

        tracked = tracker.track_across_frames(results)

        # Should be sorted by frame_index
        assert tracked[0].frame_index == 0
        assert tracked[1].frame_index == 1
        assert tracked[2].frame_index == 2
        # All should have same person_id (consecutive)
        assert tracked[0].person_id == tracked[1].person_id == tracked[2].person_id

    def test_all_frames_fully_occluded(self):
        """All frames with heavy occlusion should all be flagged."""
        tracker = PoseTracker()

        full_names = ALL_KEYPOINT_NAMES  # 17 keypoints
        # Heavy occlusion on all frames
        heavy_partial = full_names[:9]  # 8 missing ≈ 47%

        results = [
            make_pose_result(frame_index=0, keypoint_names=full_names),
            make_pose_result(frame_index=1, keypoint_names=heavy_partial),
            make_pose_result(frame_index=2, keypoint_names=heavy_partial),
            make_pose_result(frame_index=3, keypoint_names=heavy_partial),
            make_pose_result(frame_index=4, keypoint_names=full_names),
        ]

        tracked = tracker.track_across_frames(results)

        assert tracked[0].is_low_confidence is False
        assert tracked[1].is_low_confidence is True
        assert tracked[2].is_low_confidence is True
        assert tracked[3].is_low_confidence is True
        assert tracked[4].is_low_confidence is False


# --- Tests for constants ---


class TestTrackerConstants:
    """Tests for tracker module constants."""

    def test_max_identity_gap_is_3(self):
        """MAX_IDENTITY_GAP_FRAMES should be 3 per requirement 3.2."""
        assert MAX_IDENTITY_GAP_FRAMES == 3

    def test_max_interpolation_gap_is_5(self):
        """MAX_INTERPOLATION_GAP_FRAMES should be 5 per requirement 3.3."""
        assert MAX_INTERPOLATION_GAP_FRAMES == 5

    def test_max_occlusion_ratio_is_0_4(self):
        """MAX_OCCLUSION_RATIO should be 0.4 per requirement 3.3/3.4."""
        assert MAX_OCCLUSION_RATIO == 0.4
