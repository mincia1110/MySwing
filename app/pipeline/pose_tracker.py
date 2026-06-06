"""Multi-frame pose tracking, interpolation, and confidence flagging module.

This module provides a PoseTracker class that handles:
- Tracking keypoint identity across consecutive frames (person_id maintenance)
- Interpolating occluded keypoints from adjacent frame data
- Flagging low-confidence frames when occlusion exceeds thresholds

Requirements:
- 3.2: Track keypoints across frames, maintain identity with ≤3 frame gap
- 3.3: Interpolate occluded keypoints (≤5 frames, <40% occluded)
- 3.4: Flag low-confidence frames (>5 frames or ≥40% occluded)
"""

from __future__ import annotations

import logging
from dataclasses import replace

from app.models.pose import Keypoint, PoseResult

logger = logging.getLogger(__name__)

# Tracking constants
MAX_IDENTITY_GAP_FRAMES: int = 3
MAX_INTERPOLATION_GAP_FRAMES: int = 5
MAX_OCCLUSION_RATIO: float = 0.4


class PoseTracker:
    """Multi-frame pose tracker with interpolation and confidence flagging.

    Tracks a person's identity across frames, interpolates missing keypoints
    when occlusion is within acceptable bounds, and flags frames as
    low-confidence when thresholds are exceeded.

    Args:
        max_identity_gap: Maximum consecutive frames of tracking loss before
            re-detection is triggered. Defaults to 3.
        max_interpolation_gap: Maximum consecutive occluded frames for which
            interpolation is applied. Defaults to 5.
        max_occlusion_ratio: Maximum ratio of occluded keypoints for
            interpolation to be applied. Defaults to 0.4.
    """

    def __init__(
        self,
        max_identity_gap: int = MAX_IDENTITY_GAP_FRAMES,
        max_interpolation_gap: int = MAX_INTERPOLATION_GAP_FRAMES,
        max_occlusion_ratio: float = MAX_OCCLUSION_RATIO,
    ) -> None:
        self.max_identity_gap = max_identity_gap
        self.max_interpolation_gap = max_interpolation_gap
        self.max_occlusion_ratio = max_occlusion_ratio
        self._next_person_id: int = 0

    def _generate_person_id(self) -> int:
        """Generate a new unique person_id."""
        person_id = self._next_person_id
        self._next_person_id += 1
        return person_id

    def track_across_frames(
        self, pose_results: list[PoseResult]
    ) -> list[PoseResult]:
        """Process a sequence of pose results maintaining identity across frames.

        Assigns consistent person_id values across frames. If tracking is lost
        for more than max_identity_gap consecutive frames, a new person_id is
        assigned (re-detection).

        Args:
            pose_results: List of PoseResult objects ordered by frame_index.

        Returns:
            List of PoseResult objects with consistent person_id assignments,
            interpolated keypoints where applicable, and low-confidence flags set.
        """
        if not pose_results:
            return []

        # Sort by frame_index to ensure correct ordering
        sorted_results = sorted(pose_results, key=lambda r: r.frame_index)

        # Phase 1: Assign person IDs based on tracking gaps
        tracked_results = self._assign_person_ids(sorted_results)

        # Phase 2: Interpolate occluded keypoints
        interpolated_results = self._interpolate_occluded(tracked_results)

        # Phase 3: Flag low-confidence frames
        flagged_results = self._apply_low_confidence_flags(interpolated_results)

        return flagged_results

    def _assign_person_ids(
        self, sorted_results: list[PoseResult]
    ) -> list[PoseResult]:
        """Assign person IDs based on frame gaps.

        Maintains identity when gap ≤ max_identity_gap frames.
        Assigns new person_id when gap > max_identity_gap frames.
        """
        if not sorted_results:
            return []

        current_person_id = self._generate_person_id()
        result_list: list[PoseResult] = []

        prev_frame_index: int | None = None

        for pose_result in sorted_results:
            if prev_frame_index is not None:
                gap = pose_result.frame_index - prev_frame_index - 1

                if not self._maintain_identity(
                    pose_result, result_list[-1], gap
                ):
                    # Gap exceeds threshold: re-detection, assign new ID
                    current_person_id = self._generate_person_id()

            updated_result = replace(pose_result, person_id=current_person_id)
            result_list.append(updated_result)
            prev_frame_index = pose_result.frame_index

        return result_list

    def _maintain_identity(
        self,
        current: PoseResult,
        previous: PoseResult,
        gap_frames: int,
    ) -> bool:
        """Determine if the current detection is the same person as previous.

        Identity is maintained if the gap between detections is within
        the allowed threshold (≤ max_identity_gap consecutive frames).

        Args:
            current: Current frame's pose result.
            previous: Previous frame's pose result.
            gap_frames: Number of frames between previous and current detection
                where tracking was lost (exclusive of both endpoints).

        Returns:
            True if identity should be maintained, False if re-detection needed.
        """
        return gap_frames <= self.max_identity_gap

    def _interpolate_occluded(
        self,
        pose_sequence: list[PoseResult],
        max_gap: int | None = None,
        max_occlusion_ratio: float | None = None,
    ) -> list[PoseResult]:
        """Interpolate occluded keypoints using adjacent frame data.

        For each frame, if keypoints are missing (occluded) and the occlusion
        is within acceptable bounds (≤ max_gap consecutive frames AND
        < max_occlusion_ratio of keypoints missing), linearly interpolate
        from the nearest frames that have those keypoints.

        Args:
            pose_sequence: List of PoseResult objects with assigned person_ids.
            max_gap: Maximum consecutive frames of occlusion for interpolation.
                Defaults to self.max_interpolation_gap.
            max_occlusion_ratio: Maximum ratio of missing keypoints for
                interpolation. Defaults to self.max_occlusion_ratio.

        Returns:
            List of PoseResult objects with interpolated keypoints where applicable.
        """
        if max_gap is None:
            max_gap = self.max_interpolation_gap
        if max_occlusion_ratio is None:
            max_occlusion_ratio = self.max_occlusion_ratio

        if len(pose_sequence) < 2:
            return list(pose_sequence)

        # Group by person_id for interpolation within same identity
        person_groups: dict[int, list[tuple[int, PoseResult]]] = {}
        for idx, result in enumerate(pose_sequence):
            if result.person_id not in person_groups:
                person_groups[result.person_id] = []
            person_groups[result.person_id].append((idx, result))

        interpolated = list(pose_sequence)

        for person_id, group in person_groups.items():
            if len(group) < 2:
                continue

            # Collect all keypoint names that appear in this person's sequence
            all_keypoint_names: set[str] = set()
            for _, result in group:
                for kp in result.keypoints:
                    all_keypoint_names.add(kp.name)

            # For each frame in the group, check for missing keypoints
            for group_idx in range(len(group)):
                seq_idx, result = group[group_idx]
                present_names = {kp.name for kp in result.keypoints}
                missing_names = all_keypoint_names - present_names

                if not missing_names:
                    continue

                # Calculate occlusion ratio
                total_expected = len(all_keypoint_names)
                if total_expected == 0:
                    continue
                occlusion_ratio = len(missing_names) / total_expected

                # Check if occlusion ratio exceeds threshold
                if occlusion_ratio >= max_occlusion_ratio:
                    continue

                # Find the consecutive occlusion gap for this frame
                occlusion_gap = self._calculate_occlusion_gap(
                    group, group_idx, missing_names
                )

                if occlusion_gap > max_gap:
                    continue

                # Interpolate missing keypoints from adjacent frames
                interpolated_keypoints = self._linear_interpolate_keypoints(
                    group, group_idx, missing_names
                )

                if interpolated_keypoints:
                    new_keypoints = list(result.keypoints) + interpolated_keypoints
                    new_confidence = (
                        sum(kp.confidence for kp in new_keypoints) / len(new_keypoints)
                        if new_keypoints
                        else result.overall_confidence
                    )
                    interpolated[seq_idx] = replace(
                        result,
                        keypoints=new_keypoints,
                        overall_confidence=new_confidence,
                    )

        return interpolated

    def _calculate_occlusion_gap(
        self,
        group: list[tuple[int, PoseResult]],
        current_group_idx: int,
        missing_names: set[str],
    ) -> int:
        """Calculate the consecutive occlusion gap for missing keypoints.

        Counts how many consecutive frames (including current) have the same
        keypoints missing.

        Returns:
            Number of consecutive frames with these keypoints occluded.
        """
        gap = 1  # Current frame counts

        # Look backward
        for i in range(current_group_idx - 1, -1, -1):
            _, prev_result = group[i]
            prev_present = {kp.name for kp in prev_result.keypoints}
            if missing_names - prev_present:  # Still missing some
                gap += 1
            else:
                break

        # Look forward
        for i in range(current_group_idx + 1, len(group)):
            _, next_result = group[i]
            next_present = {kp.name for kp in next_result.keypoints}
            if missing_names - next_present:  # Still missing some
                gap += 1
            else:
                break

        return gap

    def _linear_interpolate_keypoints(
        self,
        group: list[tuple[int, PoseResult]],
        current_group_idx: int,
        missing_names: set[str],
    ) -> list[Keypoint]:
        """Linearly interpolate missing keypoints from adjacent frames.

        Finds the nearest previous and next frames that have the missing
        keypoints and performs linear interpolation based on frame distance.

        Returns:
            List of interpolated Keypoint objects.
        """
        interpolated: list[Keypoint] = []
        current_frame_index = group[current_group_idx][1].frame_index

        for name in missing_names:
            # Find nearest previous frame with this keypoint
            prev_kp: Keypoint | None = None
            prev_frame_idx: int | None = None
            for i in range(current_group_idx - 1, -1, -1):
                _, prev_result = group[i]
                kp = next(
                    (k for k in prev_result.keypoints if k.name == name), None
                )
                if kp is not None:
                    prev_kp = kp
                    prev_frame_idx = prev_result.frame_index
                    break

            # Find nearest next frame with this keypoint
            next_kp: Keypoint | None = None
            next_frame_idx: int | None = None
            for i in range(current_group_idx + 1, len(group)):
                _, next_result = group[i]
                kp = next(
                    (k for k in next_result.keypoints if k.name == name), None
                )
                if kp is not None:
                    next_kp = kp
                    next_frame_idx = next_result.frame_index
                    break

            # Interpolate if we have both endpoints
            if prev_kp is not None and next_kp is not None:
                assert prev_frame_idx is not None
                assert next_frame_idx is not None
                total_dist = next_frame_idx - prev_frame_idx
                if total_dist > 0:
                    t = (current_frame_index - prev_frame_idx) / total_dist
                    interpolated.append(
                        Keypoint(
                            x=prev_kp.x + t * (next_kp.x - prev_kp.x),
                            y=prev_kp.y + t * (next_kp.y - prev_kp.y),
                            z=prev_kp.z + t * (next_kp.z - prev_kp.z),
                            confidence=prev_kp.confidence
                            + t * (next_kp.confidence - prev_kp.confidence),
                            name=name,
                        )
                    )
            elif prev_kp is not None:
                # Only previous available: use last known position with reduced confidence
                interpolated.append(
                    replace(prev_kp, confidence=prev_kp.confidence * 0.5)
                )
            elif next_kp is not None:
                # Only next available: use next position with reduced confidence
                interpolated.append(
                    replace(next_kp, confidence=next_kp.confidence * 0.5)
                )

        return interpolated

    def _apply_low_confidence_flags(
        self, pose_sequence: list[PoseResult]
    ) -> list[PoseResult]:
        """Apply low-confidence flags to frames exceeding occlusion thresholds.

        Flags frames where:
        - Occlusion persists beyond max_interpolation_gap consecutive frames, OR
        - More than max_occlusion_ratio of keypoints are occluded

        Args:
            pose_sequence: List of PoseResult objects.

        Returns:
            List of PoseResult objects with is_low_confidence flags set.
        """
        if not pose_sequence:
            return []

        # Group by person_id
        person_groups: dict[int, list[tuple[int, PoseResult]]] = {}
        for idx, result in enumerate(pose_sequence):
            if result.person_id not in person_groups:
                person_groups[result.person_id] = []
            person_groups[result.person_id].append((idx, result))

        flagged = list(pose_sequence)

        for person_id, group in person_groups.items():
            if len(group) < 2:
                continue

            # Determine expected keypoint count from the group
            all_keypoint_names: set[str] = set()
            for _, result in group:
                for kp in result.keypoints:
                    all_keypoint_names.add(kp.name)

            total_expected = len(all_keypoint_names)
            if total_expected == 0:
                continue

            for group_idx in range(len(group)):
                seq_idx, result = group[group_idx]
                present_names = {kp.name for kp in result.keypoints}
                missing_names = all_keypoint_names - present_names
                occlusion_ratio = len(missing_names) / total_expected

                # Check occlusion ratio threshold
                if occlusion_ratio >= self.max_occlusion_ratio:
                    flagged[seq_idx] = self._flag_low_confidence(
                        result, 0, occlusion_ratio
                    )
                    continue

                # Check consecutive occlusion gap
                if missing_names:
                    occlusion_gap = self._calculate_occlusion_gap(
                        group, group_idx, missing_names
                    )
                    if occlusion_gap > self.max_interpolation_gap:
                        flagged[seq_idx] = self._flag_low_confidence(
                            result, occlusion_gap, occlusion_ratio
                        )

        return flagged

    def _flag_low_confidence(
        self,
        pose_result: PoseResult,
        occlusion_frames: int,
        occlusion_ratio: float,
    ) -> PoseResult:
        """Flag a frame as low-confidence due to excessive occlusion.

        Sets is_low_confidence=True on the PoseResult, indicating it should
        be excluded from downstream analysis.

        Args:
            pose_result: The PoseResult to flag.
            occlusion_frames: Number of consecutive frames with occlusion.
            occlusion_ratio: Ratio of occluded keypoints (0.0 to 1.0).

        Returns:
            Updated PoseResult with is_low_confidence=True.
        """
        logger.debug(
            f"Frame {pose_result.frame_index} flagged as low-confidence: "
            f"occlusion_frames={occlusion_frames}, "
            f"occlusion_ratio={occlusion_ratio:.2f}"
        )
        return replace(pose_result, is_low_confidence=True)
