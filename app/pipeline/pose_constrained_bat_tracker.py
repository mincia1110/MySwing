"""Pose-constrained OpenCV bat line tracker.

This tracker uses MediaPipe pose keypoints to constrain where OpenCV should
look for bat-like line segments. WristBatEstimator remains the prior/fallback:
line detections are used when they are coherent enough, and wrist estimates
fill gaps without inflating tracking_accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.pose import Keypoint, PoseResult
from app.pipeline.wrist_bat_estimator import WristBatEstimator


@dataclass(frozen=True)
class BatLineCandidate:
    frame_index: int
    p1: tuple[float, float]
    p2: tuple[float, float]
    hand_anchor: tuple[float, float]
    center: tuple[float, float]
    bat_head: tuple[float, float]
    angle_deg: float
    length_px: float
    observation_score: float


class PoseConstrainedBatTracker:
    """Track bat line segments in a pose-constrained hand ROI."""

    def __init__(
        self,
        min_keypoint_confidence: float = 0.3,
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
        hough_threshold: int = 20,
        min_line_length_ratio: float = 0.12,
        max_line_gap_ratio: float = 0.04,
    ) -> None:
        self.min_keypoint_confidence = min_keypoint_confidence
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.hough_threshold = hough_threshold
        self.min_line_length_ratio = min_line_length_ratio
        self.max_line_gap_ratio = max_line_gap_ratio

    def track(
        self,
        frames: list[np.ndarray],
        pose_sequence: list[PoseResult],
        video_width: int,
        video_height: int,
        fps: float,
        bat_length_normalized: float = 0.25,
        wrist_prior: BatTrajectory | None = None,
    ) -> BatTrajectory:
        """Track a bat trajectory with line observations and wrist fallback."""
        del fps  # Reserved for future temporal plausibility thresholds.

        if not frames:
            return wrist_prior if wrist_prior is not None else BatTrajectory()

        if not pose_sequence:
            return wrist_prior if wrist_prior is not None else self._empty_for_frames(len(frames))

        width = max(int(video_width), 1)
        height = max(int(video_height), 1)
        expected_length_px = max(8.0, float(bat_length_normalized) * height)

        pose_by_frame = {pose.frame_index: pose for pose in pose_sequence}
        top_candidates: list[list[BatLineCandidate]] = []
        for frame_index, frame in enumerate(frames):
            pose = pose_by_frame.get(frame_index)
            if pose is None:
                top_candidates.append([])
                continue
            candidates = self._candidates_for_frame(
                frame_index=frame_index,
                frame=frame,
                previous_frame=frames[frame_index - 1] if frame_index > 0 else None,
                next_frame=frames[frame_index + 1] if frame_index + 1 < len(frames) else None,
                pose=pose,
                video_width=width,
                video_height=height,
                expected_length_px=expected_length_px,
                wrist_prior=wrist_prior,
            )
            top_candidates.append(
                sorted(candidates, key=lambda c: c.observation_score, reverse=True)[:5]
            )

        selected = self._select_temporally(top_candidates)
        line_detection_count = sum(1 for candidate in selected if candidate is not None)
        line_coverage = line_detection_count / len(frames) if frames else 0.0

        # A single-frame unit clip can be accepted with one strong observation.
        enough_line_signal = (
            line_detection_count >= 1
            if len(frames) <= 2
            else line_detection_count >= 2 and line_coverage >= 0.30
        )
        if not enough_line_signal and wrist_prior is not None:
            return wrist_prior

        detections: list[BatDetectionResult] = []
        for frame_index, candidate in enumerate(selected):
            if candidate is not None:
                detections.append(self._detection_from_candidate(candidate))
                continue

            prior = self._prior_detection(wrist_prior, frame_index)
            if prior is not None:
                detections.append(self._predicted_copy(prior))
            else:
                detections.append(self._no_detection(frame_index))

        # tracking_accuracy reflects OpenCV line observations only. Wrist-filled
        # frames are useful continuity priors but are not counted as tracked.
        trajectory = self._build_trajectory(detections)
        trajectory.tracking_accuracy = line_coverage
        return trajectory

    def _candidates_for_frame(
        self,
        frame_index: int,
        frame: np.ndarray,
        previous_frame: np.ndarray | None,
        next_frame: np.ndarray | None,
        pose: PoseResult,
        video_width: int,
        video_height: int,
        expected_length_px: float,
        wrist_prior: BatTrajectory | None,
    ) -> list[BatLineCandidate]:
        hand = self._hand_anchor(pose)
        if hand is None:
            return []

        hand_px = (hand[0] * video_width, hand[1] * video_height)
        roi = self._hand_roi(hand_px, expected_length_px, video_width, video_height)
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            return []

        frame_roi = self._normalize_frame(frame)[y1:y2, x1:x2]
        prev_roi = (
            self._normalize_frame(previous_frame)[y1:y2, x1:x2]
            if previous_frame is not None
            else None
        )
        next_roi = (
            self._normalize_frame(next_frame)[y1:y2, x1:x2]
            if next_frame is not None
            else None
        )

        gray = self._gray(frame_roi)
        motion_mask = self._motion_mask(gray, prev_roi, next_roi)
        segments = self._line_segments(gray, expected_length_px, video_height)
        prior_angle = self._prior_angle(wrist_prior, frame_index)
        wrist_angle = self._wrist_angle(pose, video_width, video_height)
        angle_prior = prior_angle if prior_angle is not None else wrist_angle

        candidates: list[BatLineCandidate] = []
        for x_a, y_a, x_b, y_b, support in segments:
            p1_px = (x1 + x_a, y1 + y_a)
            p2_px = (x1 + x_b, y1 + y_b)
            candidate = self._score_segment(
                frame_index=frame_index,
                p1_px=p1_px,
                p2_px=p2_px,
                hand_px=hand_px,
                hand_norm=hand,
                expected_length_px=expected_length_px,
                video_width=video_width,
                video_height=video_height,
                angle_prior=angle_prior,
                motion_mask=motion_mask,
                roi_origin=(x1, y1),
                edge_support=support,
                pose=pose,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _line_segments(
        self,
        gray: np.ndarray,
        expected_length_px: float,
        video_height: int,
    ) -> list[tuple[float, float, float, float, float]]:
        min_len = max(6, int(self.min_line_length_ratio * video_height))
        max_gap = max(2, int(self.max_line_gap_ratio * video_height))
        try:
            import cv2

            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=self.hough_threshold,
                minLineLength=min_len,
                maxLineGap=max_gap,
            )
            if lines is None:
                return []
            segments = []
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = [float(v) for v in line]
                support = self._sample_support(edges, (x1, y1), (x2, y2)) / 255.0
                segments.append((x1, y1, x2, y2, support))
            return segments
        except ImportError:
            return self._numpy_line_segments(
                gray,
                min_len=min_len,
                expected_length_px=expected_length_px,
            )

    def _numpy_line_segments(
        self,
        gray: np.ndarray,
        min_len: int,
        expected_length_px: float,
    ) -> list[tuple[float, float, float, float, float]]:
        if gray.size == 0:
            return []
        threshold = max(40.0, float(gray.mean()) + 2.0 * float(gray.std()))
        mask = gray >= threshold
        components = self._connected_components(mask)
        segments: list[tuple[float, float, float, float, float]] = []
        for points in components:
            if len(points) < min_len:
                continue
            ys = points[:, 0].astype(float)
            xs = points[:, 1].astype(float)
            coords = np.column_stack((xs, ys))
            center = coords.mean(axis=0)
            _, _, vh = np.linalg.svd(coords - center, full_matrices=False)
            direction = vh[0]
            projections = (coords - center) @ direction
            p1 = center + direction * projections.min()
            p2 = center + direction * projections.max()
            length = float(np.linalg.norm(p2 - p1))
            if length < min_len or length < expected_length_px * 0.35:
                continue
            support = min(1.0, len(points) / max(length, 1.0))
            segments.append((float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), support))
        return segments

    def _connected_components(self, mask: np.ndarray) -> list[np.ndarray]:
        visited = np.zeros(mask.shape, dtype=bool)
        components: list[np.ndarray] = []
        height, width = mask.shape
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                points = []
                while stack:
                    cy, cx = stack.pop()
                    points.append((cy, cx))
                    for ny in range(max(0, cy - 1), min(height, cy + 2)):
                        for nx in range(max(0, cx - 1), min(width, cx + 2)):
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                components.append(np.array(points, dtype=int))
        return components

    def _score_segment(
        self,
        frame_index: int,
        p1_px: tuple[float, float],
        p2_px: tuple[float, float],
        hand_px: tuple[float, float],
        hand_norm: tuple[float, float],
        expected_length_px: float,
        video_width: int,
        video_height: int,
        angle_prior: float | None,
        motion_mask: np.ndarray | None,
        roi_origin: tuple[int, int],
        edge_support: float,
        pose: PoseResult,
    ) -> BatLineCandidate | None:
        length = self._distance(p1_px, p2_px)
        if length < expected_length_px * 0.35:
            return None

        d1 = self._distance(p1_px, hand_px)
        d2 = self._distance(p2_px, hand_px)
        hand_side = p1_px if d1 <= d2 else p2_px
        head_px = p2_px if d1 <= d2 else p1_px

        endpoint_score = max(0.0, 1.0 - min(d1, d2) / max(expected_length_px * 0.55, 1.0))
        length_score = math.exp(
            -abs(length - expected_length_px) / max(expected_length_px * 0.75, 1.0)
        )
        angle = self._angle_deg(hand_side, head_px)
        angle_score = 0.5
        if angle_prior is not None:
            angle_score = max(0.0, 1.0 - self._angle_delta(angle, angle_prior) / 90.0)
        motion_score = self._motion_support(motion_mask, hand_side, head_px, roi_origin)
        torso_penalty = self._torso_penalty(head_px, pose, video_width, video_height)

        score = (
            0.30 * endpoint_score
            + 0.25 * length_score
            + 0.20 * angle_score
            + 0.15 * motion_score
            + 0.10 * max(0.0, min(edge_support, 1.0))
        ) * torso_penalty

        if score < 0.25:
            return None

        head_norm = (head_px[0] / video_width, head_px[1] / video_height)
        hand_side_norm = (hand_side[0] / video_width, hand_side[1] / video_height)
        center = (
            (hand_side_norm[0] + head_norm[0]) / 2.0,
            (hand_side_norm[1] + head_norm[1]) / 2.0,
        )
        return BatLineCandidate(
            frame_index=frame_index,
            p1=(p1_px[0] / video_width, p1_px[1] / video_height),
            p2=(p2_px[0] / video_width, p2_px[1] / video_height),
            hand_anchor=hand_norm,
            center=center,
            bat_head=head_norm,
            angle_deg=angle,
            length_px=length,
            observation_score=float(max(0.0, min(score, 1.0))),
        )

    def _select_temporally(
        self, frame_candidates: list[list[BatLineCandidate]]
    ) -> list[BatLineCandidate | None]:
        if not frame_candidates:
            return []

        states: list[list[BatLineCandidate | None]] = [
            candidates if candidates else [None] for candidates in frame_candidates
        ]
        scores: list[list[float]] = []
        parents: list[list[int]] = []
        for frame_idx, candidates in enumerate(states):
            frame_scores: list[float] = []
            frame_parents: list[int] = []
            for candidate in candidates:
                observation = candidate.observation_score if candidate is not None else 0.0
                if frame_idx == 0:
                    frame_scores.append(observation)
                    frame_parents.append(-1)
                    continue
                best_prev_score = -1e9
                best_prev_index = 0
                for prev_idx, prev in enumerate(states[frame_idx - 1]):
                    transition = self._transition_cost(prev, candidate)
                    score = scores[frame_idx - 1][prev_idx] + observation - transition
                    if score > best_prev_score:
                        best_prev_score = score
                        best_prev_index = prev_idx
                frame_scores.append(best_prev_score)
                frame_parents.append(best_prev_index)
            scores.append(frame_scores)
            parents.append(frame_parents)

        idx = max(range(len(scores[-1])), key=lambda i: scores[-1][i])
        selected: list[BatLineCandidate | None] = [None] * len(states)
        for frame_idx in range(len(states) - 1, -1, -1):
            selected[frame_idx] = states[frame_idx][idx]
            idx = parents[frame_idx][idx]
            if idx < 0 and frame_idx > 0:
                idx = 0
        return selected

    def _transition_cost(
        self,
        previous: BatLineCandidate | None,
        current: BatLineCandidate | None,
    ) -> float:
        if previous is None or current is None:
            return 0.08
        head_jump = self._distance(previous.bat_head, current.bat_head)
        angle_jump = self._angle_delta(previous.angle_deg, current.angle_deg) / 180.0
        length_jump = abs(previous.length_px - current.length_px) / max(
            previous.length_px,
            current.length_px,
            1.0,
        )
        return 0.75 * min(head_jump / 0.25, 1.5) + 0.25 * angle_jump + 0.20 * length_jump

    def _hand_anchor(self, pose: PoseResult) -> tuple[float, float] | None:
        wrists = [
            kp for kp in pose.keypoints
            if kp.name in ("left_wrist", "right_wrist")
            and kp.confidence >= self.min_keypoint_confidence
        ]
        if not wrists:
            return None
        if len(wrists) == 1:
            return wrists[0].x, wrists[0].y
        return sum(kp.x for kp in wrists) / len(wrists), sum(kp.y for kp in wrists) / len(wrists)

    def _wrist_angle(self, pose: PoseResult, video_width: int, video_height: int) -> float | None:
        for side in ("right", "left"):
            wrist = self._find_keypoint(pose.keypoints, f"{side}_wrist")
            elbow = self._find_keypoint(pose.keypoints, f"{side}_elbow")
            if wrist is None or elbow is None:
                continue
            dx = (wrist.x - elbow.x) * video_width
            dy = (wrist.y - elbow.y) * video_height
            if abs(dx) + abs(dy) > 1e-6:
                return self._normalize_angle(math.degrees(math.atan2(dy, dx)))
        return None

    def _find_keypoint(self, keypoints: Sequence[Keypoint], name: str) -> Keypoint | None:
        for kp in keypoints:
            if kp.name == name and kp.confidence >= self.min_keypoint_confidence:
                return kp
        return None

    def _hand_roi(
        self,
        hand_px: tuple[float, float],
        expected_length_px: float,
        video_width: int,
        video_height: int,
    ) -> tuple[int, int, int, int]:
        radius = int(max(expected_length_px * 1.4, 32.0))
        cx, cy = hand_px
        return (
            max(0, int(cx - radius)),
            max(0, int(cy - radius)),
            min(video_width, int(cx + radius)),
            min(video_height, int(cy + radius)),
        )

    def _motion_mask(
        self,
        gray: np.ndarray,
        previous_roi: np.ndarray | None,
        next_roi: np.ndarray | None,
    ) -> np.ndarray | None:
        masks = []
        try:
            import cv2
        except ImportError:
            cv2 = None
        for other in (previous_roi, next_roi):
            if other is None or other.size == 0:
                continue
            other_gray = self._gray(other)
            if other_gray.shape != gray.shape:
                continue
            diff = np.abs(gray.astype(np.int16) - other_gray.astype(np.int16)).astype(np.uint8)
            if cv2 is not None:
                _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            else:
                mask = (diff > 20).astype(np.uint8) * 255
            masks.append(mask)
        if not masks:
            return None
        return np.maximum.reduce(masks)

    def _motion_support(
        self,
        motion_mask: np.ndarray | None,
        p1_px: tuple[float, float],
        p2_px: tuple[float, float],
        roi_origin: tuple[int, int],
    ) -> float:
        if motion_mask is None:
            return 0.5
        local_p1 = (p1_px[0] - roi_origin[0], p1_px[1] - roi_origin[1])
        local_p2 = (p2_px[0] - roi_origin[0], p2_px[1] - roi_origin[1])
        return self._sample_support(motion_mask, local_p1, local_p2) / 255.0

    def _sample_support(
        self,
        image: np.ndarray,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        x1, y1 = p1
        x2, y2 = p2
        steps = max(2, int(math.hypot(x2 - x1, y2 - y1)))
        xs = np.linspace(x1, x2, steps).round().astype(int)
        ys = np.linspace(y1, y2, steps).round().astype(int)
        valid = (xs >= 0) & (xs < image.shape[1]) & (ys >= 0) & (ys < image.shape[0])
        if not valid.any():
            return 0.0
        return float(image[ys[valid], xs[valid]].mean())

    def _torso_penalty(
        self,
        head_px: tuple[float, float],
        pose: PoseResult,
        video_width: int,
        video_height: int,
    ) -> float:
        torso_names = {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}
        torso = [
            (kp.x * video_width, kp.y * video_height)
            for kp in pose.keypoints
            if kp.name in torso_names and kp.confidence >= self.min_keypoint_confidence
        ]
        if len(torso) < 2:
            return 1.0
        min_x = min(p[0] for p in torso)
        max_x = max(p[0] for p in torso)
        min_y = min(p[1] for p in torso)
        max_y = max(p[1] for p in torso)
        pad = 0.04 * video_height
        x, y = head_px
        if min_x - pad <= x <= max_x + pad and min_y - pad <= y <= max_y + pad:
            return 0.75
        return 1.0

    def _prior_angle(self, wrist_prior: BatTrajectory | None, frame_index: int) -> float | None:
        prior = self._prior_detection(wrist_prior, frame_index)
        if prior is None or not prior.detected:
            return None
        return prior.orientation_angle

    def _prior_detection(
        self, wrist_prior: BatTrajectory | None, frame_index: int
    ) -> BatDetectionResult | None:
        if wrist_prior is None:
            return None
        for detection in wrist_prior.detections:
            if detection.frame_index == frame_index:
                return detection
        return None

    def _detection_from_candidate(self, candidate: BatLineCandidate) -> BatDetectionResult:
        return BatDetectionResult(
            frame_index=candidate.frame_index,
            detected=True,
            position=candidate.center,
            orientation_angle=candidate.angle_deg,
            length_pixels=candidate.length_px,
            confidence=candidate.observation_score,
            is_predicted=False,
            coordinate_space="normalized",
            bat_head_position=candidate.bat_head,
        )

    def _predicted_copy(self, prior: BatDetectionResult) -> BatDetectionResult:
        return BatDetectionResult(
            frame_index=prior.frame_index,
            detected=prior.detected,
            position=prior.position,
            orientation_angle=prior.orientation_angle,
            length_pixels=prior.length_pixels,
            confidence=prior.confidence,
            is_predicted=True,
            coordinate_space=prior.coordinate_space,
            bat_head_position=prior.bat_head_position,
        )

    def _no_detection(self, frame_index: int) -> BatDetectionResult:
        return BatDetectionResult(
            frame_index=frame_index,
            detected=False,
            position=(0.0, 0.0),
            orientation_angle=0.0,
            length_pixels=0.0,
            confidence=0.0,
            is_predicted=False,
            coordinate_space="normalized",
        )

    def _empty_for_frames(self, count: int) -> BatTrajectory:
        return self._build_trajectory([self._no_detection(i) for i in range(count)])

    def _build_trajectory(self, detections: list[BatDetectionResult]) -> BatTrajectory:
        speeds = [
            WristBatEstimator._calculate_speed(detections[i - 1], detections[i])
            for i in range(1, len(detections))
        ]
        return BatTrajectory(
            detections=detections,
            bat_speed_pixels_per_frame=speeds,
            tracking_accuracy=(
                sum(1 for d in detections if d.detected and not d.is_predicted) / len(detections)
                if detections else 0.0
            ),
            tracking_failures=WristBatEstimator._identify_tracking_failures(detections),
        )

    def _normalize_frame(self, frame: np.ndarray | None) -> np.ndarray:
        if frame is None:
            return np.empty((0, 0, 3), dtype=np.uint8)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            return frame
        if frame.shape[2] >= 3:
            return frame[:, :, :3]
        return frame

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        try:
            import cv2

            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except ImportError:
            return np.mean(frame[:, :, :3], axis=2).astype(np.uint8)

    def _angle_deg(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return self._normalize_angle(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))

    def _normalize_angle(self, angle: float) -> float:
        angle = angle % 360.0
        return angle if angle >= 0 else angle + 360.0

    def _angle_delta(self, angle_a: float, angle_b: float) -> float:
        return abs((angle_a - angle_b + 180.0) % 360.0 - 180.0)

    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
