"""RTMPose ONNX pose estimation backend.

The backend intentionally uses the lightweight ``rtmlib`` wrapper instead of
the full MMPose/MMCV stack.  This keeps inference available on Apple Silicon
through ONNX Runtime CPU while using the official OpenMMLab RTMPose models.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from app.models.pose import Keypoint, PoseResult
from app.pipeline.constants import DEFAULT_MIN_CONFIDENCE
from app.pipeline.pose_estimator import _append_keypoint_aliases

logger = logging.getLogger(__name__)

try:
    from rtmlib import Body as RTMLibBody

    RTMLIB_AVAILABLE = True
except ImportError:
    RTMLibBody = None  # type: ignore[assignment]
    RTMLIB_AVAILABLE = False


RTMPOSE_MODES = ("lightweight", "balanced", "performance")

# RTMLib's Body presets use the COCO 17-keypoint output order.
COCO_TO_ESSENTIAL: dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

PRIMARY_PERSON_JOINTS = tuple(range(5, 17))


class RTMPoseEstimator:
    """Estimate one primary batter pose with RTMPose ONNX models.

    ``mode`` selects RTMLib's official OpenMMLab detector/pose pair:
    lightweight (RTMPose-s), balanced (RTMPose-m), or performance
    (RTMPose-x at 384x288).  Model artifacts are cached by RTMLib outside the
    repository and are never written into user-owned video/output folders.
    """

    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        mode: str = "balanced",
        inference_backend: str = "onnxruntime",
        device: str = "cpu",
        detector_model: str | None = None,
        pose_model: str | None = None,
        solution_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if mode not in RTMPOSE_MODES:
            raise ValueError(
                f"unknown RTMPose mode {mode!r}; choose from {RTMPOSE_MODES}"
            )

        self.min_confidence = float(min_confidence)
        self.mode = mode
        self.inference_backend = inference_backend
        self.device = device
        self._solution: Any | None = None
        self._previous_bbox: np.ndarray | None = None

        factory = solution_factory
        if factory is None:
            if not RTMLIB_AVAILABLE:
                logger.warning(
                    "RTMPose is unavailable. Install with: pip install '.[rtmpose]'"
                )
                return
            factory = RTMLibBody

        kwargs: dict[str, Any] = {
            "mode": mode,
            "backend": inference_backend,
            "device": device,
            "to_openpose": False,
        }
        if detector_model is not None:
            kwargs["det"] = detector_model
        if pose_model is not None:
            kwargs["pose"] = pose_model
        self._solution = factory(**kwargs)

    @property
    def is_available(self) -> bool:
        return self._solution is not None

    def detect_keypoints(self, frame: np.ndarray) -> PoseResult:
        return self.process_frame(frame, frame_index=0)

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> PoseResult:
        if not self.is_available:
            raise RuntimeError(
                "RTMPose is not available. Install with: pip install '.[rtmpose]'"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (height, width, 3)")

        started = time.perf_counter()
        coordinates, scores = self._infer_primary_person(frame)
        keypoints = self._extract_primary_keypoints(
            coordinates,
            scores,
            frame_width=int(frame.shape[1]),
            frame_height=int(frame.shape[0]),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms > 100.0:
            logger.debug(
                "RTMPose frame %s took %.1fms on %s/%s",
                frame_index,
                elapsed_ms,
                self.inference_backend,
                self.device,
            )

        overall_confidence = (
            sum(keypoint.confidence for keypoint in keypoints) / len(keypoints)
            if keypoints
            else 0.0
        )
        return PoseResult(
            frame_index=frame_index,
            keypoints=keypoints,
            person_id=0,
            is_primary_batter=True,
            overall_confidence=overall_confidence,
            is_low_confidence=overall_confidence < self.min_confidence,
        )

    def _infer_primary_person(self, frame: np.ndarray) -> tuple[Any, Any]:
        """Run RTMPose on only one batter instead of every detected spectator."""
        detector = getattr(self._solution, "det_model", None)
        pose_model = getattr(self._solution, "pose_model", None)
        if detector is None or pose_model is None:
            return self._solution(frame)

        boxes = np.asarray(detector(frame), dtype=np.float64)
        if boxes.size == 0:
            return np.empty((0, len(COCO_TO_ESSENTIAL), 2)), np.empty(
                (0, len(COCO_TO_ESSENTIAL))
            )
        if boxes.ndim == 1:
            boxes = boxes[np.newaxis, ...]
        if boxes.ndim != 2 or boxes.shape[1] < 4:
            raise ValueError("RTMPose detector output must have shape (people, 4+)")

        primary_bbox = self._select_primary_bbox(
            boxes[:, :4],
            frame_width=int(frame.shape[1]),
            frame_height=int(frame.shape[0]),
        )
        self._previous_bbox = primary_bbox.copy()
        return pose_model(frame, bboxes=[primary_bbox.tolist()])

    def _select_primary_bbox(
        self,
        boxes: np.ndarray,
        *,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """Prefer the largest central detection, with continuity across frames."""
        frame_area = max(float(frame_width * frame_height), 1.0)
        frame_diagonal = max(math.hypot(frame_width, frame_height), 1.0)
        best_box = boxes[0]
        best_score = -math.inf
        for box in boxes:
            x1, y1, x2, y2 = (float(value) for value in box[:4])
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width <= 0.0 or height <= 0.0:
                continue
            area_score = width * height / frame_area
            center_distance = math.hypot(
                (x1 + x2) / 2.0 - frame_width / 2.0,
                (y1 + y2) / 2.0 - frame_height / 2.0,
            ) / frame_diagonal
            continuity = (
                self._bbox_iou(box, self._previous_bbox)
                if self._previous_bbox is not None
                else 0.0
            )
            score = 2.0 * area_score - 0.15 * center_distance + 0.5 * continuity
            if score > best_score:
                best_box = box
                best_score = score
        return np.asarray(best_box[:4], dtype=np.float64)

    @staticmethod
    def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
            0.0, float(box_a[3] - box_a[1])
        )
        area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
            0.0, float(box_b[3] - box_b[1])
        )
        union = area_a + area_b - intersection
        return intersection / union if union > 0.0 else 0.0

    def _extract_primary_keypoints(
        self,
        coordinates: Any,
        scores: Any,
        *,
        frame_width: int,
        frame_height: int,
    ) -> list[Keypoint]:
        if frame_width <= 0 or frame_height <= 0:
            return []

        coordinate_array = np.asarray(coordinates, dtype=np.float64)
        score_array = np.asarray(scores, dtype=np.float64)
        if coordinate_array.size == 0 or score_array.size == 0:
            return []
        if coordinate_array.ndim == 2:
            coordinate_array = coordinate_array[np.newaxis, ...]
        if score_array.ndim == 1:
            score_array = score_array[np.newaxis, ...]
        if (
            coordinate_array.ndim != 3
            or coordinate_array.shape[2] < 2
            or score_array.ndim != 2
            or coordinate_array.shape[:2] != score_array.shape
            or coordinate_array.shape[1] < len(COCO_TO_ESSENTIAL)
        ):
            raise ValueError(
                "RTMPose output must have shapes (people, keypoints, 2) and "
                "(people, keypoints)"
            )

        person_index = self._select_primary_person(
            coordinate_array,
            score_array,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        person_coordinates = coordinate_array[person_index]
        person_scores = score_array[person_index]

        keypoints: list[Keypoint] = []
        for index, name in COCO_TO_ESSENTIAL.items():
            x_px, y_px = person_coordinates[index, :2]
            confidence = float(person_scores[index])
            if (
                not math.isfinite(float(x_px))
                or not math.isfinite(float(y_px))
                or not math.isfinite(confidence)
                or confidence < self.min_confidence
            ):
                continue
            keypoints.append(
                Keypoint(
                    x=float(x_px) / frame_width,
                    y=float(y_px) / frame_height,
                    z=0.0,
                    confidence=max(0.0, min(confidence, 1.0)),
                    name=name,
                )
            )

        self._append_derived_keypoints(keypoints)
        _append_keypoint_aliases(keypoints)
        return keypoints

    def _select_primary_person(
        self,
        coordinates: np.ndarray,
        scores: np.ndarray,
        *,
        frame_width: int,
        frame_height: int,
    ) -> int:
        """Select the confident, frame-central person for single-batter clips."""
        best_index = 0
        best_rank = (-1, -math.inf, -math.inf)
        for person_index in range(coordinates.shape[0]):
            person_scores = scores[person_index, PRIMARY_PERSON_JOINTS]
            finite_scores = person_scores[np.isfinite(person_scores)]
            visible_count = int(
                np.count_nonzero(finite_scores >= self.min_confidence)
            )
            mean_confidence = (
                float(np.mean(np.clip(finite_scores, 0.0, 1.0)))
                if finite_scores.size
                else 0.0
            )

            torso_indices = (5, 6, 11, 12)
            torso = coordinates[person_index, torso_indices, :2]
            valid_torso = torso[np.all(np.isfinite(torso), axis=1)]
            if valid_torso.size:
                center = np.mean(valid_torso, axis=0)
                normalized_distance = math.hypot(
                    float(center[0]) / frame_width - 0.5,
                    float(center[1]) / frame_height - 0.5,
                )
            else:
                normalized_distance = math.inf
            rank = (visible_count, mean_confidence, -normalized_distance)
            if rank > best_rank:
                best_index = person_index
                best_rank = rank
        return best_index

    @staticmethod
    def _append_derived_keypoints(keypoints: list[Keypoint]) -> None:
        by_name = {keypoint.name: keypoint for keypoint in keypoints}
        left_shoulder = by_name.get("left_shoulder")
        right_shoulder = by_name.get("right_shoulder")
        if left_shoulder is None or right_shoulder is None:
            return
        confidence = min(left_shoulder.confidence, right_shoulder.confidence)
        midpoint_x = (left_shoulder.x + right_shoulder.x) / 2.0
        midpoint_y = (left_shoulder.y + right_shoulder.y) / 2.0
        keypoints.extend(
            [
                Keypoint(
                    x=midpoint_x,
                    y=midpoint_y,
                    z=0.0,
                    confidence=confidence,
                    name="spine",
                ),
                Keypoint(
                    x=midpoint_x,
                    y=midpoint_y - 0.02,
                    z=0.0,
                    confidence=confidence,
                    name="neck",
                ),
            ]
        )

    def close(self) -> None:
        self._solution = None
        self._previous_bbox = None

    def __enter__(self) -> "RTMPoseEstimator":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
