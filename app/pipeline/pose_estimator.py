"""MediaPipe Pose-based keypoint detection module.

This module provides a PoseEstimator class that wraps MediaPipe Pose for
detecting body keypoints in video frames. It maps MediaPipe's 33 landmarks
to 17+ essential keypoints needed for baseball swing analysis.

The module is designed with an abstraction layer that gracefully handles
MediaPipe import errors, allowing for mock/stub implementations in testing.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import numpy as np

from app.models.pose import Keypoint, PoseResult
from app.pipeline.constants import (
    ADDITIONAL_MEDIAPIPE_LANDMARKS,
    DEFAULT_MIN_CONFIDENCE,
    KEYPOINT_ALIASES,
    MEDIAPIPE_TO_ESSENTIAL,
    SPINE_KEYPOINT_NAME,
    SPINE_SOURCE_INDICES,
)

logger = logging.getLogger(__name__)


def _append_keypoint_aliases(keypoints: list[Keypoint]) -> None:
    """Append backward-compatible aliases for canonical keypoint names."""
    by_name = {kp.name: kp for kp in keypoints}
    for source_name, alias_name in KEYPOINT_ALIASES.items():
        source = by_name.get(source_name)
        if source is None or alias_name in by_name:
            continue
        keypoints.append(
            Keypoint(
                x=source.x,
                y=source.y,
                z=source.z,
                confidence=source.confidence,
                name=alias_name,
            )
        )

# Try to import MediaPipe; if unavailable, set a flag
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None  # type: ignore[assignment]
    MEDIAPIPE_AVAILABLE = False
    logger.warning(
        "MediaPipe is not installed. PoseEstimator will not perform actual inference. "
        "Install with: pip install mediapipe"
    )


class PoseEstimatorProtocol(Protocol):
    """Protocol defining the interface for pose estimation."""

    def detect_keypoints(self, frame: np.ndarray) -> PoseResult:
        """Detect keypoints in a single frame."""
        ...

    def process_frame(self, frame: np.ndarray, frame_index: int) -> PoseResult:
        """Process a frame with its index and return pose result."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class PoseEstimator:
    """MediaPipe Pose-based keypoint detector.

    Detects body keypoints in video frames using MediaPipe Pose,
    maps 33 MediaPipe landmarks to 17+ essential keypoints, and
    filters by confidence score.

    Args:
        min_confidence: Minimum confidence threshold for keypoint detection.
            Keypoints below this threshold are excluded from results.
            Defaults to 0.5.
        static_image_mode: If True, treats each frame independently (no tracking).
            If False, uses tracking for better performance on video sequences.
            Defaults to False.
        model_complexity: Model complexity (0, 1, or 2). Higher is more accurate
            but slower. Defaults to 1.
    """

    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        static_image_mode: bool = False,
        model_complexity: int = 1,
    ) -> None:
        self.min_confidence = min_confidence
        self._static_image_mode = static_image_mode
        self._model_complexity = model_complexity
        self._pose = None

        if MEDIAPIPE_AVAILABLE:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )

    @property
    def is_available(self) -> bool:
        """Check if MediaPipe is available for inference."""
        return MEDIAPIPE_AVAILABLE and self._pose is not None

    def detect_keypoints(self, frame: np.ndarray) -> PoseResult:
        """Detect keypoints in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            PoseResult with detected keypoints filtered by confidence.

        Raises:
            RuntimeError: If MediaPipe is not available.
        """
        return self.process_frame(frame, frame_index=0)

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> PoseResult:
        """Process a single frame and return pose estimation result.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            frame_index: Index of the frame in the video sequence.

        Returns:
            PoseResult with detected keypoints, confidence scores, and metadata.

        Raises:
            RuntimeError: If MediaPipe is not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "MediaPipe is not available. Install with: pip install mediapipe"
            )

        start_time = time.time()

        # MediaPipe expects RGB input
        frame_rgb = frame[:, :, ::-1] if frame.shape[2] == 3 else frame
        results = self._pose.process(frame_rgb)

        keypoints = self._extract_keypoints(results)
        filtered_keypoints = self._filter_by_confidence(keypoints)

        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > 100.0:
            logger.warning(
                f"Frame {frame_index} processing took {elapsed_ms:.1f}ms "
                f"(exceeds 100ms target)"
            )

        overall_confidence = self._calculate_overall_confidence(filtered_keypoints)

        return PoseResult(
            frame_index=frame_index,
            keypoints=filtered_keypoints,
            person_id=0,
            is_primary_batter=True,
            overall_confidence=overall_confidence,
            is_low_confidence=overall_confidence < self.min_confidence,
        )

    def _extract_keypoints(self, results: object) -> list[Keypoint]:
        """Extract essential keypoints from MediaPipe results.

        Maps MediaPipe's 33 landmarks to our 17+ essential keypoints,
        including the calculated spine midpoint.
        """
        keypoints: list[Keypoint] = []

        if not hasattr(results, "pose_landmarks") or results.pose_landmarks is None:
            return keypoints

        landmarks = results.pose_landmarks.landmark

        # Map direct MediaPipe landmarks to essential keypoints
        for mp_index, name in MEDIAPIPE_TO_ESSENTIAL.items():
            if mp_index < len(landmarks):
                lm = landmarks[mp_index]
                keypoints.append(
                    Keypoint(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        confidence=lm.visibility,
                        name=name,
                    )
                )

        # Map additional landmarks (ears)
        for mp_index, name in ADDITIONAL_MEDIAPIPE_LANDMARKS.items():
            if mp_index < len(landmarks):
                lm = landmarks[mp_index]
                keypoints.append(
                    Keypoint(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        confidence=lm.visibility,
                        name=name,
                    )
                )

        # Calculate spine as midpoint of left_shoulder and right_shoulder
        left_idx, right_idx = SPINE_SOURCE_INDICES
        if left_idx < len(landmarks) and right_idx < len(landmarks):
            left_lm = landmarks[left_idx]
            right_lm = landmarks[right_idx]
            spine_confidence = min(left_lm.visibility, right_lm.visibility)
            keypoints.append(
                Keypoint(
                    x=(left_lm.x + right_lm.x) / 2,
                    y=(left_lm.y + right_lm.y) / 2,
                    z=(left_lm.z + right_lm.z) / 2,
                    confidence=spine_confidence,
                    name=SPINE_KEYPOINT_NAME,
                )
            )

        # Calculate neck as midpoint of shoulders (slightly higher)
        if left_idx < len(landmarks) and right_idx < len(landmarks):
            left_lm = landmarks[left_idx]
            right_lm = landmarks[right_idx]
            neck_confidence = min(left_lm.visibility, right_lm.visibility)
            keypoints.append(
                Keypoint(
                    x=(left_lm.x + right_lm.x) / 2,
                    y=(left_lm.y + right_lm.y) / 2 - 0.02,  # slightly above spine
                    z=(left_lm.z + right_lm.z) / 2,
                    confidence=neck_confidence,
                    name="neck",
                )
            )

        _append_keypoint_aliases(keypoints)
        return keypoints

    def _filter_by_confidence(self, keypoints: list[Keypoint]) -> list[Keypoint]:
        """Filter keypoints by minimum confidence threshold.

        Args:
            keypoints: List of detected keypoints.

        Returns:
            List of keypoints with confidence >= min_confidence.
        """
        return [kp for kp in keypoints if kp.confidence >= self.min_confidence]

    def _calculate_overall_confidence(self, keypoints: list[Keypoint]) -> float:
        """Calculate overall confidence as mean of all keypoint confidences.

        Args:
            keypoints: List of filtered keypoints.

        Returns:
            Mean confidence score, or 0.0 if no keypoints detected.
        """
        if not keypoints:
            return 0.0
        return sum(kp.confidence for kp in keypoints) / len(keypoints)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def extract_keypoints_from_landmarks(
    landmarks: list[object],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[Keypoint]:
    """Utility function to extract essential keypoints from raw landmark data.

    This function can be used independently of the PoseEstimator class
    for testing or when landmarks are obtained from other sources.

    Args:
        landmarks: List of landmark objects with x, y, z, visibility attributes.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of filtered essential keypoints.
    """
    keypoints: list[Keypoint] = []

    # Direct mappings
    for mp_index, name in MEDIAPIPE_TO_ESSENTIAL.items():
        if mp_index < len(landmarks):
            lm = landmarks[mp_index]
            kp = Keypoint(
                x=lm.x,  # type: ignore[attr-defined]
                y=lm.y,  # type: ignore[attr-defined]
                z=lm.z,  # type: ignore[attr-defined]
                confidence=lm.visibility,  # type: ignore[attr-defined]
                name=name,
            )
            if kp.confidence >= min_confidence:
                keypoints.append(kp)

    # Additional landmarks
    for mp_index, name in ADDITIONAL_MEDIAPIPE_LANDMARKS.items():
        if mp_index < len(landmarks):
            lm = landmarks[mp_index]
            kp = Keypoint(
                x=lm.x,  # type: ignore[attr-defined]
                y=lm.y,  # type: ignore[attr-defined]
                z=lm.z,  # type: ignore[attr-defined]
                confidence=lm.visibility,  # type: ignore[attr-defined]
                name=name,
            )
            if kp.confidence >= min_confidence:
                keypoints.append(kp)

    # Spine (midpoint of shoulders)
    left_idx, right_idx = SPINE_SOURCE_INDICES
    if left_idx < len(landmarks) and right_idx < len(landmarks):
        left_lm = landmarks[left_idx]
        right_lm = landmarks[right_idx]
        spine_conf = min(
            left_lm.visibility,  # type: ignore[attr-defined]
            right_lm.visibility,  # type: ignore[attr-defined]
        )
        if spine_conf >= min_confidence:
            keypoints.append(
                Keypoint(
                    x=(left_lm.x + right_lm.x) / 2,  # type: ignore[attr-defined]
                    y=(left_lm.y + right_lm.y) / 2,  # type: ignore[attr-defined]
                    z=(left_lm.z + right_lm.z) / 2,  # type: ignore[attr-defined]
                    confidence=spine_conf,
                    name=SPINE_KEYPOINT_NAME,
                )
            )

    _append_keypoint_aliases(keypoints)
    return keypoints
