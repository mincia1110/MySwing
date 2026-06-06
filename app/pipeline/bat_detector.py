"""Generic bat detection helpers and test doubles.

The production pipeline currently uses wrist/elbow keypoints via
``WristBatEstimator`` as the primary bat trajectory source. This module keeps
shared bounding-box geometry helpers and a mock detector for tests or future
non-AGPL detector adapters.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Protocol, Tuple

import numpy as np

from app.models.bat import BatDetectionResult


class BatDetectorProtocol(Protocol):
    """Protocol defining the bat detection interface."""

    def detect_bat(self, frame: np.ndarray, frame_index: int) -> BatDetectionResult:
        """Detect bat in a single frame."""
        ...


class BaseBatDetector(ABC):
    """Abstract base class for bat detection implementations."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = 0.9,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def detect_bat(self, frame: np.ndarray, frame_index: int) -> BatDetectionResult:
        """Detect bat in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            frame_index: Index of the frame in the video sequence.

        Returns:
            BatDetectionResult with position, orientation, length, and confidence.
        """
        ...

    @staticmethod
    def _calculate_orientation(bbox: Tuple[float, float, float, float]) -> float:
        """Calculate bat orientation angle from bounding box.

        Uses the bounding box aspect ratio and diagonal to estimate orientation.
        For elongated objects (like bats), the major axis direction gives orientation.

        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates.

        Returns:
            Orientation angle in degrees (0-360).
        """
        x1, y1, x2, y2 = bbox
        dx = x2 - x1
        dy = y2 - y1

        # Angle of the diagonal (major axis of elongated bbox)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Normalize to 0-360 range
        if angle_deg < 0:
            angle_deg += 360.0

        return angle_deg

    @staticmethod
    def _calculate_length(bbox: Tuple[float, float, float, float]) -> float:
        """Calculate bat length in pixels from bounding box diagonal.

        For an elongated object like a bat, the diagonal of the bounding box
        approximates the object's length.

        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates.

        Returns:
            Length in pixels.
        """
        x1, y1, x2, y2 = bbox
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _calculate_position(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculate bat center position from bounding box.

        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates.

        Returns:
            (center_x, center_y) pixel coordinates.
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return (center_x, center_y)

    @staticmethod
    def _is_elongated(bbox: Tuple[float, float, float, float], min_aspect_ratio: float = 2.5) -> bool:
        """Check if a bounding box represents an elongated object.

        Bats are elongated objects with high aspect ratios. This helps
        distinguish bats from other detected objects.

        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates.
            min_aspect_ratio: Minimum aspect ratio to consider elongated.

        Returns:
            True if the object is elongated enough to be a bat candidate.
        """
        x1, y1, x2, y2 = bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        if width == 0 or height == 0:
            return False

        aspect_ratio = max(width, height) / min(width, height)
        return aspect_ratio >= min_aspect_ratio

    def _create_no_detection_result(self, frame_index: int) -> BatDetectionResult:
        """Create a BatDetectionResult indicating no bat was detected."""
        return BatDetectionResult(
            frame_index=frame_index,
            detected=False,
            position=(0.0, 0.0),
            orientation_angle=0.0,
            length_pixels=0.0,
            confidence=0.0,
            is_predicted=False,
        )


class MockBatDetector(BaseBatDetector):
    """Mock bat detector for testing detector-like consumers.

    Allows setting predetermined detection results for testing purposes.
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = 0.9,
        mock_detections: dict[int, BatDetectionResult] | None = None,
    ) -> None:
        super().__init__(model_path=model_path, confidence_threshold=confidence_threshold)
        self._mock_detections: dict[int, BatDetectionResult] = mock_detections or {}

    def set_detection(self, frame_index: int, detection: BatDetectionResult) -> None:
        """Set a mock detection result for a specific frame."""
        self._mock_detections[frame_index] = detection

    def detect_bat(self, frame: np.ndarray, frame_index: int) -> BatDetectionResult:
        """Return mock detection result for the given frame.

        Args:
            frame: BGR image (ignored in mock).
            frame_index: Frame index to look up.

        Returns:
            Mock BatDetectionResult or no-detection result.
        """
        if frame_index in self._mock_detections:
            detection = self._mock_detections[frame_index]
            # Still apply confidence threshold filtering
            if detection.confidence < self.confidence_threshold:
                return self._create_no_detection_result(frame_index)
            return detection

        return self._create_no_detection_result(frame_index)


def create_bat_detector(
    model_path: str | None = None,
    confidence_threshold: float = 0.9,
    use_mock: bool = True,
) -> BaseBatDetector:
    """Create a detector-like test double.

    The production bat trajectory path is wrist-based and does not depend on an
    object detector. This factory intentionally returns ``MockBatDetector`` so
    tests and future adapters can still exercise the legacy detector protocol
    without pulling in detector-specific copyleft dependencies.
    """
    return MockBatDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )
