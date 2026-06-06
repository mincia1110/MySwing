"""Unit tests for the bat detector module (Task 6.1).

Tests BatDetectionResult construction, orientation/length/position calculations,
confidence threshold filtering, and mock detector interface correctness.
"""

import math

import numpy as np
import pytest

from app.models.bat import BatDetectionResult
from app.pipeline.bat_detector import (
    BaseBatDetector,
    MockBatDetector,
    create_bat_detector,
)


class TestBatDetectionResultConstruction:
    """Test BatDetectionResult dataclass construction."""

    def test_basic_construction(self):
        result = BatDetectionResult(
            frame_index=0,
            detected=True,
            position=(320.0, 240.0),
            orientation_angle=45.0,
            length_pixels=150.0,
            confidence=0.95,
            is_predicted=False,
        )
        assert result.frame_index == 0
        assert result.detected is True
        assert result.position == (320.0, 240.0)
        assert result.orientation_angle == 45.0
        assert result.length_pixels == 150.0
        assert result.confidence == 0.95
        assert result.is_predicted is False

    def test_no_detection_result(self):
        result = BatDetectionResult(
            frame_index=5,
            detected=False,
            position=(0.0, 0.0),
            orientation_angle=0.0,
            length_pixels=0.0,
            confidence=0.0,
            is_predicted=False,
        )
        assert result.detected is False
        assert result.confidence == 0.0

    def test_predicted_result(self):
        result = BatDetectionResult(
            frame_index=10,
            detected=True,
            position=(100.0, 200.0),
            orientation_angle=90.0,
            length_pixels=120.0,
            confidence=0.85,
            is_predicted=True,
        )
        assert result.is_predicted is True


class TestOrientationCalculation:
    """Test _calculate_orientation for various angles."""

    def test_horizontal_right(self):
        """Horizontal bbox pointing right → 0 degrees."""
        bbox = (0.0, 0.0, 200.0, 10.0)
        angle = BaseBatDetector._calculate_orientation(bbox)
        assert 0.0 <= angle < 5.0 or angle > 355.0

    def test_horizontal_left(self):
        """Horizontal bbox pointing left → ~180 degrees."""
        bbox = (200.0, 10.0, 0.0, 0.0)
        angle = BaseBatDetector._calculate_orientation(bbox)
        assert 175.0 <= angle <= 185.0

    def test_vertical_down(self):
        """Vertical bbox pointing down → ~90 degrees."""
        bbox = (0.0, 0.0, 10.0, 200.0)
        angle = BaseBatDetector._calculate_orientation(bbox)
        assert 85.0 <= angle <= 95.0

    def test_vertical_up(self):
        """Vertical bbox pointing up → ~270 degrees."""
        bbox = (10.0, 200.0, 0.0, 0.0)
        angle = BaseBatDetector._calculate_orientation(bbox)
        assert 265.0 <= angle <= 275.0

    def test_diagonal_45_degrees(self):
        """Diagonal bbox → ~45 degrees."""
        bbox = (0.0, 0.0, 100.0, 100.0)
        angle = BaseBatDetector._calculate_orientation(bbox)
        assert 40.0 <= angle <= 50.0

    def test_diagonal_135_degrees(self):
        """Diagonal bbox → ~135 degrees."""
        bbox = (100.0, 0.0, 0.0, 100.0)
        angle = BaseBatDetector._calculate_orientation(bbox)
        assert 130.0 <= angle <= 140.0

    def test_angle_always_in_range(self):
        """Orientation angle is always in [0, 360) range."""
        test_bboxes = [
            (0.0, 0.0, 100.0, 50.0),
            (100.0, 50.0, 0.0, 0.0),
            (50.0, 0.0, 50.0, 100.0),
            (0.0, 100.0, 100.0, 0.0),
        ]
        for bbox in test_bboxes:
            angle = BaseBatDetector._calculate_orientation(bbox)
            assert 0.0 <= angle < 360.0, f"Angle {angle} out of range for bbox {bbox}"


class TestLengthCalculation:
    """Test _calculate_length for various bounding boxes."""

    def test_horizontal_bbox(self):
        """Horizontal bbox length is the diagonal."""
        bbox = (0.0, 0.0, 100.0, 10.0)
        length = BaseBatDetector._calculate_length(bbox)
        expected = math.sqrt(100**2 + 10**2)
        assert abs(length - expected) < 0.01

    def test_vertical_bbox(self):
        """Vertical bbox length."""
        bbox = (0.0, 0.0, 10.0, 200.0)
        length = BaseBatDetector._calculate_length(bbox)
        expected = math.sqrt(10**2 + 200**2)
        assert abs(length - expected) < 0.01

    def test_square_bbox(self):
        """Square bbox diagonal."""
        bbox = (0.0, 0.0, 100.0, 100.0)
        length = BaseBatDetector._calculate_length(bbox)
        expected = math.sqrt(100**2 + 100**2)
        assert abs(length - expected) < 0.01

    def test_zero_size_bbox(self):
        """Zero-size bbox has zero length."""
        bbox = (50.0, 50.0, 50.0, 50.0)
        length = BaseBatDetector._calculate_length(bbox)
        assert length == 0.0

    def test_typical_bat_bbox(self):
        """Typical bat-like elongated bbox."""
        bbox = (100.0, 200.0, 300.0, 220.0)
        length = BaseBatDetector._calculate_length(bbox)
        expected = math.sqrt(200**2 + 20**2)
        assert abs(length - expected) < 0.01


class TestPositionCalculation:
    """Test _calculate_position from bounding box."""

    def test_center_of_bbox(self):
        bbox = (100.0, 200.0, 300.0, 400.0)
        pos = BaseBatDetector._calculate_position(bbox)
        assert pos == (200.0, 300.0)

    def test_origin_bbox(self):
        bbox = (0.0, 0.0, 100.0, 100.0)
        pos = BaseBatDetector._calculate_position(bbox)
        assert pos == (50.0, 50.0)

    def test_small_bbox(self):
        bbox = (10.0, 10.0, 12.0, 12.0)
        pos = BaseBatDetector._calculate_position(bbox)
        assert pos == (11.0, 11.0)

    def test_asymmetric_bbox(self):
        bbox = (0.0, 0.0, 640.0, 20.0)
        pos = BaseBatDetector._calculate_position(bbox)
        assert pos == (320.0, 10.0)


class TestElongatedCheck:
    """Test _is_elongated for bat vs non-bat shapes."""

    def test_elongated_horizontal(self):
        """Wide horizontal bbox is elongated."""
        bbox = (0.0, 0.0, 200.0, 20.0)
        assert BaseBatDetector._is_elongated(bbox) is True

    def test_elongated_vertical(self):
        """Tall vertical bbox is elongated."""
        bbox = (0.0, 0.0, 20.0, 200.0)
        assert BaseBatDetector._is_elongated(bbox) is True

    def test_square_not_elongated(self):
        """Square bbox is not elongated."""
        bbox = (0.0, 0.0, 100.0, 100.0)
        assert BaseBatDetector._is_elongated(bbox) is False

    def test_slightly_rectangular_not_elongated(self):
        """Slightly rectangular bbox (2:1) is not elongated enough."""
        bbox = (0.0, 0.0, 100.0, 50.0)
        assert BaseBatDetector._is_elongated(bbox) is False

    def test_zero_width(self):
        """Zero width bbox is not elongated."""
        bbox = (50.0, 0.0, 50.0, 100.0)
        assert BaseBatDetector._is_elongated(bbox) is False

    def test_zero_height(self):
        """Zero height bbox is not elongated."""
        bbox = (0.0, 50.0, 100.0, 50.0)
        assert BaseBatDetector._is_elongated(bbox) is False

    def test_custom_threshold(self):
        """Custom aspect ratio threshold."""
        bbox = (0.0, 0.0, 100.0, 50.0)  # 2:1 ratio
        assert BaseBatDetector._is_elongated(bbox, min_aspect_ratio=1.5) is True
        assert BaseBatDetector._is_elongated(bbox, min_aspect_ratio=3.0) is False


class TestConfidenceThresholdFiltering:
    """Test confidence threshold filtering (≥90% accepted, <90% rejected)."""

    def test_high_confidence_accepted(self):
        """Detection with confidence ≥ 0.9 is accepted."""
        detector = MockBatDetector(confidence_threshold=0.9)
        detection = BatDetectionResult(
            frame_index=0,
            detected=True,
            position=(100.0, 200.0),
            orientation_angle=45.0,
            length_pixels=150.0,
            confidence=0.95,
            is_predicted=False,
        )
        detector.set_detection(0, detection)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_bat(frame, 0)
        assert result.detected is True
        assert result.confidence == 0.95

    def test_exact_threshold_accepted(self):
        """Detection with confidence exactly 0.9 is accepted."""
        detector = MockBatDetector(confidence_threshold=0.9)
        detection = BatDetectionResult(
            frame_index=0,
            detected=True,
            position=(100.0, 200.0),
            orientation_angle=45.0,
            length_pixels=150.0,
            confidence=0.9,
            is_predicted=False,
        )
        detector.set_detection(0, detection)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_bat(frame, 0)
        assert result.detected is True
        assert result.confidence == 0.9

    def test_low_confidence_rejected(self):
        """Detection with confidence < 0.9 is rejected."""
        detector = MockBatDetector(confidence_threshold=0.9)
        detection = BatDetectionResult(
            frame_index=0,
            detected=True,
            position=(100.0, 200.0),
            orientation_angle=45.0,
            length_pixels=150.0,
            confidence=0.85,
            is_predicted=False,
        )
        detector.set_detection(0, detection)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_bat(frame, 0)
        assert result.detected is False
        assert result.confidence == 0.0

    def test_very_low_confidence_rejected(self):
        """Detection with very low confidence is rejected."""
        detector = MockBatDetector(confidence_threshold=0.9)
        detection = BatDetectionResult(
            frame_index=0,
            detected=True,
            position=(100.0, 200.0),
            orientation_angle=45.0,
            length_pixels=150.0,
            confidence=0.3,
            is_predicted=False,
        )
        detector.set_detection(0, detection)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_bat(frame, 0)
        assert result.detected is False

    def test_custom_threshold(self):
        """Custom confidence threshold works correctly."""
        detector = MockBatDetector(confidence_threshold=0.7)
        detection = BatDetectionResult(
            frame_index=0,
            detected=True,
            position=(100.0, 200.0),
            orientation_angle=45.0,
            length_pixels=150.0,
            confidence=0.75,
            is_predicted=False,
        )
        detector.set_detection(0, detection)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_bat(frame, 0)
        assert result.detected is True


class TestMockBatDetector:
    """Test MockBatDetector behavior."""

    def test_no_detection_for_unknown_frame(self):
        """Returns no-detection for frames without mock data."""
        detector = MockBatDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_bat(frame, 99)
        assert result.detected is False
        assert result.frame_index == 99

    def test_set_and_retrieve_detection(self):
        """Can set and retrieve mock detections."""
        detector = MockBatDetector(confidence_threshold=0.9)
        detection = BatDetectionResult(
            frame_index=5,
            detected=True,
            position=(320.0, 240.0),
            orientation_angle=90.0,
            length_pixels=180.0,
            confidence=0.92,
            is_predicted=False,
        )
        detector.set_detection(5, detection)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_bat(frame, 5)
        assert result.detected is True
        assert result.position == (320.0, 240.0)
        assert result.orientation_angle == 90.0
        assert result.length_pixels == 180.0

    def test_multiple_frames(self):
        """Can handle detections across multiple frames."""
        detector = MockBatDetector(confidence_threshold=0.9)
        for i in range(5):
            detection = BatDetectionResult(
                frame_index=i,
                detected=True,
                position=(100.0 + i * 10, 200.0),
                orientation_angle=float(i * 30),
                length_pixels=150.0,
                confidence=0.95,
                is_predicted=False,
            )
            detector.set_detection(i, detection)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(5):
            result = detector.detect_bat(frame, i)
            assert result.detected is True
            assert result.position[0] == 100.0 + i * 10


class TestCreateBatDetectorFactory:
    """Test the factory function create_bat_detector."""

    def test_create_mock_detector(self):
        """Factory creates MockBatDetector when use_mock=True."""
        detector = create_bat_detector(use_mock=True)
        assert isinstance(detector, MockBatDetector)

    def test_create_default_detector_is_mock(self):
        """Factory creates MockBatDetector by default to avoid AGPL detector deps."""
        detector = create_bat_detector(use_mock=False)
        assert isinstance(detector, MockBatDetector)

    def test_custom_confidence_threshold(self):
        """Factory passes confidence threshold correctly."""
        detector = create_bat_detector(confidence_threshold=0.8, use_mock=True)
        assert detector.confidence_threshold == 0.8

    def test_custom_model_path(self):
        """Factory passes model path correctly."""
        detector = create_bat_detector(model_path="/path/to/model.pt", use_mock=True)
        assert detector.model_path == "/path/to/model.pt"
