"""Unit tests for the PoseEstimator module.

Tests keypoint mapping, confidence filtering, PoseResult construction,
and interface correctness without requiring actual MediaPipe inference.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.models.pose import Keypoint, PoseResult
from app.pipeline.constants import (
    DEFAULT_MIN_CONFIDENCE,
    ESSENTIAL_KEYPOINT_NAMES,
    MEDIAPIPE_TO_ESSENTIAL,
    SPINE_KEYPOINT_NAME,
    SPINE_SOURCE_INDICES,
)
from app.pipeline.pose_estimator import (
    PoseEstimator,
    extract_keypoints_from_landmarks,
)

# --- Helper fixtures and mock data ---


@dataclass
class MockLandmark:
    """Mock MediaPipe landmark for testing."""

    x: float
    y: float
    z: float
    visibility: float


def create_mock_landmarks(
    count: int = 33, default_confidence: float = 0.9
) -> list[MockLandmark]:
    """Create a list of mock landmarks simulating MediaPipe output."""
    landmarks = []
    for i in range(count):
        landmarks.append(
            MockLandmark(
                x=0.1 + (i * 0.02),
                y=0.2 + (i * 0.015),
                z=-0.1 + (i * 0.005),
                visibility=default_confidence,
            )
        )
    return landmarks


def create_mock_landmarks_with_varied_confidence(
    high_conf: float = 0.9, low_conf: float = 0.3
) -> list[MockLandmark]:
    """Create landmarks where some have low confidence."""
    landmarks = create_mock_landmarks(33, high_conf)
    # Set some essential keypoints to low confidence
    # Index 13 (left_elbow), 14 (right_elbow) → low confidence
    landmarks[13] = MockLandmark(x=0.5, y=0.5, z=0.0, visibility=low_conf)
    landmarks[14] = MockLandmark(x=0.6, y=0.5, z=0.0, visibility=low_conf)
    return landmarks


# --- Tests for constants ---


class TestConstants:
    """Tests for keypoint mapping constants."""

    def test_mediapipe_to_essential_has_17_direct_mappings(self):
        """MEDIAPIPE_TO_ESSENTIAL should map 17 MediaPipe indices to names."""
        assert len(MEDIAPIPE_TO_ESSENTIAL) == 17

    def test_mediapipe_indices_are_valid(self):
        """All MediaPipe indices should be in range 0-32."""
        for idx in MEDIAPIPE_TO_ESSENTIAL:
            assert 0 <= idx <= 32

    def test_essential_keypoint_names_has_22_entries(self):
        """ESSENTIAL_KEYPOINT_NAMES should include canonical names and aliases."""
        assert len(ESSENTIAL_KEYPOINT_NAMES) == 22

    def test_all_direct_mappings_in_essential_names(self):
        """All directly mapped keypoint names should be in ESSENTIAL_KEYPOINT_NAMES."""
        for name in MEDIAPIPE_TO_ESSENTIAL.values():
            assert name in ESSENTIAL_KEYPOINT_NAMES

    def test_spine_in_essential_names(self):
        """Spine (calculated) should be in ESSENTIAL_KEYPOINT_NAMES."""
        assert SPINE_KEYPOINT_NAME in ESSENTIAL_KEYPOINT_NAMES

    def test_spine_source_indices_are_shoulders(self):
        """Spine source indices should be left_shoulder (11) and right_shoulder (12)."""
        assert SPINE_SOURCE_INDICES == (11, 12)

    def test_default_min_confidence(self):
        """Default minimum confidence should be 0.5."""
        assert DEFAULT_MIN_CONFIDENCE == 0.5

    def test_essential_names_include_calculated_keypoints(self):
        """Essential names should include spine and neck (calculated keypoints)."""
        assert "spine" in ESSENTIAL_KEYPOINT_NAMES
        assert "neck" in ESSENTIAL_KEYPOINT_NAMES


# --- Tests for keypoint mapping (33 → 17+) ---


class TestKeypointMapping:
    """Tests for the 33 → 17+ keypoint mapping logic."""

    def test_extract_all_essential_keypoints_from_high_confidence_landmarks(self):
        """All direct keypoints, aliases, spine, and ears are extracted."""
        landmarks = create_mock_landmarks(33, default_confidence=0.9)
        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.5)

        # 17 direct + 2 additional (ears) + 1 spine + 1 head alias.
        assert len(keypoints) >= 21

    def test_mapping_produces_correct_names(self):
        """Extracted keypoints should have correct names from the mapping."""
        landmarks = create_mock_landmarks(33, default_confidence=0.9)
        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.5)

        names = {kp.name for kp in keypoints}
        # Check essential direct mappings are present
        assert "nose" in names
        assert "head" in names
        assert "left_eye" in names
        assert "right_eye" in names
        assert "left_shoulder" in names
        assert "right_shoulder" in names
        assert "left_index" in names
        assert "right_index" in names
        assert "left_hip" in names
        assert "right_hip" in names
        assert "left_wrist" in names
        assert "right_wrist" in names
        assert "left_ankle" in names
        assert "right_ankle" in names
        assert "spine" in names

    def test_spine_is_midpoint_of_shoulders(self):
        """Spine keypoint should be the midpoint of left and right shoulders."""
        landmarks = create_mock_landmarks(33, default_confidence=0.9)
        # Set specific shoulder positions
        landmarks[11] = MockLandmark(x=0.3, y=0.4, z=-0.1, visibility=0.95)
        landmarks[12] = MockLandmark(x=0.7, y=0.4, z=-0.1, visibility=0.85)

        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.5)
        spine = next((kp for kp in keypoints if kp.name == "spine"), None)

        assert spine is not None
        assert abs(spine.x - 0.5) < 1e-6  # midpoint of 0.3 and 0.7
        assert abs(spine.y - 0.4) < 1e-6
        assert abs(spine.z - (-0.1)) < 1e-6
        # Spine confidence = min of both shoulders
        assert abs(spine.confidence - 0.85) < 1e-6

    def test_nose_maps_to_canonical_name_and_head_alias(self):
        """MediaPipe landmark 0 should emit canonical 'nose' and legacy 'head'."""
        landmarks = create_mock_landmarks(33, default_confidence=0.9)
        landmarks[0] = MockLandmark(x=0.5, y=0.2, z=0.0, visibility=0.95)

        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.5)
        nose = next((kp for kp in keypoints if kp.name == "nose"), None)
        head = next((kp for kp in keypoints if kp.name == "head"), None)

        assert nose is not None
        assert head is not None
        assert abs(nose.x - 0.5) < 1e-6
        assert abs(nose.y - 0.2) < 1e-6
        assert abs(head.x - 0.5) < 1e-6
        assert abs(head.y - 0.2) < 1e-6

    def test_ears_are_included_from_additional_landmarks(self):
        """Ear landmarks (7, 8) should be included as additional keypoints."""
        landmarks = create_mock_landmarks(33, default_confidence=0.9)
        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.5)

        names = {kp.name for kp in keypoints}
        assert "left_ear" in names
        assert "right_ear" in names


# --- Tests for confidence filtering ---


class TestConfidenceFiltering:
    """Tests for keypoint confidence score filtering."""

    def test_keypoints_below_threshold_are_excluded(self):
        """Keypoints with confidence < 0.5 should be excluded."""
        landmarks = create_mock_landmarks_with_varied_confidence(
            high_conf=0.9, low_conf=0.3
        )
        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.5)

        # left_elbow and right_elbow have 0.3 confidence → excluded
        names = {kp.name for kp in keypoints}
        assert "left_elbow" not in names
        assert "right_elbow" not in names

    def test_keypoints_at_threshold_are_included(self):
        """Keypoints with confidence exactly at threshold should be included."""
        landmarks = create_mock_landmarks(33, default_confidence=0.5)
        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.5)

        # All should be included since confidence == threshold
        assert len(keypoints) > 0

    def test_keypoints_above_threshold_are_included(self):
        """Keypoints with confidence > threshold should be included."""
        landmarks = create_mock_landmarks(33, default_confidence=0.9)
        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.5)

        # 17 direct + 2 ears + 1 spine + 1 head alias.
        assert len(keypoints) == 21

    def test_all_keypoints_below_threshold_returns_empty(self):
        """If all keypoints are below threshold, result should be empty."""
        landmarks = create_mock_landmarks(33, default_confidence=0.2)
        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.5)

        assert len(keypoints) == 0

    def test_spine_excluded_when_shoulder_confidence_low(self):
        """Spine should be excluded if either shoulder has low confidence."""
        landmarks = create_mock_landmarks(33, default_confidence=0.9)
        # Set left_shoulder to low confidence
        landmarks[11] = MockLandmark(x=0.3, y=0.4, z=0.0, visibility=0.3)

        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.5)
        spine = next((kp for kp in keypoints if kp.name == "spine"), None)

        # Spine confidence = min(0.3, 0.9) = 0.3 < 0.5 → excluded
        assert spine is None

    def test_custom_threshold(self):
        """Custom confidence threshold should be respected."""
        landmarks = create_mock_landmarks(33, default_confidence=0.7)
        # With threshold 0.8, all should be excluded
        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.8)
        assert len(keypoints) == 0

        # With threshold 0.6, all should be included
        keypoints = extract_keypoints_from_landmarks(landmarks, min_confidence=0.6)
        assert len(keypoints) > 0


# --- Tests for PoseResult construction ---


class TestPoseResultConstruction:
    """Tests for PoseResult construction from detected keypoints."""

    def test_pose_result_has_correct_frame_index(self):
        """PoseResult should have the correct frame_index."""
        result = PoseResult(
            frame_index=42,
            keypoints=[],
            person_id=0,
            is_primary_batter=True,
            overall_confidence=0.0,
            is_low_confidence=True,
        )
        assert result.frame_index == 42

    def test_pose_result_with_keypoints(self):
        """PoseResult should contain the filtered keypoints."""
        keypoints = [
            Keypoint(x=0.5, y=0.3, z=0.0, confidence=0.9, name="head"),
            Keypoint(x=0.4, y=0.5, z=0.0, confidence=0.85, name="left_shoulder"),
        ]
        result = PoseResult(
            frame_index=0,
            keypoints=keypoints,
            person_id=0,
            is_primary_batter=True,
            overall_confidence=0.875,
            is_low_confidence=False,
        )
        assert len(result.keypoints) == 2
        assert result.keypoints[0].name == "head"
        assert result.overall_confidence == 0.875

    def test_pose_result_low_confidence_flag(self):
        """is_low_confidence should be True when overall confidence < threshold."""
        result = PoseResult(
            frame_index=0,
            keypoints=[],
            person_id=0,
            is_primary_batter=True,
            overall_confidence=0.3,
            is_low_confidence=True,
        )
        assert result.is_low_confidence is True

    def test_pose_result_empty_keypoints_when_no_detection(self):
        """PoseResult should have empty keypoints when nothing is detected."""
        result = PoseResult(
            frame_index=5,
            keypoints=[],
            person_id=0,
            is_primary_batter=True,
            overall_confidence=0.0,
            is_low_confidence=True,
        )
        assert len(result.keypoints) == 0
        assert result.overall_confidence == 0.0


# --- Tests for PoseEstimator interface ---


class TestPoseEstimatorInterface:
    """Tests for PoseEstimator class interface and behavior."""

    def test_pose_estimator_protocol_compliance(self):
        """PoseEstimator should satisfy the PoseEstimatorProtocol."""
        # Verify the class has the required methods
        assert hasattr(PoseEstimator, "detect_keypoints")
        assert hasattr(PoseEstimator, "process_frame")
        assert hasattr(PoseEstimator, "close")

    def test_default_min_confidence(self):
        """PoseEstimator should default to 0.5 min_confidence."""
        with patch(
            "app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", False
        ):
            estimator = PoseEstimator.__new__(PoseEstimator)
            estimator.min_confidence = DEFAULT_MIN_CONFIDENCE
            estimator._pose = None
            estimator._static_image_mode = False
            estimator._model_complexity = 1
            assert estimator.min_confidence == 0.5

    def test_custom_min_confidence(self):
        """PoseEstimator should accept custom min_confidence."""
        with patch(
            "app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", False
        ):
            estimator = PoseEstimator.__new__(PoseEstimator)
            estimator.min_confidence = 0.7
            estimator._pose = None
            assert estimator.min_confidence == 0.7

    def test_is_available_false_without_mediapipe(self):
        """is_available should be False when MediaPipe is not installed."""
        with patch(
            "app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", False
        ):
            estimator = PoseEstimator.__new__(PoseEstimator)
            estimator._pose = None
            estimator.min_confidence = 0.5
            assert estimator.is_available is False

    def test_process_frame_raises_without_mediapipe(self):
        """process_frame should raise RuntimeError when MediaPipe is unavailable."""
        with patch(
            "app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", False
        ):
            estimator = PoseEstimator.__new__(PoseEstimator)
            estimator._pose = None
            estimator.min_confidence = 0.5

            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            with pytest.raises(RuntimeError, match="MediaPipe is not available"):
                estimator.process_frame(frame, frame_index=0)

    def test_detect_keypoints_raises_without_mediapipe(self):
        """detect_keypoints should raise RuntimeError when MediaPipe is unavailable."""
        with patch(
            "app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", False
        ):
            estimator = PoseEstimator.__new__(PoseEstimator)
            estimator._pose = None
            estimator.min_confidence = 0.5

            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            with pytest.raises(RuntimeError, match="MediaPipe is not available"):
                estimator.detect_keypoints(frame)

    def test_context_manager_support(self):
        """PoseEstimator should support context manager protocol."""
        with patch(
            "app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", False
        ):
            estimator = PoseEstimator.__new__(PoseEstimator)
            estimator._pose = None
            estimator.min_confidence = 0.5

            # Should not raise
            with estimator as est:
                assert est is estimator

    def test_close_sets_pose_to_none(self):
        """close() should release the MediaPipe Pose instance."""
        mock_pose = MagicMock()
        with patch(
            "app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", True
        ):
            estimator = PoseEstimator.__new__(PoseEstimator)
            estimator._pose = mock_pose
            estimator.min_confidence = 0.5

            estimator.close()
            assert estimator._pose is None
            mock_pose.close.assert_called_once()


# --- Tests for PoseEstimator with mocked MediaPipe ---


class TestPoseEstimatorWithMockedMediaPipe:
    """Tests using mocked MediaPipe to verify processing logic."""

    def _create_estimator_with_mock(self) -> tuple[PoseEstimator, MagicMock]:
        """Create a PoseEstimator with a mocked MediaPipe Pose instance."""
        mock_pose = MagicMock()
        estimator = PoseEstimator.__new__(PoseEstimator)
        estimator._pose = mock_pose
        estimator.min_confidence = 0.5
        estimator._static_image_mode = False
        estimator._model_complexity = 1
        return estimator, mock_pose

    @patch("app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", True)
    def test_process_frame_returns_pose_result(self):
        """process_frame should return a PoseResult with correct frame_index."""
        estimator, mock_pose = self._create_estimator_with_mock()

        # Mock MediaPipe results with landmarks
        mock_results = MagicMock()
        mock_landmarks = create_mock_landmarks(33, default_confidence=0.9)
        mock_results.pose_landmarks.landmark = mock_landmarks
        mock_pose.process.return_value = mock_results

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = estimator.process_frame(frame, frame_index=10)

        assert isinstance(result, PoseResult)
        assert result.frame_index == 10
        assert len(result.keypoints) > 0

    @patch("app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", True)
    def test_process_frame_filters_low_confidence(self):
        """process_frame should filter out keypoints below min_confidence."""
        estimator, mock_pose = self._create_estimator_with_mock()

        # Create landmarks with mixed confidence
        mock_landmarks = create_mock_landmarks_with_varied_confidence(
            high_conf=0.9, low_conf=0.3
        )
        mock_results = MagicMock()
        mock_results.pose_landmarks.landmark = mock_landmarks
        mock_pose.process.return_value = mock_results

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = estimator.process_frame(frame, frame_index=0)

        # Verify low-confidence keypoints are excluded
        names = {kp.name for kp in result.keypoints}
        assert "left_elbow" not in names
        assert "right_elbow" not in names

    @patch("app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", True)
    def test_process_frame_no_detection(self):
        """process_frame should return empty keypoints when no pose detected."""
        estimator, mock_pose = self._create_estimator_with_mock()

        # Mock no detection
        mock_results = MagicMock()
        mock_results.pose_landmarks = None
        mock_pose.process.return_value = mock_results

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = estimator.process_frame(frame, frame_index=5)

        assert isinstance(result, PoseResult)
        assert len(result.keypoints) == 0
        assert result.overall_confidence == 0.0
        assert result.is_low_confidence is True

    @patch("app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", True)
    def test_process_frame_calculates_overall_confidence(self):
        """process_frame should calculate overall confidence as mean of keypoints."""
        estimator, mock_pose = self._create_estimator_with_mock()

        # All landmarks at 0.8 confidence
        mock_landmarks = create_mock_landmarks(33, default_confidence=0.8)
        mock_results = MagicMock()
        mock_results.pose_landmarks.landmark = mock_landmarks
        mock_pose.process.return_value = mock_results

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = estimator.process_frame(frame, frame_index=0)

        # All keypoints have 0.8 confidence, spine = min(0.8, 0.8) = 0.8
        assert abs(result.overall_confidence - 0.8) < 0.01

    @patch("app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", True)
    def test_detect_keypoints_delegates_to_process_frame(self):
        """detect_keypoints should call process_frame with frame_index=0."""
        estimator, mock_pose = self._create_estimator_with_mock()

        mock_results = MagicMock()
        mock_results.pose_landmarks = None
        mock_pose.process.return_value = mock_results

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = estimator.detect_keypoints(frame)

        assert result.frame_index == 0

    @patch("app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", True)
    def test_process_frame_includes_spine_keypoint(self):
        """process_frame should include calculated spine keypoint."""
        estimator, mock_pose = self._create_estimator_with_mock()

        mock_landmarks = create_mock_landmarks(33, default_confidence=0.9)
        mock_results = MagicMock()
        mock_results.pose_landmarks.landmark = mock_landmarks
        mock_pose.process.return_value = mock_results

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = estimator.process_frame(frame, frame_index=0)

        names = {kp.name for kp in result.keypoints}
        assert "spine" in names

    @patch("app.pipeline.pose_estimator.MEDIAPIPE_AVAILABLE", True)
    def test_process_frame_converts_bgr_to_rgb(self):
        """process_frame should convert BGR frame to RGB for MediaPipe."""
        estimator, mock_pose = self._create_estimator_with_mock()

        mock_results = MagicMock()
        mock_results.pose_landmarks = None
        mock_pose.process.return_value = mock_results

        # Create a frame with known BGR values
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue channel
        frame[:, :, 2] = 0  # Red channel

        estimator.process_frame(frame, frame_index=0)

        # Verify process was called (the conversion happens internally)
        mock_pose.process.assert_called_once()
        # The passed frame should be RGB (reversed channels)
        passed_frame = mock_pose.process.call_args[0][0]
        assert passed_frame[0, 0, 0] == 0  # Red (was Blue)
        assert passed_frame[0, 0, 2] == 255  # Blue (was Red)
