"""Unit tests for OverlayRenderer (Task 11.1).

Tests:
- Skeleton drawing produces modified frame
- Bat trajectory drawing produces modified frame
- Output video is valid (can be opened by OpenCV)
- Empty pose/trajectory graceful handling
"""

import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.pose import Keypoint, PoseResult
from app.pipeline.report_generator import OverlayRenderer
from app.services.s3_client import S3Client


@pytest.fixture
def renderer() -> OverlayRenderer:
    """Create an OverlayRenderer instance."""
    return OverlayRenderer()


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a sample black frame (480x640, BGR)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_pose() -> PoseResult:
    """Create a sample PoseResult with keypoints for skeleton drawing."""
    keypoints = [
        Keypoint(x=0.5, y=0.2, z=0.0, confidence=0.9, name="nose"),
        Keypoint(x=0.48, y=0.19, z=0.0, confidence=0.85, name="left_eye"),
        Keypoint(x=0.52, y=0.19, z=0.0, confidence=0.85, name="right_eye"),
        Keypoint(x=0.46, y=0.2, z=0.0, confidence=0.8, name="left_ear"),
        Keypoint(x=0.54, y=0.2, z=0.0, confidence=0.8, name="right_ear"),
        Keypoint(x=0.45, y=0.35, z=0.0, confidence=0.9, name="left_shoulder"),
        Keypoint(x=0.55, y=0.35, z=0.0, confidence=0.9, name="right_shoulder"),
        Keypoint(x=0.4, y=0.5, z=0.0, confidence=0.85, name="left_elbow"),
        Keypoint(x=0.6, y=0.5, z=0.0, confidence=0.85, name="right_elbow"),
        Keypoint(x=0.38, y=0.65, z=0.0, confidence=0.8, name="left_wrist"),
        Keypoint(x=0.62, y=0.65, z=0.0, confidence=0.8, name="right_wrist"),
        Keypoint(x=0.47, y=0.6, z=0.0, confidence=0.9, name="left_hip"),
        Keypoint(x=0.53, y=0.6, z=0.0, confidence=0.9, name="right_hip"),
        Keypoint(x=0.46, y=0.75, z=0.0, confidence=0.85, name="left_knee"),
        Keypoint(x=0.54, y=0.75, z=0.0, confidence=0.85, name="right_knee"),
        Keypoint(x=0.45, y=0.9, z=0.0, confidence=0.8, name="left_ankle"),
        Keypoint(x=0.55, y=0.9, z=0.0, confidence=0.8, name="right_ankle"),
    ]
    return PoseResult(
        frame_index=0,
        keypoints=keypoints,
        person_id=1,
        is_primary_batter=True,
        overall_confidence=0.85,
        is_low_confidence=False,
    )


@pytest.fixture
def sample_bat_trajectory() -> BatTrajectory:
    """Create a sample BatTrajectory with detections across multiple frames."""
    detections = []
    for i in range(15):
        det = BatDetectionResult(
            frame_index=i,
            detected=True,
            position=(200.0 + i * 10, 300.0 - i * 5),
            orientation_angle=45.0 + i * 2,
            length_pixels=80.0,
            confidence=0.95,
            is_predicted=False,
        )
        detections.append(det)
    return BatTrajectory(
        detections=detections,
        bat_speed_pixels_per_frame=[15.0] * 14,
        tracking_accuracy=0.95,
        tracking_failures=[],
    )


class TestDrawSkeleton:
    """Tests for _draw_skeleton method."""

    def test_skeleton_drawing_modifies_frame(
        self, renderer: OverlayRenderer, sample_frame: np.ndarray, sample_pose: PoseResult
    ):
        """Skeleton drawing should produce a modified frame (not all zeros)."""
        original = sample_frame.copy()
        result = renderer._draw_skeleton(sample_frame, sample_pose)

        # Frame should be modified (not identical to original black frame)
        assert not np.array_equal(result, original)

    def test_skeleton_drawing_preserves_frame_shape(
        self, renderer: OverlayRenderer, sample_frame: np.ndarray, sample_pose: PoseResult
    ):
        """Skeleton drawing should not change frame dimensions."""
        result = renderer._draw_skeleton(sample_frame, sample_pose)
        assert result.shape == sample_frame.shape

    def test_skeleton_with_empty_keypoints(
        self, renderer: OverlayRenderer, sample_frame: np.ndarray
    ):
        """Skeleton drawing with empty keypoints should return unmodified frame."""
        empty_pose = PoseResult(
            frame_index=0,
            keypoints=[],
            person_id=1,
            is_primary_batter=True,
            overall_confidence=0.0,
            is_low_confidence=True,
        )
        original = sample_frame.copy()
        result = renderer._draw_skeleton(sample_frame, empty_pose)
        assert np.array_equal(result, original)

    def test_skeleton_with_low_confidence_keypoints(
        self, renderer: OverlayRenderer, sample_frame: np.ndarray
    ):
        """Keypoints with very low confidence (<0.1) should not be drawn."""
        low_conf_pose = PoseResult(
            frame_index=0,
            keypoints=[
                Keypoint(x=0.5, y=0.5, z=0.0, confidence=0.05, name="nose"),
                Keypoint(x=0.4, y=0.4, z=0.0, confidence=0.05, name="left_eye"),
            ],
            person_id=1,
            is_primary_batter=True,
            overall_confidence=0.05,
            is_low_confidence=True,
        )
        original = sample_frame.copy()
        result = renderer._draw_skeleton(sample_frame, low_conf_pose)
        # Very low confidence keypoints should not be drawn
        assert np.array_equal(result, original)

    def test_skeleton_color_coding_high_confidence(
        self, renderer: OverlayRenderer, sample_frame: np.ndarray
    ):
        """High confidence keypoints should be drawn in green."""
        pose = PoseResult(
            frame_index=0,
            keypoints=[
                Keypoint(x=0.5, y=0.5, z=0.0, confidence=0.9, name="left_shoulder"),
            ],
            person_id=1,
            is_primary_batter=True,
            overall_confidence=0.9,
            is_low_confidence=False,
        )
        result = renderer._draw_skeleton(sample_frame, pose)
        # Check that green pixels exist (high confidence = green)
        green_pixels = np.where(
            (result[:, :, 1] > 200) & (result[:, :, 0] < 50) & (result[:, :, 2] < 50)
        )
        assert len(green_pixels[0]) > 0


class TestDrawBatTrajectory:
    """Tests for _draw_bat_trajectory method."""

    def test_bat_trajectory_drawing_modifies_frame(
        self,
        renderer: OverlayRenderer,
        sample_frame: np.ndarray,
        sample_bat_trajectory: BatTrajectory,
    ):
        """Bat trajectory drawing should produce a modified frame."""
        original = sample_frame.copy()
        result = renderer._draw_bat_trajectory(
            sample_frame, sample_bat_trajectory, current_frame=10
        )
        assert not np.array_equal(result, original)

    def test_bat_trajectory_preserves_frame_shape(
        self,
        renderer: OverlayRenderer,
        sample_frame: np.ndarray,
        sample_bat_trajectory: BatTrajectory,
    ):
        """Bat trajectory drawing should not change frame dimensions."""
        result = renderer._draw_bat_trajectory(
            sample_frame, sample_bat_trajectory, current_frame=5
        )
        assert result.shape == sample_frame.shape

    def test_bat_trajectory_with_empty_detections(
        self, renderer: OverlayRenderer, sample_frame: np.ndarray
    ):
        """Empty bat trajectory should return unmodified frame."""
        empty_trajectory = BatTrajectory(
            detections=[],
            bat_speed_pixels_per_frame=[],
            tracking_accuracy=0.0,
            tracking_failures=[],
        )
        original = sample_frame.copy()
        result = renderer._draw_bat_trajectory(
            sample_frame, empty_trajectory, current_frame=0
        )
        assert np.array_equal(result, original)

    def test_bat_trajectory_with_undetected_frame(
        self, renderer: OverlayRenderer, sample_frame: np.ndarray
    ):
        """Frame where bat is not detected should not draw bat segment."""
        trajectory = BatTrajectory(
            detections=[
                BatDetectionResult(
                    frame_index=0,
                    detected=False,
                    position=(0.0, 0.0),
                    orientation_angle=0.0,
                    length_pixels=0.0,
                    confidence=0.0,
                    is_predicted=False,
                )
            ],
            bat_speed_pixels_per_frame=[],
            tracking_accuracy=0.0,
            tracking_failures=[],
        )
        original = sample_frame.copy()
        result = renderer._draw_bat_trajectory(
            sample_frame, trajectory, current_frame=0
        )
        # Undetected bat should not modify the frame
        assert np.array_equal(result, original)

    def test_bat_trail_draws_multiple_positions(
        self,
        renderer: OverlayRenderer,
        sample_frame: np.ndarray,
        sample_bat_trajectory: BatTrajectory,
    ):
        """Trail should draw lines connecting previous bat positions."""
        # Render at frame 10 - should have trail from frames 1-10
        result = renderer._draw_bat_trajectory(
            sample_frame, sample_bat_trajectory, current_frame=10
        )
        # Frame should be modified (trail + current bat drawn)
        assert not np.array_equal(result, np.zeros_like(sample_frame))


class TestRenderOverlayVideo:
    """Tests for render_overlay_video method."""

    def test_output_video_is_valid(
        self,
        renderer: OverlayRenderer,
        sample_pose: PoseResult,
        sample_bat_trajectory: BatTrajectory,
    ):
        """Output video should be openable by OpenCV."""
        # Create 15 frames
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(15)]

        # Create pose sequence for all frames
        pose_sequence = []
        for i in range(15):
            pose = PoseResult(
                frame_index=i,
                keypoints=sample_pose.keypoints,
                person_id=1,
                is_primary_batter=True,
                overall_confidence=0.85,
                is_low_confidence=False,
            )
            pose_sequence.append(pose)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        try:
            result_path = renderer.render_overlay_video(
                frames=frames,
                pose_sequence=pose_sequence,
                bat_trajectory=sample_bat_trajectory,
                output_path=output_path,
                fps=30.0,
            )

            # Verify the output video can be opened
            cap = cv2.VideoCapture(result_path)
            assert cap.isOpened()

            # Verify frame count
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert frame_count == 15

            # Verify dimensions
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert width == 640
            assert height == 480

            cap.release()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_output_video_with_empty_frames(self, renderer: OverlayRenderer):
        """Empty frames list should still produce a valid output path."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        try:
            result_path = renderer.render_overlay_video(
                frames=[],
                pose_sequence=[],
                bat_trajectory=BatTrajectory(),
                output_path=output_path,
                fps=30.0,
            )
            assert result_path == output_path
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_output_video_with_empty_pose_and_trajectory(
        self, renderer: OverlayRenderer
    ):
        """Video with frames but no pose/trajectory should still be valid."""
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        try:
            result_path = renderer.render_overlay_video(
                frames=frames,
                pose_sequence=[],
                bat_trajectory=BatTrajectory(),
                output_path=output_path,
                fps=30.0,
            )

            # Video should still be valid
            cap = cv2.VideoCapture(result_path)
            assert cap.isOpened()
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert frame_count == 5
            cap.release()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_render_returns_output_path(
        self, renderer: OverlayRenderer, sample_pose: PoseResult
    ):
        """render_overlay_video should return the output path."""
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        try:
            result = renderer.render_overlay_video(
                frames=frames,
                pose_sequence=[],
                bat_trajectory=BatTrajectory(),
                output_path=output_path,
                fps=30.0,
            )
            assert result == output_path
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestUploadToS3:
    """Tests for _upload_to_s3 method."""

    def test_upload_to_s3_returns_key(self, renderer: OverlayRenderer):
        """_upload_to_s3 should return an S3 key."""
        # Create a temporary file to simulate a video
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, mode="wb"
        ) as tmp:
            tmp.write(b"fake video content")
            tmp_path = tmp.name

        try:
            mock_s3_client = MagicMock(spec=S3Client)
            mock_s3_client._client = MagicMock()
            mock_s3_client._bucket = "test-bucket"

            s3_key = renderer._upload_to_s3(tmp_path, mock_s3_client)

            # Key should start with overlays/ and end with .mp4
            assert s3_key.startswith("overlays/")
            assert s3_key.endswith(".mp4")

            # put_object should have been called
            mock_s3_client._client.put_object.assert_called_once()
            call_kwargs = mock_s3_client._client.put_object.call_args[1]
            assert call_kwargs["Bucket"] == "test-bucket"
            assert call_kwargs["Key"] == s3_key
            assert call_kwargs["ContentType"] == "video/mp4"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
