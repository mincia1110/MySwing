"""Video quality checker service.

Provides quality validation for uploaded video files including brightness,
swing arc framing, resolution, and frame rate stability checks.
Validates Requirements 9.1-9.8.
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from app.models.enums import QualityStatus
from app.models.video import QualityCheckResult

logger = logging.getLogger(__name__)

# Quality check thresholds
BRIGHTNESS_LUX_THRESHOLD: float = 40.0
BRIGHTNESS_PIXEL_TO_LUX_FACTOR: float = 0.5
SWING_ARC_VISIBILITY_THRESHOLD: float = 80.0
MIN_RESOLUTION_HEIGHT_QUALITY: int = 720
FRAME_RATE_VARIATION_THRESHOLD: float = 10.0
TARGET_FPS: float = 30.0
FRAME_SAMPLE_INTERVAL: int = 10  # Sample every 10th frame
PERSON_DETECTION_SAMPLE_LIMIT: int = 24
MULTIPLE_PERSON_FRAME_RATIO_THRESHOLD: float = 0.20
MULTIPLE_PERSON_MIN_FRAMES: int = 2
MULTIPLE_PERSON_RAW_CANDIDATE_MIN_FRAMES: int = 3
PERSON_COLUMN_GROUPING_RATIO: float = 0.15


class VideoQualityChecker:
    """Video quality checker that validates brightness, framing, resolution,
    and frame rate stability of uploaded videos.

    Quality checks must complete within 10 seconds (Requirement 9.8).
    """

    def check_quality(
        self,
        video_path: str,
        pose_results: Optional[list] = None,
    ) -> QualityCheckResult:
        """Run all quality checks on a video file and return a summary.

        Args:
            video_path: Path to the video file to check.
            pose_results: Optional pose estimation results for framing analysis.

        Returns:
            QualityCheckResult with status for each check and any warnings.
        """
        start_time = time.time()
        warnings: list[str] = []

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                logger.error("Cannot open video file for quality check: %s", video_path)
                return self._create_failure_result()

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample frames for brightness check
            frames = self._sample_frames(cap, frame_count)

            # 1. Brightness check (Requirement 9.1, 9.2)
            brightness_status, brightness_value = self._check_brightness(frames)
            if brightness_status == QualityStatus.WARNING:
                warnings.append(
                    "조명이 부족합니다. 포즈 감지 정확도가 저하될 수 있습니다. "
                    "더 밝은 환경에서 재촬영을 권장합니다."
                )

            # 2. Framing / swing arc visibility check (Requirement 9.3, 9.4)
            framing_status, swing_arc_visibility = self._check_framing(
                frames, pose_results
            )
            if framing_status == QualityStatus.WARNING:
                warnings.append(
                    "스윙 아크의 가시성이 부족합니다. "
                    "더 넓은 각도로 재촬영을 권장합니다."
                )

            # 3. Resolution check (Requirement 9.5)
            resolution_status = self._check_resolution(width, height)
            if resolution_status == QualityStatus.WARNING:
                warnings.append(
                    "해상도가 720p 미만입니다. 분석 정확도가 저하될 수 있습니다."
                )

            # 4. Frame rate stability check (Requirement 9.6, 9.7)
            frame_rate_status, frame_rate_variation = self._check_frame_rate_stability(
                video_path
            )
            if frame_rate_status == QualityStatus.WARNING:
                warnings.append(
                    "프레임레이트가 불안정합니다. 재촬영을 권장합니다."
                )

            if self._has_multiple_people(frames):
                warnings.append(
                    "프레임 안에 여러 사람이 감지되어 분석 대상이 모호할 수 있습니다. "
                    "타자 1명만 전신이 보이는 사이드뷰로 재촬영을 권장합니다."
                )

        finally:
            cap.release()

        elapsed = time.time() - start_time
        logger.info("Quality check completed in %.2f seconds", elapsed)

        return QualityCheckResult(
            brightness_status=brightness_status,
            framing_status=framing_status,
            resolution_status=resolution_status,
            frame_rate_stability_status=frame_rate_status,
            brightness_value=brightness_value,
            swing_arc_visibility_percent=swing_arc_visibility,
            frame_rate_variation_percent=frame_rate_variation,
            warnings=warnings,
        )

    def _sample_frames(
        self, cap: cv2.VideoCapture, frame_count: int
    ) -> list[np.ndarray]:
        """Sample every Nth frame from the video for analysis.

        Args:
            cap: OpenCV VideoCapture object (already opened).
            frame_count: Total number of frames in the video.

        Returns:
            List of sampled frames as numpy arrays.
        """
        frames: list[np.ndarray] = []
        frame_idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while frame_idx < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
            frame_idx += FRAME_SAMPLE_INTERVAL

        return frames

    def _check_brightness(
        self, frames: list[np.ndarray]
    ) -> tuple[QualityStatus, float]:
        """Check average frame brightness across sampled frames.

        Converts frames to grayscale, calculates mean pixel value,
        and maps to lux equivalent (mean_pixel_value * 0.5).
        Threshold: 40 lux equivalent (Requirement 9.1).

        Args:
            frames: List of sampled video frames.

        Returns:
            Tuple of (QualityStatus, brightness_lux_equivalent).
        """
        if not frames:
            return QualityStatus.WARNING, 0.0

        total_brightness = 0.0
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_pixel_value = float(np.mean(gray))
            total_brightness += mean_pixel_value

        avg_pixel_value = total_brightness / len(frames)
        lux_equivalent = avg_pixel_value * BRIGHTNESS_PIXEL_TO_LUX_FACTOR

        if lux_equivalent >= BRIGHTNESS_LUX_THRESHOLD:
            return QualityStatus.PASS, lux_equivalent
        else:
            return QualityStatus.WARNING, lux_equivalent

    def _check_framing(
        self,
        frames: list[np.ndarray],
        pose_results: Optional[list] = None,
    ) -> tuple[QualityStatus, float]:
        """Estimate swing arc visibility within the frame.

        If pose results are available, uses them to estimate the swing arc
        coverage. Otherwise, uses a heuristic based on frame content analysis.
        Threshold: ≥80% visibility (Requirement 9.3).

        Args:
            frames: List of sampled video frames.
            pose_results: Optional pose estimation results.

        Returns:
            Tuple of (QualityStatus, swing_arc_visibility_percent).
        """
        if not frames:
            return QualityStatus.WARNING, 0.0

        if pose_results is not None and len(pose_results) > 0:
            # Use pose results to estimate swing arc visibility
            visibility = self._estimate_visibility_from_poses(
                frames, pose_results
            )
        else:
            # Heuristic: estimate based on motion area coverage in frame
            visibility = self._estimate_visibility_heuristic(frames)

        if visibility >= SWING_ARC_VISIBILITY_THRESHOLD:
            return QualityStatus.PASS, visibility
        else:
            return QualityStatus.WARNING, visibility

    def _estimate_visibility_from_poses(
        self,
        frames: list[np.ndarray],
        pose_results: list,
    ) -> float:
        """Estimate swing arc visibility using pose keypoint data.

        Calculates what percentage of the estimated swing arc bounding box
        is within the frame boundaries.

        Args:
            frames: List of sampled video frames.
            pose_results: Pose estimation results with keypoint coordinates.

        Returns:
            Estimated visibility percentage (0-100).
        """
        if not frames or not pose_results:
            return 0.0

        frame_height, frame_width = frames[0].shape[:2]
        visible_count = 0
        total_count = 0

        for pose in pose_results:
            if not hasattr(pose, "keypoints"):
                continue
            for kp in pose.keypoints:
                total_count += 1
                # Check if keypoint is within frame bounds (normalized 0-1)
                if hasattr(kp, "x") and hasattr(kp, "y"):
                    if 0.0 <= kp.x <= 1.0 and 0.0 <= kp.y <= 1.0:
                        visible_count += 1

        if total_count == 0:
            return self._estimate_visibility_heuristic(frames)

        return (visible_count / total_count) * 100.0

    def _estimate_visibility_heuristic(
        self, frames: list[np.ndarray]
    ) -> float:
        """Estimate swing arc visibility using motion-based heuristic.

        Analyzes frame differences to detect motion area and estimates
        whether the full swing arc is captured within the frame.

        Args:
            frames: List of sampled video frames.

        Returns:
            Estimated visibility percentage (0-100).
        """
        if len(frames) < 2:
            # With only one frame, assume full visibility
            return 100.0

        frame_height, frame_width = frames[0].shape[:2]
        motion_pixels = set()

        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(prev_gray, curr_gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            # Find motion contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_pixels.add((x, y, w, h))

        if not motion_pixels:
            # No motion detected - assume full visibility
            return 100.0

        # Calculate bounding box of all motion
        all_x = []
        all_y = []
        all_x2 = []
        all_y2 = []
        for x, y, w, h in motion_pixels:
            all_x.append(x)
            all_y.append(y)
            all_x2.append(x + w)
            all_y2.append(y + h)

        min_x = min(all_x)
        min_y = min(all_y)
        max_x = max(all_x2)
        max_y = max(all_y2)

        # Check if motion bounding box is well within frame (with margin)
        margin = 0.05  # 5% margin from edges
        margin_x = int(frame_width * margin)
        margin_y = int(frame_height * margin)

        # If motion extends to edges, visibility may be reduced
        edge_violations = 0
        total_edges = 4

        if min_x <= margin_x:
            edge_violations += 1
        if min_y <= margin_y:
            edge_violations += 1
        if max_x >= frame_width - margin_x:
            edge_violations += 1
        if max_y >= frame_height - margin_y:
            edge_violations += 1

        # Estimate visibility: each edge violation reduces visibility
        visibility = 100.0 - (edge_violations / total_edges) * 30.0

        return max(0.0, min(100.0, visibility))

    def _check_resolution(self, width: int, height: int) -> QualityStatus:
        """Check if video resolution meets quality requirements.

        Produces a warning if resolution is below 720p (Requirement 9.5).

        Args:
            width: Video width in pixels.
            height: Video height in pixels.

        Returns:
            QualityStatus.PASS if ≥720p, QualityStatus.WARNING otherwise.
        """
        if height >= MIN_RESOLUTION_HEIGHT_QUALITY:
            return QualityStatus.PASS
        else:
            return QualityStatus.WARNING

    def _has_multiple_people(self, frames: list[np.ndarray]) -> bool:
        """Detect whether sampled frames likely contain more than one person.

        The swing pipeline assumes a single visible batter. When multiple people
        are present, pose estimation can switch subjects between frames and make
        the overlay unusable, so this is surfaced as a quality warning.
        """
        if not frames:
            return False

        if len(frames) <= PERSON_DETECTION_SAMPLE_LIMIT:
            sampled_frames = frames
        else:
            sample_indices = np.linspace(
                0,
                len(frames) - 1,
                PERSON_DETECTION_SAMPLE_LIMIT,
                dtype=int,
            )
            sampled_frames = [frames[index] for index in sample_indices]

        multi_person_frames = 0
        raw_multi_candidate_frames = 0

        for frame in sampled_frames:
            if self._detect_people_in_frame(frame) >= 2:
                multi_person_frames += 1
            elif self._detect_person_candidate_count(frame) >= 2:
                raw_multi_candidate_frames += 1

        if multi_person_frames >= MULTIPLE_PERSON_MIN_FRAMES:
            return True

        if raw_multi_candidate_frames >= MULTIPLE_PERSON_RAW_CANDIDATE_MIN_FRAMES:
            return True

        frame_ratio = multi_person_frames / len(sampled_frames)
        return frame_ratio >= MULTIPLE_PERSON_FRAME_RATIO_THRESHOLD

    def _detect_people_in_frame(self, frame: np.ndarray) -> int:
        """Return the number of person-like regions in a frame."""
        rect_list, frame_width = self._detect_person_rects(frame)
        if not rect_list:
            return 0

        return self._count_distinct_person_columns(rect_list, frame_width)

    def _detect_person_candidate_count(self, frame: np.ndarray) -> int:
        """Return the raw count of person-like detector rectangles."""
        rect_list, _frame_width = self._detect_person_rects(frame)
        return len(rect_list)

    def _detect_person_rects(self, frame: np.ndarray) -> tuple[list[list[int]], int]:
        """Return non-max-suppressed person detector rectangles."""
        frame_height, frame_width = frame.shape[:2]
        scale = 1.0
        if frame_width > 640:
            scale = 640.0 / frame_width
            resized = cv2.resize(
                frame,
                (640, max(1, int(frame_height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = frame

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05,
            hitThreshold=-0.5,
        )
        if len(rects) == 0:
            return [], frame_width

        rect_list = []
        for x, y, w, h in rects:
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)
            rect_list.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(
            rect_list,
            [float(weight) for weight in weights],
            score_threshold=0.3,
            nms_threshold=0.4,
        )
        if len(indices) == 0:
            return [], frame_width

        selected_rects = [rect_list[int(index)] for index in np.array(indices).flatten()]
        return selected_rects, frame_width

    def _count_distinct_person_columns(
        self,
        rects: list[list[int]],
        frame_width: int,
    ) -> int:
        """Count spatially distinct people from detector rectangles.

        HOG can split one batter into separate upper/lower-body rectangles.
        Rectangles whose horizontal centers are close are treated as the same
        person, which reduces false positives for single-player side views.
        """
        if not rects:
            return 0

        center_threshold = max(1.0, frame_width * PERSON_COLUMN_GROUPING_RATIO)
        centers = sorted(x + w / 2.0 for x, _y, w, _h in rects)
        groups = 1
        group_center = centers[0]

        for center in centers[1:]:
            if abs(center - group_center) > center_threshold:
                groups += 1
                group_center = center
            else:
                group_center = (group_center + center) / 2.0

        return groups

    def _check_frame_rate_stability(
        self, video_path: str
    ) -> tuple[QualityStatus, float]:
        """Measure frame rate variation throughout the video.

        Reads frame timestamps and calculates instantaneous fps between
        consecutive frames. Variation = (max_fps - min_fps) / target_fps * 100.
        Pass if variation < 10% (Requirement 9.6).

        Args:
            video_path: Path to the video file.

        Returns:
            Tuple of (QualityStatus, frame_rate_variation_percent).
        """
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return QualityStatus.WARNING, 100.0

            timestamps: list[float] = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_fps = cap.get(cv2.CAP_PROP_FPS)

            if target_fps <= 0:
                target_fps = TARGET_FPS

            # Read timestamps from frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(min(frame_count, 300)):  # Limit to 300 frames for performance
                ret = cap.grab()
                if not ret:
                    break
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamps.append(timestamp_ms)

            if len(timestamps) < 2:
                # Not enough frames to measure stability
                return QualityStatus.PASS, 0.0

            # Calculate instantaneous fps between consecutive frames
            instantaneous_fps: list[float] = []
            for i in range(1, len(timestamps)):
                dt_ms = timestamps[i] - timestamps[i - 1]
                if dt_ms > 0:
                    fps = 1000.0 / dt_ms
                    instantaneous_fps.append(fps)

            if not instantaneous_fps:
                return QualityStatus.PASS, 0.0

            max_fps = max(instantaneous_fps)
            min_fps = min(instantaneous_fps)

            # Variation = (max_fps - min_fps) / target_fps * 100
            variation = (max_fps - min_fps) / target_fps * 100.0

            if variation < FRAME_RATE_VARIATION_THRESHOLD:
                return QualityStatus.PASS, variation
            else:
                return QualityStatus.WARNING, variation

        finally:
            cap.release()

    def _create_failure_result(self) -> QualityCheckResult:
        """Create a failure result when the video cannot be opened.

        Returns:
            QualityCheckResult with all warnings set.
        """
        return QualityCheckResult(
            brightness_status=QualityStatus.WARNING,
            framing_status=QualityStatus.WARNING,
            resolution_status=QualityStatus.WARNING,
            frame_rate_stability_status=QualityStatus.WARNING,
            brightness_value=0.0,
            swing_arc_visibility_percent=0.0,
            frame_rate_variation_percent=100.0,
            warnings=["비디오 파일을 열 수 없습니다."],
        )
