"""Report generation module for swing analysis (Requirements 8.1-8.8).

Contains:
- OverlayRenderer: Renders pose skeleton and bat trajectory overlays on video (8.2)
- MetricsTableBuilder: Builds metrics table with color-coded ratings (8.3, 8.6)
- DrillRecommender: Recommends 1-3 drills per weakness (8.4)
- ComparisonViewBuilder: Builds user vs pro reference comparison view (8.5)
- TrendAnalyzer: Builds trend data from analysis history (8.7, 8.8)
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cv2
import numpy as np

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.enums import MetricRating
from app.models.evaluation import DrillRecommendation, ImprovementArea, MetricEvaluation
from app.models.pose import Keypoint, PoseResult
from app.models.report import MetricDataPoint, TrendData
from app.models.swing import SwingPhaseResult
from app.services.s3_client import S3Client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Overlay Renderer constants
# ---------------------------------------------------------------------------

# Skeleton connection definitions (pairs of keypoint names)
# Based on MediaPipe Pose landmark connections
SKELETON_CONNECTIONS: List[tuple[str, str]] = [
    # Head
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Left arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # Right arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Left leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    # Right leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Confidence thresholds for color coding
HIGH_CONFIDENCE_THRESHOLD = 0.7
MEDIUM_CONFIDENCE_THRESHOLD = 0.4

# Colors (BGR format for OpenCV)
COLOR_HIGH_CONFIDENCE = (0, 255, 0)  # Green
COLOR_MEDIUM_CONFIDENCE = (0, 255, 255)  # Yellow
COLOR_LOW_CONFIDENCE = (0, 0, 255)  # Red

# Drawing parameters
KEYPOINT_RADIUS = 4
SKELETON_LINE_THICKNESS = 2
BAT_LINE_THICKNESS = 3
BAT_TRAIL_MAX_FRAMES = 10


def _confidence_color(confidence: float) -> tuple[int, int, int]:
    """Get color based on confidence level.

    Args:
        confidence: Confidence score between 0 and 1.

    Returns:
        BGR color tuple.
    """
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return COLOR_HIGH_CONFIDENCE
    elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return COLOR_MEDIUM_CONFIDENCE
    else:
        return COLOR_LOW_CONFIDENCE


class OverlayRenderer:
    """Renders pose skeleton and bat trajectory overlays on video frames.

    Generates an overlay video with:
    - Pose skeleton connections drawn as colored lines
    - Keypoint circles at each joint
    - Bat position as a line segment
    - Bat trailing path (last 10 frames) as fading lines

    Validates: Requirements 8.2
    """

    def render_overlay_video(
        self,
        frames: List[np.ndarray],
        pose_sequence: List[PoseResult],
        bat_trajectory: BatTrajectory,
        output_path: str,
        fps: float = 30.0,
    ) -> str:
        """Render overlay video with pose skeleton and bat trajectory.

        Args:
            frames: List of video frames as numpy arrays (BGR).
            pose_sequence: List of PoseResult for each frame.
            bat_trajectory: BatTrajectory with detection results.
            output_path: Path to write the output video file.
            fps: Frames per second for the output video.

        Returns:
            The output_path where the video was written.
        """
        if not frames:
            logger.warning("No frames provided for overlay rendering")
            return output_path

        height, width = frames[0].shape[:2]
        # Use H.264 codec for browser compatibility; fall back to mp4v if unavailable
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            # Fallback to mp4v if H.264 encoder not available
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            # Build a lookup for pose results by frame index
            pose_by_frame: Dict[int, PoseResult] = {}
            for pose in pose_sequence:
                pose_by_frame[pose.frame_index] = pose

            for frame_idx, frame in enumerate(frames):
                overlay_frame = frame.copy()

                # Draw skeleton if pose data available for this frame
                pose = pose_by_frame.get(frame_idx)
                if pose is not None:
                    overlay_frame = self._draw_skeleton(overlay_frame, pose)

                # Draw bat trajectory
                overlay_frame = self._draw_bat_trajectory(
                    overlay_frame, bat_trajectory, frame_idx
                )

                writer.write(overlay_frame)
        finally:
            writer.release()

        # Re-encode to H.264 for browser compatibility
        temp_h264_path = output_path + ".h264.mp4"
        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", output_path,
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                temp_h264_path,
            ]
            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, timeout=60
            )
            if result.returncode == 0:
                os.replace(temp_h264_path, output_path)
                logger.info("Re-encoded overlay video to H.264: %s", output_path)
            else:
                logger.warning(
                    "ffmpeg H.264 re-encoding failed (returncode=%d), keeping mp4v version",
                    result.returncode,
                )
                if os.path.exists(temp_h264_path):
                    os.unlink(temp_h264_path)
        except FileNotFoundError:
            logger.warning("ffmpeg not found, overlay video remains in mp4v format")
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg re-encoding timed out, keeping mp4v version")
            if os.path.exists(temp_h264_path):
                os.unlink(temp_h264_path)

        logger.info(f"Overlay video written to: {output_path}")
        return output_path

    def _draw_skeleton(
        self, frame: np.ndarray, pose: PoseResult
    ) -> np.ndarray:
        """Draw pose skeleton on a frame.

        Draws keypoint circles and connection lines between keypoints.
        Colors are based on confidence: green (high), yellow (medium), red (low).

        Args:
            frame: The video frame to draw on (BGR numpy array).
            pose: PoseResult containing keypoints for this frame.

        Returns:
            The frame with skeleton overlay drawn.
        """
        if not pose.keypoints:
            return frame

        height, width = frame.shape[:2]

        # Build keypoint lookup by name
        kp_map: Dict[str, Keypoint] = {}
        for kp in pose.keypoints:
            kp_map[kp.name] = kp

        # Draw connections
        for start_name, end_name in SKELETON_CONNECTIONS:
            start_kp = kp_map.get(start_name)
            end_kp = kp_map.get(end_name)

            if start_kp is None or end_kp is None:
                continue

            # Skip low-confidence keypoints (below minimum threshold)
            if start_kp.confidence < 0.1 or end_kp.confidence < 0.1:
                continue

            # Convert normalized coordinates to pixel coordinates
            start_pt = (int(start_kp.x * width), int(start_kp.y * height))
            end_pt = (int(end_kp.x * width), int(end_kp.y * height))

            # Use the lower confidence of the two keypoints for line color
            min_confidence = min(start_kp.confidence, end_kp.confidence)
            color = _confidence_color(min_confidence)

            cv2.line(frame, start_pt, end_pt, color, SKELETON_LINE_THICKNESS)

        # Draw keypoint circles
        for kp in pose.keypoints:
            if kp.confidence < 0.1:
                continue

            pt = (int(kp.x * width), int(kp.y * height))
            color = _confidence_color(kp.confidence)
            cv2.circle(frame, pt, KEYPOINT_RADIUS, color, -1)

        return frame

    def _draw_bat_trajectory(
        self,
        frame: np.ndarray,
        bat_trajectory: BatTrajectory,
        current_frame: int,
    ) -> np.ndarray:
        """Draw bat position and trailing path on a frame.

        Draws the bat as a line segment using its position, orientation, and length.
        Also draws a trailing path from the last 10 frames as fading lines.

        Args:
            frame: The video frame to draw on (BGR numpy array).
            bat_trajectory: BatTrajectory with all detection results.
            current_frame: The current frame index being rendered.

        Returns:
            The frame with bat trajectory overlay drawn.
        """
        if not bat_trajectory.detections:
            return frame

        # Build detection lookup by frame index
        det_by_frame: Dict[int, BatDetectionResult] = {}
        for det in bat_trajectory.detections:
            det_by_frame[det.frame_index] = det

        # Draw trailing path (last BAT_TRAIL_MAX_FRAMES frames)
        trail_start = max(0, current_frame - BAT_TRAIL_MAX_FRAMES + 1)
        trail_positions: List[tuple[int, int]] = []

        for trail_frame in range(trail_start, current_frame + 1):
            det = det_by_frame.get(trail_frame)
            if det is not None and det.detected:
                center = (int(det.position[0]), int(det.position[1]))
                trail_positions.append(center)

        # Draw trail as fading lines
        if len(trail_positions) >= 2:
            for i in range(1, len(trail_positions)):
                # Fade alpha based on position in trail (older = more faded)
                alpha = int(255 * (i / len(trail_positions)))
                trail_color = (alpha, 165, 0)  # Fading orange trail
                thickness = max(1, BAT_LINE_THICKNESS - 1)
                cv2.line(
                    frame,
                    trail_positions[i - 1],
                    trail_positions[i],
                    trail_color,
                    thickness,
                )

        # Draw current bat position as a line segment
        current_det = det_by_frame.get(current_frame)
        if current_det is not None and current_det.detected:
            self._draw_bat_segment(frame, current_det)

        return frame

    def _draw_bat_segment(
        self, frame: np.ndarray, detection: BatDetectionResult
    ) -> None:
        """Draw the bat as a line segment based on position, orientation, and length.

        Args:
            frame: The video frame to draw on.
            detection: BatDetectionResult with position, angle, and length.
        """
        cx, cy = detection.position
        angle_rad = math.radians(detection.orientation_angle)
        half_length = detection.length_pixels / 2.0

        # Calculate endpoints of the bat line segment
        dx = half_length * math.cos(angle_rad)
        dy = half_length * math.sin(angle_rad)

        pt1 = (int(cx - dx), int(cy - dy))
        pt2 = (int(cx + dx), int(cy + dy))

        # Draw bat as a cyan line
        bat_color = (255, 255, 0)  # Cyan in BGR
        cv2.line(frame, pt1, pt2, bat_color, BAT_LINE_THICKNESS)

        # Draw small circles at bat endpoints
        cv2.circle(frame, pt1, 3, bat_color, -1)
        cv2.circle(frame, pt2, 3, bat_color, -1)

    def _upload_to_s3(self, local_path: str, s3_client: S3Client) -> str:
        """Upload the overlay video to S3 and return the S3 key.

        Args:
            local_path: Path to the local video file.
            s3_client: S3Client instance for upload operations.

        Returns:
            The S3 key where the video was uploaded.
        """
        filename = os.path.basename(local_path)
        s3_key = f"overlays/{uuid4()}/{filename}"

        with open(local_path, "rb") as f:
            video_bytes = f.read()

        s3_client._client.put_object(
            Bucket=s3_client._bucket,
            Key=s3_key,
            Body=video_bytes,
            ContentType="video/mp4",
        )

        logger.info(f"Overlay video uploaded to S3: {s3_key}")
        return s3_key


# Unit display mapping for formatting measured values
_UNIT_DISPLAY: Dict[str, str] = {
    "km/h": "km/h",
    "degrees": "°",
    "°": "°",
    "ms": "ms",
    "ratio": "",
    "dps": "°/s",
    "%": "%",
}


class MetricsTableBuilder:
    """Builds a metrics table with color-coded ratings (Requirements 8.3, 8.6).

    Generates a tabular representation where each row contains:
    - metric_name: Name of the metric
    - measured_value: Formatted value with unit (e.g., "108.5 km/h", "12.3°")
    - color_code: Color indicator ("green", "yellow", "red")
    - rating: Human-readable rating string
    """

    # Mapping from color_code to human-readable rating
    _RATING_LABELS: Dict[str, str] = {
        "green": "optimal",
        "yellow": "acceptable",
        "red": "outside",
    }

    def build_metrics_table(
        self, evaluations: List[MetricEvaluation]
    ) -> List[Dict[str, Any]]:
        """Build a metrics table from metric evaluations.

        Each row contains the metric name, formatted measured value with unit,
        color code, and a human-readable rating label.

        Args:
            evaluations: List of MetricEvaluation objects from swing evaluation.

        Returns:
            List of dictionaries, one per metric, each containing:
            - metric_name (str): Name of the metric
            - measured_value (str): Formatted value with unit
            - color_code (str): "green", "yellow", or "red"
            - rating (str): "optimal", "acceptable", or "outside"
        """
        table: List[Dict[str, Any]] = []

        for evaluation in evaluations:
            row = {
                "metric_name": evaluation.metric_name,
                "measured_value": self._format_value(
                    evaluation.measured_value, evaluation.unit
                ),
                "color_code": evaluation.color_code,
                "rating": self._RATING_LABELS.get(
                    evaluation.color_code, "unknown"
                ),
            }
            table.append(row)

        return table

    @staticmethod
    def _format_value(value: float, unit: str) -> str:
        """Format a measured value with its unit for display.

        Applies appropriate formatting based on the unit type:
        - km/h: "108.5 km/h"
        - degrees/°: "12.3°"
        - ms: "45.2 ms"
        - ratio: "0.85" (no unit suffix)
        - dps: "450.0°/s"
        - %: "15.2%"

        Args:
            value: The numeric measured value.
            unit: The unit string from MetricEvaluation.

        Returns:
            Formatted string with value and unit.
        """
        display_unit = _UNIT_DISPLAY.get(unit, unit)

        # Format the numeric value
        if unit == "ratio":
            # Ratio values displayed with 2 decimal places, no unit
            return f"{value:.2f}"
        elif unit in ("degrees", "°"):
            # Degree values with unit symbol directly attached
            return f"{value:.1f}{display_unit}"
        elif unit == "dps":
            # Degrees per second with unit directly attached
            return f"{value:.1f}{display_unit}"
        elif unit == "%":
            # Percentage with symbol directly attached
            return f"{value:.1f}{display_unit}"
        elif unit == "ms":
            # Milliseconds with space before unit
            return f"{value:.1f} {display_unit}"
        elif unit == "km/h":
            # Speed with space before unit
            return f"{value:.1f} {display_unit}"
        else:
            # Default: value with space and unit
            if display_unit:
                return f"{value:.1f} {display_unit}"
            return f"{value:.1f}"

# In-memory drill database mapping metric names to available drills.
#
# Each metric entry has two directions:
#   - "below": 지표가 기준 범위 미달(under target)일 때의 교정 드릴
#   - "above": 지표가 기준 범위 초과(over target)일 때의 교정 드릴
#
# Notes:
#   * missing_direction_label은 DrillRecommender가 잘못된 방향의 드릴을 추천하지 않도록
#     fallback 동작을 명시한다.
#   * 모든 metric/direction이 비어있으면 추천을 비활성화해 사용자에게 일반 드릴을
#     노출하지 않는다 (한국어 일반 메시지 또한 metric/direction 라벨에 한정).
DRILL_DATABASE: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    "bat_speed": {
        "below": [
            {
                "drill_name": "오버로드 배트 티 타격",
                "description": (
                    "무거운 배트로 티 타격을 반복해 오버로드 원리로 임팩트 시 "
                    "배트 스피드를 끌어올린다."
                ),
            },
            {
                "drill_name": "로테이션 미디신볼 던지기",
                "description": (
                    "코어 로테이션 파워를 키우기 위해 미디신볼을 회전하며 던지는 "
                    "드릴을 수행한다."
                ),
            },
            {
                "drill_name": "저항 밴드 스윙",
                "description": (
                    "저항 밴드를 부착한 상태에서 스윙을 반복해 엉덩이/몸통의 "
                    "폭발적 회전을 강화한다."
                ),
            },
        ],
        "above": [
            {
                "drill_name": "컨트롤 스윙 드릴",
                "description": (
                    "최대 속도 대신 배트 컨트롤과 컨택 품질에 집중해 과도한 "
                    "스피드를 안정화한다."
                ),
            },
            {
                "drill_name": "소프트 토스 정확도 드릴",
                "description": (
                    "정해진 존을 노려 칠 수 있도록 소프트 토스를 반복해 통제된 "
                    "배트 스피드를 만든다."
                ),
            },
        ],
    },
    "attack_angle": {
        "below": [
            {
                "drill_name": "로우 티 드릴",
                "description": (
                    "무릎 높이에 티를 설치해 어퍼 스윙 경로로 일관되게 "
                    "타격하는 연습을 한다."
                ),
            },
            {
                "drill_name": "상향 스윙 패스 드릴",
                "description": (
                    "공이 들어오는 궤도와 맞도록 배트가 살짝 위로 올라가는 "
                    "스윙 경로를 반복해 익힌다."
                ),
            },
            {
                "drill_name": "다단계 티 높이 드릴",
                "description": (
                    "서로 다른 티 높이를 번갈아 치며 스트라이크 존 전 영역에서 "
                    "일관된 공격각을 만든다."
                ),
            },
        ],
        "above": [
            {
                "drill_name": "레벨 스윙 드릴",
                "description": (
                    "과도한 어퍼컷을 줄이기 위해 존 전체에서 배트 패스를 "
                    "수평으로 유지하는 연습을 한다."
                ),
            },
            {
                "drill_name": "탑 핸드 컨트롤 드릴",
                "description": (
                    "탑 핸드 컨트롤을 강화해 스윙 플레인을 평평하게 만들어 "
                    "뜬공을 줄인다."
                ),
            },
            {
                "drill_name": "하이 티 라인드라이브",
                "description": (
                    "가슴 높이로 티를 세팅하고 라인드라이브를 노려 플라이볼이 "
                    "아닌 컨택을 유도한다."
                ),
            },
        ],
    },
    "hip_shoulder_separation": {
        "below": [
            {
                "drill_name": "힙 리드 드릴",
                "description": (
                    "어깨를 뒤로 유지한 채 골반 회전으로 스윙을 시작하도록 "
                    "반복 연습한다."
                ),
            },
            {
                "drill_name": "힙-숄더 분리 스트레칭 루틴",
                "description": (
                    "힙-숄더 분리를 늘리기 위한 다이나믹 스트레칭 루틴을 "
                    "매일 수행한다."
                ),
            },
            {
                "drill_name": "타월 드릴",
                "description": (
                    "가슴 앞에 타월을 감고 회전 중 적절한 분리가 유지되는지 "
                    "체감하는 드릴이다."
                ),
            },
        ],
        "above": [
            {
                "drill_name": "커넥션 드릴",
                "description": (
                    "과도한 분리를 줄이기 위해 상체와 하체가 분리되지 않도록 "
                    "연결성을 유지한다."
                ),
            },
            {
                "drill_name": "동기화 회전 드릴",
                "description": (
                    "골반과 어깨가 과하게 분리되지 않고 함께 회전하도록 "
                    "부드러운 회전 리듬을 반복한다."
                ),
            },
        ],
    },
    "hand_path_efficiency": {
        "below": [
            {
                "drill_name": "다이렉트 패스 티 드릴",
                "description": (
                    "손을 볼로 직접 보내도록 의식하고, 불필요한 움직임을 "
                    "최소화한 채 반복한다."
                ),
            },
            {
                "drill_name": "숏 배트 드릴",
                "description": (
                    "짧은 배트로 스윙해 컴팩트하고 효율적인 핸드 패스를 "
                    "강제한다."
                ),
            },
            {
                "drill_name": "커넥션 볼 드릴",
                "description": (
                    "팔과 토르소 사이에 볼을 끼워 스윙 내내 연결감을 "
                    "유지하도록 한다."
                ),
            },
        ],
        "above": [
            {
                "drill_name": "팔 뻗기 익스텐션 드릴",
                "description": (
                    "컨택 이후 팔을 충분히 뻗어 배트가 공을 따라가는 구간을 "
                    "길게 만들고 힘 전달을 높인다."
                ),
            },
        ],
    },
    "stride_length_cm": {
        "below": [
            {
                "drill_name": "스트라이드 확장 드릴",
                "description": (
                    "앞발을 목표 지점까지 안정적으로 내딛는 연습을 반복해 "
                    "스윙 중 충분한 보폭을 만든다."
                ),
            },
            {
                "drill_name": "라인 스텝 드릴",
                "description": (
                    "지면에 표시한 라인을 따라 앞발을 내딛으며 체중 이동과 "
                    "스트라이드 방향을 함께 익힌다."
                ),
            },
        ],
        "above": [
            {
                "drill_name": "보폭 줄이기 컨트롤 드릴",
                "description": (
                    "평소보다 짧은 보폭으로 스윙해 앞발이 과하게 멀리 나가지 "
                    "않도록 착지 위치를 조절한다."
                ),
            },
            {
                "drill_name": "균형 착지 드릴",
                "description": (
                    "앞발 착지 후 몸이 앞으로 쏠리지 않도록 멈춤 동작을 넣어 "
                    "균형 잡힌 스트라이드를 만든다."
                ),
            },
        ],
    },
    "cog_sway_cm": {
        "below": [
            {
                "drill_name": "바닥 라인 밸런스 드릴",
                "description": (
                    "지면에 라인을 그어 스윙 중 머리/어깨가 라인 위에 머무르도록 "
                    "정렬 안정성을 훈련한다."
                ),
            },
            {
                "drill_name": "미니밴드 힙 안정화 드릴",
                "description": (
                    "골반 주변에 미니밴드를 걸고 코어를 고정한 채 스윙해 "
                    "수평 흔들림을 줄인다."
                ),
            },
        ],
        "above": [
            {
                "drill_name": "힙 텐션 릴리즈 드릴",
                "description": (
                    "무릎을 살짝 굽힌 채 머리/코어 정렬을 유지하도록 호흡과 "
                    "리듬을 맞추는 드릴로 과도한 좌우 흔들림을 줄인다."
                ),
            },
            {
                "drill_name": "스텝 백 풋워크 드릴",
                "description": (
                    "스턴스 폭과 스텝 길이를 점진적으로 줄여 스윙 중 무게중심이 "
                    "옆으로 과도하게 이동하지 않도록 교정한다."
                ),
            },
        ],
    },
    "cog_drop_cm": {
        "below": [
            {
                "drill_name": "뒷다리 드라이브 드릴",
                "description": (
                    "뒷다리와 엉덩이 힘을 쓰는 동작을 반복해 로드 구간에서 "
                    "안정적으로 체중을 낮추고 하체 힘을 유지한다."
                ),
            },
            {
                "drill_name": "힙 힌지 리프트 드릴",
                "description": (
                    "힙 힌지를 활용한 점진적 하강-상승 동작으로 임팩트 시 적정 "
                    "수직 움직임을 만든다."
                ),
            },
        ],
        "above": [
            {
                "drill_name": "코어 안정화 플랭크",
                "description": (
                    "플랭크와 사이드 플랭크 변형으로 코어 안정성을 강화해 "
                    "무게중심이 과하게 낮아지는 동작을 줄인다."
                ),
            },
            {
                "drill_name": "미니 스쿼트 자세 유지 드릴",
                "description": (
                    "무릎 굴곡을 일정 범위에서 유지한 채 스윙을 반복해 "
                    "무게중심의 과도한 수직 하강을 방지한다."
                ),
            },
        ],
    },
    "head_stability_cm": {
        "below": [
            {
                "drill_name": "시선 고정 드릴",
                "description": (
                    "공과 임팩트 지점을 끝까지 보도록 시선을 고정해 머리 이탈을 "
                    "줄인다."
                ),
            },
            {
                "drill_name": "체어 로테이션 드릴",
                "description": (
                    "의자에 앉아 회전 동작만 분리해 연습해 상체/머리의 "
                    "불필요한 흔들림을 줄인다."
                ),
            },
        ],
        "above": [
            {
                "drill_name": "스쿼트 스윙 안정화 드릴",
                "description": (
                    "얕은 스쿼트 자세에서 스윙을 반복해 머리/상체 정렬을 "
                    "강제하고 과도한 움직임을 교정한다."
                ),
            },
            {
                "drill_name": "바닥 마커 트래킹 드릴",
                "description": (
                    "지면에 표시한 마커를 응시한 채 스윙해 머리 이동 폭을 "
                    "시각적으로 교정한다."
                ),
            },
        ],
    },
    "front_knee_flexion_degrees": {
        "below": [
            {
                "drill_name": "로우 포지션 홀드 드릴",
                "description": (
                    "임팩트 직전 앞무릎 굴곡을 유지하도록 정지 동작을 반복해 "
                    "스트라이드 착지 시 굴곡을 확보한다."
                ),
            },
            {
                "drill_name": "싱글 레그 스쿼트",
                "description": (
                    "한 발 스쿼트로 앞발 지지를 강화해 임팩트 시 무릎 굴곡을 "
                    "증가시킨다."
                ),
            },
        ],
        "above": [
            {
                "drill_name": "업라이트 포지션 리프트 드릴",
                "description": (
                    "앞무릎이 과도하게 굽혀지지 않도록 상체를 살짝 일으키며 "
                    "스트라이드 길이를 조정하는 드릴이다."
                ),
            },
            {
                "drill_name": "스트라이드 길이 컨트롤",
                "description": (
                    "스트라이드 거리를 점진적으로 줄여 임팩트 시 앞무릎 "
                    "굴곡이 과도해지지 않도록 조절한다."
                ),
            },
        ],
    },
    "spine_angle_degrees": {
        "below": [
            {
                "drill_name": "몸통 기울임 드릴",
                "description": (
                    "로드 시점에 몸통이 적절히 기울어진 자세를 만들 수 있도록 "
                    "셋업과 로드 동작을 천천히 반복한다."
                ),
            },
            {
                "drill_name": "힙 힌지 무브먼트",
                "description": (
                    "힙 힌지를 활용해 몸통을 기울이는 동작을 반복하며 로드 시 "
                    "적절한 척추 경사각을 만든다."
                ),
            },
        ],
        "above": [
            {
                "drill_name": "업라이트 셋업 드릴",
                "description": (
                    "스탠스에서 척추가 수직에 가깝도록 정렬을 교정해 과도한 "
                    "전방 경사를 줄인다."
                ),
            },
            {
                "drill_name": "코어 앵글 드릴",
                "description": (
                    "코어 각도를 시각적으로 체크하며 셋업-로드 동작을 반복해 "
                    "척추 기울기 상한을 관리한다."
                ),
            },
        ],
    },
}

# Metric 중 일부(예: stride_length_cm)는 미달/초과가 동일 동작을 의미하지 않을 수 있어
# DrillRecommender가 한 방향의 매핑만 가진 경우 반대 방향 fallback을 절대 사용하지 않는다.
# 이 집합은 DrillRecommender가 잘못된 방향의 드릴을 노출하지 않도록 보호한다.
PROTECTED_DIRECTIONAL_METRICS: frozenset[str] = frozenset(
    {
        "bat_speed",
        "attack_angle",
        "hip_shoulder_separation",
        "hand_path_efficiency",
        "stride_length_cm",
        "cog_sway_cm",
        "cog_drop_cm",
        "head_stability_cm",
        "front_knee_flexion_degrees",
        "spine_angle_degrees",
    }
)



class DrillRecommender:
    """Recommends drills based on identified improvement areas.

    For each improvement area, recommends 1-3 drills from the predefined
    drill database based on the target metric and whether the value is
    above or below the reference range.

    Validates: Requirements 8.4
    """

    def __init__(self, drill_database: Dict[str, Dict[str, List[Dict[str, str]]]] | None = None):
        """Initialize with optional custom drill database.

        Args:
            drill_database: Optional custom drill database. Uses default if None.
        """
        self._drill_database = drill_database if drill_database is not None else DRILL_DATABASE

    def recommend_drills(
        self, improvements: List[ImprovementArea]
    ) -> List[DrillRecommendation]:
        """Recommend 1-3 drills for each identified improvement area.

        Selects drills based on whether the metric is above or below range.

        Args:
            improvements: List of improvement areas ranked by deviation.

        Returns:
            List of DrillRecommendation objects, 1-3 per improvement area.

        Notes:
            * 미달/초과에 따라 다른 방법을 추천한다.
            * 보호 대상 metric(`PROTECTED_DIRECTIONAL_METRICS`)에 대해 한 방향 매핑이
              비어있으면 반대 방향을 fallback으로 사용하지 않고 한국어 "방향 미정의"
              안내 한 건만 노출한다.
        """
        recommendations: List[DrillRecommendation] = []

        for improvement in improvements:
            metric_name = improvement.metric_name
            metric_drills = self._drill_database.get(metric_name)
            metric_label = self._format_metric_label(metric_name)

            # 해당 metric 자체가 매핑에 없으면 일반 안내로 폴백한다.
            if not metric_drills:
                # If no drills found for this metric, provide a generic recommendation
                recommendations.append(
                    self._build_generic_recommendation(
                        metric_name, metric_label, "generic"
                    )
                )
                continue

            direction = "above" if improvement.rating == "above_range" else "below"
            drills = metric_drills.get(direction, [])
            direction_label = "기준 초과" if direction == "above" else "기준 미달"

            # 보호 대상 metric은 반대 방향 fallback을 사용하지 않는다.
            if not drills and metric_name in PROTECTED_DIRECTIONAL_METRICS:
                recommendations.append(
                    self._build_generic_recommendation(
                        metric_name,
                        metric_label,
                        direction,
                        suffix=f" ({metric_label} {direction_label} 전용 드릴 미정)",
                    )
                )
                continue

            if not drills:
                drills = metric_drills.get(
                    "below" if direction == "above" else "above", []
                )

            if not drills:
                recommendations.append(
                    self._build_generic_recommendation(
                        metric_name, metric_label, "generic"
                    )
                )
                continue

            # Recommend 1-3 drills based on deviation severity
            num_drills = self._determine_drill_count(improvement.deviation_percent)
            selected_drills = drills[:num_drills]

            for drill_info in selected_drills:
                recommendations.append(
                    DrillRecommendation(
                        drill_name=drill_info["drill_name"],
                        target_metric=metric_name,
                        description=drill_info["description"],
                        direction=direction,
                    )
                )

        return recommendations

    @staticmethod
    def _format_metric_label(metric_name: str) -> str:
        """사람이 읽기 좋은 한국어 metric 라벨을 반환한다."""
        labels = {
            "bat_speed": "배트 스피드",
            "attack_angle": "공격각",
            "hip_shoulder_separation": "힙-숄더 분리",
            "hand_path_efficiency": "핸드 패스 효율",
            "stride_length_cm": "스트라이드 길이",
            "cog_sway_cm": "무게중심 좌우 흔들림",
            "cog_drop_cm": "무게중심 수직 하강",
            "head_stability_cm": "머리 안정성",
            "front_knee_flexion_degrees": "앞무릎 굴곡",
            "spine_angle_degrees": "척추 경사각",
        }
        if metric_name in labels:
            return labels[metric_name]
        return metric_name.replace("_", " ").title()

    @classmethod
    def _build_generic_recommendation(
        cls,
        metric_name: str,
        metric_label: str,
        direction: str,
        suffix: str = "",
    ) -> DrillRecommendation:
        """방향 비정의/미매핑 케이스에서 사용할 한국어 일반 안내."""
        if direction == "above":
            base = (
                f"{metric_label}이(가) 기준 범위를 초과했습니다. "
                "코치/영상 분석을 통해 과도한 동작을 줄이는 방향의 훈련을 "
                "추가로 설계해 주세요."
            )
        elif direction == "below":
            base = (
                f"{metric_label}이(가) 기준 범위에 미달합니다. "
                "코치/영상 분석을 통해 부족한 동작을 보강하는 방향의 훈련을 "
                "추가로 설계해 주세요."
            )
        else:
            base = (
                f"{metric_label}에 대한 전용 드릴 데이터가 아직 없습니다. "
                "코치와 함께 맞춤 훈련 계획을 설계해 주세요."
            )
        description = base + suffix
        return DrillRecommendation(
            drill_name="맞춤 훈련 설계 필요",
            target_metric=metric_name,
            description=description,
            direction=direction,
        )

    def _determine_drill_count(self, deviation_percent: float) -> int:
        """Determine how many drills to recommend based on deviation severity.

        Args:
            deviation_percent: The deviation percentage from the reference range.

        Returns:
            Number of drills to recommend (1-3).
        """
        if deviation_percent >= 30.0:
            return 3
        elif deviation_percent >= 15.0:
            return 2
        else:
            return 1


class ComparisonViewBuilder:
    """Builds a comparison view between user swing and professional reference.

    Creates a side-by-side phase duration comparison highlighting
    differences between the user's swing and a reference swing.

    Validates: Requirements 8.5
    """

    def build_comparison(
        self,
        user_phases: SwingPhaseResult,
        reference_phases: SwingPhaseResult,
    ) -> Dict[str, Any]:
        """Build a comparison view between user and reference swing phases.

        Args:
            user_phases: The user's swing phase classification result.
            reference_phases: The professional reference swing phase result.

        Returns:
            Dictionary containing the comparison view data with phase-by-phase
            duration comparison and highlighted differences.
        """
        phase_comparisons: List[Dict[str, Any]] = []

        # Get all phases from both user and reference
        all_phases = list(user_phases.phase_durations_ms.keys()) + [
            p for p in reference_phases.phase_durations_ms.keys()
            if p not in user_phases.phase_durations_ms
        ]

        for phase in all_phases:
            user_duration = user_phases.phase_durations_ms.get(phase)
            reference_duration = reference_phases.phase_durations_ms.get(phase)

            difference_ms: Optional[float] = None
            difference_percent: Optional[float] = None
            if user_duration is not None and reference_duration is not None and reference_duration > 0:
                difference_ms = user_duration - reference_duration
                difference_percent = (difference_ms / reference_duration) * 100.0

            phase_comparisons.append(
                {
                    "phase": phase.value,
                    "user_duration_ms": user_duration,
                    "reference_duration_ms": reference_duration,
                    "difference_ms": difference_ms,
                    "difference_percent": round(difference_percent, 1) if difference_percent is not None else None,
                    "is_significant": abs(difference_percent) > 20.0 if difference_percent is not None else False,
                }
            )

        # Build total duration comparison
        user_total = sum(
            d for d in user_phases.phase_durations_ms.values() if d is not None
        )
        reference_total = sum(
            d for d in reference_phases.phase_durations_ms.values() if d is not None
        )

        return {
            "phase_comparisons": phase_comparisons,
            "user_total_duration_ms": user_total,
            "reference_total_duration_ms": reference_total,
            "total_difference_ms": user_total - reference_total,
            "significant_differences": [
                pc for pc in phase_comparisons if pc["is_significant"]
            ],
        }


# Maximum number of recordings to include in trend data
MAX_TREND_RECORDINGS = 30

# Minimum total recordings (including current) required for trend data
MIN_RECORDINGS_FOR_TREND = 2


class TrendAnalyzer:
    """Builds trend data from analysis history for the report.

    Implements Requirements 8.7 and 8.8:
    - If total recordings (including current) >= 2: build TrendData with metrics_history
    - If total recordings < 2: return None
    - Limit to most recent 30 recordings in chronological order
    """

    def build_trend_data(
        self,
        analysis_history: list[dict],
        current_analysis: dict,
    ) -> Optional[TrendData]:
        """Build trend data from previous analysis history and current analysis.

        Args:
            analysis_history: List of previous analysis results (from DB query).
                Each dict should contain:
                - analysis_id: str
                - recorded_at: datetime
                - metrics: dict[str, dict] with keys like "bat_speed", "attack_angle", etc.
                  Each metric dict has: value (float), rating (str or MetricRating)
            current_analysis: The current analysis being generated.
                Same structure as items in analysis_history.

        Returns:
            TrendData if total recordings (history + current) >= 2, else None.
        """
        # Combine history with current analysis
        all_recordings = list(analysis_history) + [current_analysis]
        total_recordings = len(all_recordings)

        # Requirement 8.8: If fewer than 2 total recordings, omit trend data
        if total_recordings < MIN_RECORDINGS_FOR_TREND:
            return None

        # Sort by recorded_at ascending (chronological order)
        all_recordings.sort(key=lambda r: r["recorded_at"])

        # Limit to most recent 30 recordings
        recent_recordings = all_recordings[-MAX_TREND_RECORDINGS:]

        # Group by metric_name
        metrics_history: dict[str, list[MetricDataPoint]] = {}

        for recording in recent_recordings:
            analysis_id = recording["analysis_id"]
            recorded_at = recording["recorded_at"]
            metrics = recording.get("metrics", {})

            for metric_name, metric_data in metrics.items():
                data_point = MetricDataPoint(
                    analysis_id=analysis_id,
                    recorded_at=recorded_at,
                    value=metric_data["value"],
                    rating=self._parse_rating(metric_data["rating"]),
                )

                if metric_name not in metrics_history:
                    metrics_history[metric_name] = []
                metrics_history[metric_name].append(data_point)

        # Calculate date range from the included recordings
        date_range = (
            recent_recordings[0]["recorded_at"],
            recent_recordings[-1]["recorded_at"],
        )

        return TrendData(
            metrics_history=metrics_history,
            total_recordings=total_recordings,
            date_range=date_range,
        )

    def _parse_rating(self, rating) -> MetricRating:
        """Parse a rating value into MetricRating enum.

        Accepts MetricRating enum values or string representations.
        """
        if isinstance(rating, MetricRating):
            return rating
        if isinstance(rating, str):
            # Handle both "within_range" and "WITHIN_RANGE" formats
            return MetricRating(rating.lower())
        raise ValueError(f"Invalid rating value: {rating}")
