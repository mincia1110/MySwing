"""Video file validation service.

Provides validation for uploaded video files including format, size,
resolution, frame rate, duration, and integrity checks.
"""

import logging
import mimetypes
import os
from pathlib import Path

import cv2

from app.core.constants import (
    IDEAL_DURATION_SEC,
    MAX_DURATION_SECONDS,
    MAX_FILE_SIZE_BYTES,
    MIN_FRAME_RATE,
    MIN_RESOLUTION_HEIGHT,
    MIN_RESOLUTION_WIDTH,
    RECOMMENDED_MAX_DURATION_SEC,
    RECOMMENDED_MIN_DURATION_SEC,
    SUPPORTED_FORMATS,
    SUPPORTED_MIME_TYPES,
)
from app.models.video import VideoInputPolicyResult, VideoMetadata, VideoValidationResult

logger = logging.getLogger(__name__)


def validate_format(file_path_or_key: str) -> bool:
    """Check if the file extension is a supported video format (mp4, mov, avi).

    Args:
        file_path_or_key: File path or S3 key to validate.

    Returns:
        True if the file extension is supported, False otherwise.
    """
    extension = Path(file_path_or_key).suffix.lower().lstrip(".")
    return extension in SUPPORTED_FORMATS


def validate_mime_type(file_path: str) -> bool:
    """Check if the file MIME type is a supported video type.

    Args:
        file_path: Path to the file to check.

    Returns:
        True if the MIME type is supported, False otherwise.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        return False
    return mime_type in SUPPORTED_MIME_TYPES


def validate_file_size(file_size_bytes: int) -> bool:
    """Check if the file size is within the allowed limit (≤500MB).

    Args:
        file_size_bytes: File size in bytes.

    Returns:
        True if the file size is within limits, False otherwise.
    """
    return file_size_bytes <= MAX_FILE_SIZE_BYTES


def validate_resolution(width: int, height: int) -> bool:
    """Check if the video resolution meets minimum requirements (≥1280×720).

    Args:
        width: Video width in pixels.
        height: Video height in pixels.

    Returns:
        True if resolution meets minimum requirements, False otherwise.
    """
    return width >= MIN_RESOLUTION_WIDTH and height >= MIN_RESOLUTION_HEIGHT


def validate_frame_rate(fps: float) -> bool:
    """Check if the frame rate meets minimum requirements (≥30fps).

    Args:
        fps: Frames per second.

    Returns:
        True if frame rate meets minimum requirements, False otherwise.
    """
    return fps >= MIN_FRAME_RATE


def validate_duration(duration_seconds: float) -> bool:
    """Check if duration passes the hard single-swing clip limit.

    MySwing is optimized for one swing per video. The recommended range is
    3-7 seconds, with a hard maximum of 10 seconds.
    """
    return 0 < duration_seconds <= MAX_DURATION_SECONDS


def _policy_base(duration_seconds: float | None) -> dict[str, float | None]:
    return {
        "duration_sec": duration_seconds,
        "ideal_duration_sec": IDEAL_DURATION_SEC,
        "recommended_min_duration_sec": RECOMMENDED_MIN_DURATION_SEC,
        "recommended_max_duration_sec": RECOMMENDED_MAX_DURATION_SEC,
        "max_duration_sec": MAX_DURATION_SECONDS,
    }


def validate_single_swing_input_policy(
    duration_seconds: float | None,
) -> VideoInputPolicyResult:
    """Return structured validation for MySwing's single-swing clip policy."""
    base = _policy_base(duration_seconds)
    recommendation = (
        "한 번의 스윙만 담긴 3~7초 클립으로 잘라 업로드하세요. "
        "약 5초 영상이 가장 적합합니다."
    )

    if duration_seconds is None or duration_seconds <= 0:
        return VideoInputPolicyResult(
            accepted=False,
            severity="error",
            reason="metadata_unavailable",
            message="영상 길이 메타데이터를 확인할 수 없어 분석을 시작할 수 없습니다.",
            recommendation=(
                "파일이 올바른 비디오인지 확인한 뒤, 한 번의 스윙만 담긴 짧은 클립으로 "
                "다시 업로드하세요."
            ),
            **base,
        )

    if duration_seconds > MAX_DURATION_SECONDS:
        return VideoInputPolicyResult(
            accepted=False,
            severity="error",
            reason="video_too_long",
            message=f"영상 길이가 {MAX_DURATION_SECONDS:g}초를 초과하여 분석할 수 없습니다.",
            recommendation=recommendation,
            **base,
        )

    if duration_seconds < RECOMMENDED_MIN_DURATION_SEC:
        return VideoInputPolicyResult(
            accepted=True,
            severity="warning",
            reason="video_too_short",
            message="영상이 권장 길이보다 짧아 일부 스윙 단계 분석이 불안정할 수 있습니다.",
            recommendation=recommendation,
            **base,
        )

    if duration_seconds > RECOMMENDED_MAX_DURATION_SEC:
        return VideoInputPolicyResult(
            accepted=True,
            severity="warning",
            reason="video_longer_than_recommended",
            message="영상이 권장 길이보다 길어 분석 정확도가 떨어질 수 있습니다.",
            recommendation=recommendation,
            **base,
        )

    return VideoInputPolicyResult(
        accepted=True,
        severity="ok",
        reason=None,
        message="단일 스윙 클립 길이가 권장 범위에 있습니다.",
        recommendation="현재 길이가 적합합니다. 약 5초 전후의 단일 스윙 클립이 가장 좋습니다.",
        **base,
    )


def extract_metadata(file_path: str) -> VideoMetadata:
    """Extract video metadata using OpenCV.

    Extracts duration, resolution, fps, and codec information from the video file.

    Args:
        file_path: Path to the video file.

    Returns:
        VideoMetadata with extracted information.

    Raises:
        ValueError: If the file cannot be opened or read by OpenCV.
    """
    cap = cv2.VideoCapture(file_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {file_path}")

        # Extract basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate duration
        if fps > 0:
            duration_seconds = frame_count / fps
        else:
            duration_seconds = 0.0

        # Extract codec (FourCC)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(
            [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
        ).strip()

        # Get file info
        file_name = os.path.basename(file_path)
        file_size_bytes = os.path.getsize(file_path)
        extension = Path(file_path).suffix.lower().lstrip(".")

        return VideoMetadata(
            file_key=file_path,
            file_name=file_name,
            file_size_bytes=file_size_bytes,
            duration_seconds=duration_seconds,
            resolution_width=width,
            resolution_height=height,
            frame_rate=fps,
            codec=codec,
            format=extension,
        )
    finally:
        cap.release()


def validate_integrity(file_path: str) -> bool:
    """Check video file integrity by attempting to decode the first few frames.

    Tries to read and decode frames to detect file corruption.

    Args:
        file_path: Path to the video file.

    Returns:
        True if the file can be decoded successfully, False otherwise.
    """
    cap = cv2.VideoCapture(file_path)
    try:
        if not cap.isOpened():
            return False

        # Try to decode the first few frames (up to 10)
        frames_to_check = min(10, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frames_to_check <= 0:
            return False

        for _ in range(frames_to_check):
            ret, frame = cap.read()
            if not ret or frame is None:
                return False

        return True
    except Exception:
        logger.exception("Error during video integrity check: %s", file_path)
        return False
    finally:
        cap.release()


def validate_video(file_path: str, file_size_bytes: int) -> VideoValidationResult:
    """Run all video validations and return a comprehensive result.

    Validates format, file size, resolution, frame rate, duration, and integrity.

    Args:
        file_path: Path to the video file.
        file_size_bytes: Size of the file in bytes.

    Returns:
        VideoValidationResult with all validation outcomes.
    """
    errors: list[str] = []

    # 1. Format validation
    format_ok = validate_format(file_path)
    if not format_ok:
        errors.append(
            "지원되지 않는 파일 형식입니다. 지원 형식: MP4, MOV, AVI"
        )

    # 2. MIME type validation
    mime_ok = validate_mime_type(file_path)
    if not mime_ok and format_ok:
        # Only add MIME error if format extension was ok but MIME doesn't match
        errors.append(
            "파일의 MIME 타입이 지원되지 않습니다."
        )
        format_ok = False

    # 3. File size validation
    size_ok = validate_file_size(file_size_bytes)
    if not size_ok:
        errors.append("최대 파일 크기(500MB)를 초과했습니다.")

    # 4. Extract metadata and validate video properties
    resolution_ok = False
    frame_rate_ok = False
    duration_ok = True

    if format_ok:
        try:
            metadata = extract_metadata(file_path)

            # Resolution validation
            resolution_ok = validate_resolution(
                metadata.resolution_width, metadata.resolution_height
            )
            if not resolution_ok:
                errors.append(
                    "최소 해상도(1280×720) 미만입니다."
                )

            # Frame rate validation
            frame_rate_ok = validate_frame_rate(metadata.frame_rate)
            if not frame_rate_ok:
                errors.append(
                    "최소 프레임레이트(30fps) 미만입니다."
                )

            # Duration validation
            duration_ok = validate_duration(metadata.duration_seconds)
            if not duration_ok:
                errors.append(
                    "최대 영상 길이(10초)를 초과했습니다. "
                    "한 번의 스윙만 담긴 짧은 클립을 업로드하세요."
                )

            # Integrity validation
            integrity_ok = validate_integrity(file_path)
            if not integrity_ok:
                errors.append("파일을 읽을 수 없습니다. 손상된 파일입니다.")
                format_ok = False

        except ValueError as e:
            errors.append(f"파일을 읽을 수 없습니다: {e}")
            format_ok = False

    is_valid = (
        format_ok
        and size_ok
        and resolution_ok
        and frame_rate_ok
        and duration_ok
    )

    return VideoValidationResult(
        is_valid=is_valid,
        format_ok=format_ok,
        size_ok=size_ok,
        resolution_ok=resolution_ok,
        frame_rate_ok=frame_rate_ok,
        errors=errors,
    )
