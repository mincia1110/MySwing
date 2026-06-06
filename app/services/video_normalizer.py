"""Frame normalization utilities for analysis-time video processing.

The analysis pipeline works on decoded OpenCV frames, not on the uploaded
container directly. Normalize those frames before pose/bat analysis so codec,
rotation metadata, frame rate, and pillarbox/letterbox differences have less
influence on downstream metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


DEFAULT_ANALYSIS_FPS = 30.0


@dataclass(frozen=True)
class FrameNormalizationResult:
    frames: list[np.ndarray]
    fps: float
    video_width: int
    video_height: int
    original_fps: float
    original_width: int
    original_height: int
    sampled_frame_indices: list[int]
    crop_box: tuple[int, int, int, int] | None
    target_fps: float


def normalize_frames_for_analysis(
    frames: Sequence[np.ndarray],
    source_fps: float,
    target_fps: float = DEFAULT_ANALYSIS_FPS,
) -> FrameNormalizationResult:
    """Normalize decoded frames for stable downstream analysis.

    Steps:
    1. Downsample high-FPS videos to target_fps using timestamp-equivalent
       frame indices. This makes 60fps originals comparable to 30fps mirrors.
    2. Crop common black pillarbox/letterbox borders from the sampled frames.
       This is important for mirrored exports stored as landscape frames with
       a portrait video centered inside black bars.
    3. Convert frames to contiguous 8-bit BGR arrays for OpenCV/MediaPipe.
    """
    if not frames:
        return FrameNormalizationResult(
            frames=[],
            fps=target_fps,
            video_width=0,
            video_height=0,
            original_fps=source_fps,
            original_width=0,
            original_height=0,
            sampled_frame_indices=[],
            crop_box=None,
            target_fps=target_fps,
        )

    first_height, first_width = frames[0].shape[:2]
    sampled_indices = _sample_frame_indices(
        frame_count=len(frames),
        source_fps=source_fps,
        target_fps=target_fps,
    )
    sampled_frames = [frames[i] for i in sampled_indices]
    crop_box = detect_active_content_box(sampled_frames)

    normalized_frames: list[np.ndarray] = []
    for frame in sampled_frames:
        normalized = _normalize_frame_array(frame)
        if crop_box is not None:
            x, y, width, height = crop_box
            normalized = normalized[y : y + height, x : x + width]
        normalized_frames.append(np.ascontiguousarray(normalized))

    if normalized_frames:
        video_height, video_width = normalized_frames[0].shape[:2]
    else:
        video_width = 0
        video_height = 0

    analysis_fps = target_fps if source_fps >= target_fps - 0.5 else max(source_fps, 1.0)

    return FrameNormalizationResult(
        frames=normalized_frames,
        fps=analysis_fps,
        video_width=video_width,
        video_height=video_height,
        original_fps=source_fps,
        original_width=first_width,
        original_height=first_height,
        sampled_frame_indices=sampled_indices,
        crop_box=crop_box,
        target_fps=target_fps,
    )


def detect_active_content_box(
    frames: Sequence[np.ndarray],
    black_threshold: int = 12,
    min_content_fraction: float = 0.01,
    min_crop_fraction: float = 0.05,
) -> tuple[int, int, int, int] | None:
    """Detect a common non-black content box across sampled frames.

    Returns None when the detected crop would be negligible. Coordinates are
    ``(x, y, width, height)`` in the input frame coordinate system.
    """
    if not frames:
        return None

    height, width = frames[0].shape[:2]
    row_has_content = np.zeros(height, dtype=bool)
    col_has_content = np.zeros(width, dtype=bool)

    # Use a bounded sample to keep preprocessing predictable.
    step = max(len(frames) // 12, 1)
    for frame in frames[::step]:
        normalized = _normalize_frame_array(frame)
        if normalized.shape[:2] != (height, width):
            continue
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        non_black = gray > black_threshold
        row_has_content |= non_black.mean(axis=1) > min_content_fraction
        col_has_content |= non_black.mean(axis=0) > min_content_fraction

    if not row_has_content.any() or not col_has_content.any():
        return None

    y_indices = np.flatnonzero(row_has_content)
    x_indices = np.flatnonzero(col_has_content)
    y0, y1 = int(y_indices[0]), int(y_indices[-1]) + 1
    x0, x1 = int(x_indices[0]), int(x_indices[-1]) + 1

    crop_width = x1 - x0
    crop_height = y1 - y0
    removed_width_fraction = 1.0 - (crop_width / width)
    removed_height_fraction = 1.0 - (crop_height / height)
    if (
        removed_width_fraction < min_crop_fraction
        and removed_height_fraction < min_crop_fraction
    ):
        return None

    return x0, y0, crop_width, crop_height


def _sample_frame_indices(
    frame_count: int,
    source_fps: float,
    target_fps: float,
) -> list[int]:
    if frame_count <= 0:
        return []
    if source_fps <= 0 or source_fps <= target_fps + 0.5:
        return list(range(frame_count))

    duration = frame_count / source_fps
    target_count = max(1, int(round(duration * target_fps)))
    indices: list[int] = []
    previous = -1
    for i in range(target_count):
        idx = min(frame_count - 1, int(round(i * source_fps / target_fps)))
        if idx != previous:
            indices.append(idx)
            previous = idx
    if indices[-1] != frame_count - 1 and (frame_count - 1) - indices[-1] < source_fps / target_fps:
        indices[-1] = frame_count - 1
    return indices


def _normalize_frame_array(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame
