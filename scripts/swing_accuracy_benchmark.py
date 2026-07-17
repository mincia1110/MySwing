"""Compare local swing-analysis pose configurations without external services.

The reported coverage and temporal-consistency measurements are diagnostic
proxies. They are not ground-truth pose or bat localization accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES_FPS = 30.0
PHASE_NAMES = (
    "stance",
    "load",
    "stride",
    "rotation",
    "impact",
    "follow_through",
)
TRANSITION_NAMES = (
    "stance_to_load",
    "load_to_stride",
    "stride_to_rotation",
    "rotation_to_impact",
    "impact_to_follow_through",
)
CRITICAL_JOINT_GROUPS = {
    "shoulders": ("left_shoulder", "right_shoulder"),
    "hips": ("left_hip", "right_hip"),
    "elbows": ("left_elbow", "right_elbow"),
    "wrists": ("left_wrist", "right_wrist"),
    "knees": ("left_knee", "right_knee"),
    "ankles": ("left_ankle", "right_ankle"),
}
SEGMENT_ENDPOINTS = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_shank": ("right_knee", "right_ankle"),
}
MIN_SEGMENT_SAMPLES = 3
MIN_OBSERVED_BAT_LINES_FOR_PHASE_FALLBACK = 3
PROXY_WARNINGS = (
    "Coverage, confidence, segment-length smoothness, and mirror-consistency "
    "checks are proxy diagnostics; they are not ground-truth keypoint or bat "
    "localization accuracy.",
    "No aggregate accuracy score is computed. Only an explicitly supplied "
    "impact-frame annotation is compared with ground truth.",
    "Speed-derived phase fallback is a conservative heuristic, not "
    "ground-truth phase or impact localization.",
)


@dataclass(frozen=True)
class CandidateConfig:
    """Pose-backend settings for one named candidate."""

    name: str
    backend: str = "mediapipe"
    min_confidence: float = 0.5
    static_image_mode: bool | None = None
    model_complexity: int | None = None
    rtmpose_mode: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "static_image_mode": self.static_image_mode,
            "model_complexity": self.model_complexity,
            "min_confidence": self.min_confidence,
            "rtmpose_mode": self.rtmpose_mode,
        }


CANDIDATE_CONFIGS = {
    "static_full": CandidateConfig(
        "static_full",
        static_image_mode=True,
        model_complexity=1,
    ),
    "static_heavy": CandidateConfig(
        "static_heavy",
        static_image_mode=True,
        model_complexity=2,
    ),
    "video_full": CandidateConfig(
        "video_full",
        static_image_mode=False,
        model_complexity=1,
    ),
    "video_heavy": CandidateConfig(
        "video_heavy",
        static_image_mode=False,
        model_complexity=2,
    ),
    "rtmpose_lightweight": CandidateConfig(
        "rtmpose_lightweight",
        backend="rtmpose",
        rtmpose_mode="lightweight",
    ),
    "rtmpose_balanced": CandidateConfig(
        "rtmpose_balanced",
        backend="rtmpose",
        rtmpose_mode="balanced",
    ),
    "rtmpose_performance": CandidateConfig(
        "rtmpose_performance",
        backend="rtmpose",
        rtmpose_mode="performance",
    ),
}


@dataclass
class BenchmarkInput:
    """Frames and provenance for one benchmark input."""

    mode: str
    path: Path
    frames: list[np.ndarray]
    width: int
    height: int
    fps: float
    sha256: str
    details: dict[str, Any]


@dataclass(frozen=True)
class PipelineComponents:
    """Lazily imported pipeline classes, injectable for service-free tests."""

    pose_estimator: Any
    pose_tracker: Any
    wrist_bat_estimator: Any
    pose_constrained_bat_tracker: Any
    swing_phase_classifier: Any
    rtmpose_estimator: Any | None = None


def find_frame_files(frames_dir: Path) -> list[Path]:
    """Return frame_*.npy files in deterministic filename order."""
    return sorted(frames_dir.glob("frame_*.npy"), key=lambda path: path.name)


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_frame_files(frame_files: Sequence[Path]) -> str:
    """Hash sorted frame filenames and complete file content deterministically."""
    digest = hashlib.sha256()
    for frame_path in sorted(frame_files, key=lambda path: path.name):
        encoded_name = frame_path.name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        size = frame_path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with frame_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _finite_capture_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _nonnegative_capture_int(value: Any) -> int | None:
    parsed = _finite_capture_float(value)
    if parsed is None or parsed < 0:
        return None
    return int(round(parsed))


def load_frames_input(
    frames_dir: Path,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float = DEFAULT_FRAMES_FPS,
) -> BenchmarkInput:
    """Load saved pipeline frames as read-only NumPy memory maps."""
    resolved = frames_dir.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"frames directory is not a directory: {resolved}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")

    frame_files = find_frame_files(resolved)
    if not frame_files:
        raise ValueError(f"no frame_*.npy files found in {resolved}")

    frames: list[np.ndarray] = []
    inferred_width: int | None = None
    inferred_height: int | None = None
    for frame_path in frame_files:
        frame = np.load(frame_path, mmap_mode="r", allow_pickle=False)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"{frame_path.name} must have shape (height, width, 3); "
                f"got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                f"{frame_path.name} must have uint8 dtype; got {frame.dtype}"
            )
        frame_height, frame_width = frame.shape[:2]
        if inferred_width is None:
            inferred_width = int(frame_width)
            inferred_height = int(frame_height)
        elif (frame_width, frame_height) != (inferred_width, inferred_height):
            raise ValueError(
                f"{frame_path.name} dimensions {frame_width}x{frame_height} do "
                f"not match {inferred_width}x{inferred_height}"
            )
        frames.append(frame)

    assert inferred_width is not None
    assert inferred_height is not None
    if width is not None and width != inferred_width:
        raise ValueError(
            f"--width {width} does not match frame width {inferred_width}"
        )
    if height is not None and height != inferred_height:
        raise ValueError(
            f"--height {height} does not match frame height {inferred_height}"
        )

    return BenchmarkInput(
        mode="frames_dir",
        path=resolved,
        frames=frames,
        width=inferred_width,
        height=inferred_height,
        fps=float(fps),
        sha256=sha256_frame_files(frame_files),
        details={
            "frame_file_pattern": "frame_*.npy",
            "first_frame_file": frame_files[0].name,
            "last_frame_file": frame_files[-1].name,
            "frame_files_read_only": True,
            "hash_scope": "sorted filenames and complete .npy file bytes",
        },
    )


def load_video_input(video_path: Path) -> BenchmarkInput:
    """Decode and normalize a video entirely in memory."""
    resolved = video_path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"video path is not a file: {resolved}")

    cv2 = importlib.import_module("cv2")
    capture = cv2.VideoCapture(str(resolved))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {resolved}")

    reported_fps = _finite_capture_float(capture.get(cv2.CAP_PROP_FPS))
    reported_width = _nonnegative_capture_int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    reported_height = _nonnegative_capture_int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    reported_frame_count = _nonnegative_capture_int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    frames: list[np.ndarray] = []
    try:
        while True:
            decoded, frame = capture.read()
            if not decoded:
                break
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"video contains no decodable frames: {resolved}")
    first_shape = frames[0].shape
    if any(frame.shape != first_shape for frame in frames):
        raise ValueError("decoded video frames do not have consistent dimensions")

    source_fps = (
        reported_fps
        if reported_fps is not None and reported_fps > 0
        else DEFAULT_FRAMES_FPS
    )
    from app.services.video_normalizer import normalize_frames_for_analysis

    normalization = normalize_frames_for_analysis(frames, source_fps=source_fps)
    if not normalization.frames:
        raise ValueError("video normalization produced no frames")

    return BenchmarkInput(
        mode="video",
        path=resolved,
        frames=normalization.frames,
        width=normalization.video_width,
        height=normalization.video_height,
        fps=float(normalization.fps),
        sha256=sha256_file(resolved),
        details={
            "hash_scope": "complete video file bytes",
            "opencv_reported_fps": reported_fps,
            "opencv_reported_width": reported_width,
            "opencv_reported_height": reported_height,
            "opencv_reported_frame_count": reported_frame_count,
            "decoded_frame_count": len(frames),
            "normalization": {
                "original_fps": normalization.original_fps,
                "original_width": normalization.original_width,
                "original_height": normalization.original_height,
                "target_fps": normalization.target_fps,
                "crop_box": normalization.crop_box,
                "sampled_frame_indices": normalization.sampled_frame_indices,
            },
        },
    )


def prepare_analysis_frames(
    benchmark_input: BenchmarkInput,
    batting_direction: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Match production handedness canonicalization for benchmark frames."""
    if batting_direction not in ("right", "left"):
        raise ValueError("batting_direction must be 'right' or 'left'")
    if batting_direction == "right":
        return benchmark_input.frames, {
            "horizontal_flip_applied": False,
            "analysis_coordinate_system": "original",
            "input_contract": "original_or_canonical_rhb",
        }
    if benchmark_input.mode == "video":
        return [
            np.ascontiguousarray(frame[:, ::-1])
            for frame in benchmark_input.frames
        ], {
            "horizontal_flip_applied": True,
            "analysis_coordinate_system": "canonical_rhb",
            "input_contract": "raw_video_canonicalized_like_production",
        }
    return benchmark_input.frames, {
        "horizontal_flip_applied": False,
        "analysis_coordinate_system": "canonical_rhb",
        "input_contract": "frames_dir_must_already_be_canonical_rhb",
    }


def select_candidate_configs(
    selected: Sequence[str] | None,
) -> list[CandidateConfig]:
    """Resolve repeatable CLI selections, preserving order and removing repeats."""
    names = list(selected) if selected else list(CANDIDATE_CONFIGS)
    resolved: list[CandidateConfig] = []
    seen: set[str] = set()
    for name in names:
        if name not in CANDIDATE_CONFIGS:
            choices = ", ".join(CANDIDATE_CONFIGS)
            raise ValueError(f"unknown candidate {name!r}; choose from {choices}")
        if name not in seen:
            resolved.append(CANDIDATE_CONFIGS[name])
            seen.add(name)
    return resolved


def select_conditioning_modes(selected: Sequence[str] | None) -> list[str]:
    """Resolve requested conditioning modes, defaulting to on and off."""
    modes = list(selected) if selected else ["on", "off"]
    resolved: list[str] = []
    for mode in modes:
        if mode not in ("on", "off"):
            raise ValueError("conditioning must be 'on' or 'off'")
        if mode not in resolved:
            resolved.append(mode)
    return resolved


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_path(
    output_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Reject output destinations in user-owned repository data directories."""
    root = repository_root.expanduser().resolve()
    lexical_output = output_path.expanduser().absolute()
    resolved_output = output_path.expanduser().resolve()
    for protected_name in ("outputs", "sample_videos"):
        protected = root / protected_name
        if _is_relative_to(lexical_output, protected.absolute()) or _is_relative_to(
            resolved_output, protected.resolve()
        ):
            raise ValueError(
                f"refusing to write benchmark output inside {protected}"
            )
    return output_path


def compute_pose_coverage(
    pose_sequence: Sequence[Any], frame_count: int
) -> dict[str, Any]:
    """Compute frame and critical-joint coverage with explicit denominators."""
    pose_by_frame = {int(pose.frame_index): pose for pose in pose_sequence}
    frames_with_pose = 0
    low_confidence_frames = 0
    total_keypoints = 0
    joint_counts = {group: 0 for group in CRITICAL_JOINT_GROUPS}

    for frame_index in range(frame_count):
        pose = pose_by_frame.get(frame_index)
        keypoints = list(getattr(pose, "keypoints", [])) if pose is not None else []
        names = {keypoint.name for keypoint in keypoints}
        if keypoints:
            frames_with_pose += 1
        if (
            pose is None
            or not keypoints
            or bool(getattr(pose, "is_low_confidence", False))
        ):
            low_confidence_frames += 1
        total_keypoints += len(keypoints)
        for group, expected_names in CRITICAL_JOINT_GROUPS.items():
            joint_counts[group] += sum(name in names for name in expected_names)

    frame_denominator = max(frame_count, 1)
    joint_denominator = max(frame_count * 2, 1)
    return {
        "pose_frame_coverage": frames_with_pose / frame_denominator,
        "pose_frames_present": frames_with_pose,
        "pose_frame_denominator": frame_count,
        "critical_joint_coverage": {
            group: count / joint_denominator for group, count in joint_counts.items()
        },
        "critical_joint_samples_present": joint_counts,
        "critical_joint_sample_denominator_per_group": frame_count * 2,
        "low_confidence_frame_fraction": (
            low_confidence_frames / frame_denominator
        ),
        "low_confidence_frames": low_confidence_frames,
        "mean_keypoint_count": total_keypoints / frame_denominator,
    }


def compute_segment_length_cv(
    pose_sequence: Sequence[Any],
    width: int,
    height: int,
    *,
    min_samples: int = MIN_SEGMENT_SAMPLES,
) -> dict[str, Any]:
    """Measure segment-length CV after normalized x/y are scaled to pixels."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")

    lengths: dict[str, list[float]] = {
        segment: [] for segment in SEGMENT_ENDPOINTS
    }
    for pose in pose_sequence:
        keypoints = {
            keypoint.name: keypoint
            for keypoint in getattr(pose, "keypoints", [])
        }
        for segment, (start_name, end_name) in SEGMENT_ENDPOINTS.items():
            start = keypoints.get(start_name)
            end = keypoints.get(end_name)
            if start is None or end is None:
                continue
            coordinates = (start.x, start.y, end.x, end.y)
            if not all(math.isfinite(float(value)) for value in coordinates):
                continue
            dx = (float(end.x) - float(start.x)) * width
            dy = (float(end.y) - float(start.y)) * height
            length = math.hypot(dx, dy)
            if length > 0 and math.isfinite(length):
                lengths[segment].append(length)

    segment_cv: dict[str, float | None] = {}
    sample_counts: dict[str, int] = {}
    for segment, values in lengths.items():
        sample_counts[segment] = len(values)
        if len(values) < min_samples:
            segment_cv[segment] = None
            continue
        mean_length = float(np.mean(values))
        if not math.isfinite(mean_length) or mean_length <= 0:
            segment_cv[segment] = None
            continue
        cv = float(np.std(values, ddof=0) / mean_length)
        segment_cv[segment] = cv if math.isfinite(cv) else None

    available = [value for value in segment_cv.values() if value is not None]
    return {
        "coordinate_space": "2d_pixel_scaled_from_normalized_xy",
        "minimum_samples_per_segment": min_samples,
        "segment_coefficient_of_variation": segment_cv,
        "segment_sample_count": sample_counts,
        "available_segment_count": len(available),
        "median_segment_length_coefficient_of_variation": (
            float(median(available)) if available else None
        ),
    }


def compute_bat_metrics(
    bat_trajectory: Any, frame_count: int, method: str
) -> dict[str, Any]:
    """Summarize observed and proxy bat coverage over input frames."""
    detections = list(getattr(bat_trajectory, "detections", []))
    valid_range = range(frame_count)
    observed_frames = {
        int(detection.frame_index)
        for detection in detections
        if int(detection.frame_index) in valid_range
        and bool(getattr(detection, "detected", False))
        and not bool(getattr(detection, "is_predicted", False))
    }
    predicted_frames = {
        int(detection.frame_index)
        for detection in detections
        if int(detection.frame_index) in valid_range
        and bool(getattr(detection, "detected", False))
        and bool(getattr(detection, "is_predicted", False))
    }
    trajectory_frames = observed_frames | predicted_frames
    denominator = max(frame_count, 1)
    return {
        "method": method,
        "observed_line_coverage": len(observed_frames) / denominator,
        "observed_line_frames": len(observed_frames),
        "predicted_fraction": len(predicted_frames) / denominator,
        "predicted_frames": len(predicted_frames),
        "trajectory_coverage": len(trajectory_frames) / denominator,
        "trajectory_frames": len(trajectory_frames),
        "frame_denominator": frame_count,
    }


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def serialize_phase_metrics(phase_result: Any) -> dict[str, Any]:
    """Serialize all expected phase ranges and transition slots."""
    phase_ranges: dict[str, list[int] | None] = {
        phase: None for phase in PHASE_NAMES
    }
    phase_durations: dict[str, float | None] = {
        phase: None for phase in PHASE_NAMES
    }
    for phase, frame_range in getattr(phase_result, "phases", {}).items():
        name = _enum_value(phase)
        if name in phase_ranges:
            phase_ranges[name] = [int(frame_range[0]), int(frame_range[1])]
    for phase, duration in getattr(
        phase_result, "phase_durations_ms", {}
    ).items():
        name = _enum_value(phase)
        if name in phase_durations:
            phase_durations[name] = float(duration)

    transition_frames: dict[str, int | None] = {
        name: None for name in TRANSITION_NAMES
    }
    transitions: list[dict[str, Any]] = []
    estimated_impact_frame: int | None = None
    estimated_impact_source: str | None = None
    for transition in getattr(phase_result, "transitions", []):
        from_phase = _enum_value(transition.from_phase)
        to_phase = _enum_value(transition.to_phase)
        transition_name = f"{from_phase}_to_{to_phase}"
        frame_index = int(transition.frame_index)
        if transition_name in transition_frames:
            transition_frames[transition_name] = frame_index
        transitions.append(
            {
                "from_phase": from_phase,
                "to_phase": to_phase,
                "frame_index": frame_index,
                "confidence": float(transition.confidence),
            }
        )
        if to_phase == "impact":
            estimated_impact_frame = frame_index
            estimated_impact_source = "rotation_to_impact_transition"

    if estimated_impact_frame is None and phase_ranges["impact"] is not None:
        estimated_impact_frame = phase_ranges["impact"][0]
        estimated_impact_source = "impact_phase_range_start"

    anomalies = [
        {
            "phase": _enum_value(anomaly.phase),
            "type": str(anomaly.anomaly_type),
            "duration_ms": (
                None
                if anomaly.duration_ms is None
                else float(anomaly.duration_ms)
            ),
        }
        for anomaly in getattr(phase_result, "anomalies", [])
    ]
    return {
        "phase_ranges": phase_ranges,
        "phase_durations_ms": phase_durations,
        "transition_frames": transition_frames,
        "transitions": transitions,
        "estimated_impact_frame": estimated_impact_frame,
        "estimated_impact_source": estimated_impact_source,
        "anomalies": anomalies,
        "classification_failures": list(
            getattr(phase_result, "classification_failures", [])
        ),
    }


def _bat_detection_point(
    detection: Any,
    video_width: float = 1.0,
    video_height: float = 1.0,
) -> tuple[float, float]:
    bat_head = getattr(detection, "bat_head_position", None)
    if isinstance(bat_head, (list, tuple)) and len(bat_head) >= 2:
        point = float(bat_head[0]), float(bat_head[1])
    else:
        position = getattr(detection, "position", (0.0, 0.0))
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            point = float(position[0]), float(position[1])
        else:
            point = (0.0, 0.0)
    if getattr(detection, "coordinate_space", "pixel") == "normalized":
        return point[0] * video_width, point[1] * video_height
    return point


def estimate_impact_frame_from_bat_speed(
    bat_trajectory: Any,
    video_width: float = 1.0,
    video_height: float = 1.0,
) -> tuple[int, float, str]:
    """Mirror the production smoothed-peak impact estimator locally."""
    detections = sorted(
        [
            detection
            for detection in getattr(bat_trajectory, "detections", [])
            if bool(getattr(detection, "detected", False))
            and not bool(getattr(detection, "is_predicted", False))
        ],
        key=lambda detection: int(detection.frame_index),
    )
    if len(detections) < 2:
        return 0, 0.0, "insufficient_detections"

    frames: list[int] = []
    raw_speeds: list[float] = []
    for previous, current in zip(detections, detections[1:]):
        previous_point = _bat_detection_point(
            previous,
            video_width,
            video_height,
        )
        current_point = _bat_detection_point(
            current,
            video_width,
            video_height,
        )
        frame_span = max(
            1,
            int(current.frame_index) - int(previous.frame_index),
        )
        raw_speeds.append(
            math.hypot(
                current_point[0] - previous_point[0],
                current_point[1] - previous_point[1],
            )
            / frame_span
        )
        frames.append(int(current.frame_index))

    if not raw_speeds:
        return 0, 0.0, "no_speed_samples"

    smoothed = [
        float(median(raw_speeds[max(0, index - 1) : index + 2]))
        for index in range(len(raw_speeds))
    ]
    peak_indices: list[int] = []
    for index, speed in enumerate(smoothed):
        left = smoothed[index - 1] if index > 0 else float("-inf")
        right = (
            smoothed[index + 1]
            if index + 1 < len(smoothed)
            else float("-inf")
        )
        if speed >= left and speed >= right and speed > 0:
            peak_indices.append(index)
    if not peak_indices:
        peak_indices = [max(range(len(smoothed)), key=smoothed.__getitem__)]

    candidate_peak_indices = peak_indices
    if len(smoothed) >= 40 and frames:
        start_frame = frames[0]
        end_frame = frames[-1]
        duration = end_frame - start_frame
        min_impact_frame = start_frame + int(duration * 0.55)
        max_impact_frame = start_frame + int(duration * 0.70)
        hitting_zone_peak_indices = [
            index
            for index in peak_indices
            if min_impact_frame <= frames[index] <= max_impact_frame
        ]
        if hitting_zone_peak_indices:
            candidate_peak_indices = hitting_zone_peak_indices

    candidate_peak_indices.sort(
        key=lambda index: smoothed[index], reverse=True
    )
    best_index = candidate_peak_indices[0]
    impact_frame = frames[best_index]
    best_speed = smoothed[best_index]
    second_speed = (
        smoothed[candidate_peak_indices[1]]
        if len(candidate_peak_indices) > 1
        else 0.0
    )
    dominance = (
        (best_speed - second_speed) / best_speed
        if best_speed > 1e-9
        else 0.0
    )
    sample_factor = min(1.0, len(smoothed) / 8.0)
    confidence = max(
        0.05,
        min(0.99, (0.4 + 0.6 * max(0.0, dominance)) * sample_factor),
    )
    return impact_frame, float(confidence), "smoothed_peak_speed"


def estimate_phases_from_speed(
    bat_trajectory: Any,
    pose_sequence: Sequence[Any],
    fps: float,
    video_width: float = 1.0,
    video_height: float = 1.0,
) -> tuple[dict[str, list[int]], dict[str, float]]:
    """Mirror the production conservative speed-based six-phase fallback."""
    del pose_sequence  # Kept in the signature to match the production helper.
    detections = list(getattr(bat_trajectory, "detections", []))
    if len(detections) < 10:
        return {}, {}

    peak_frame, _, _ = estimate_impact_frame_from_bat_speed(
        bat_trajectory,
        video_width,
        video_height,
    )
    if peak_frame <= 0:
        return {}, {}
    start_frame = int(detections[0].frame_index)
    end_frame = int(detections[-1].frame_index)
    total_duration = end_frame - start_frame
    if total_duration <= 0:
        return {}, {}

    stance_end = start_frame + int(total_duration * 0.15)
    load_end = start_frame + int(total_duration * 0.30)
    stride_end = start_frame + int(total_duration * 0.50)
    impact_start = max(start_frame, min(peak_frame, end_frame))
    impact_end = min(impact_start + 2, end_frame)
    rotation_end = max(start_frame, impact_start - 1)
    stride_end = min(stride_end, max(start_frame, rotation_end - 1))
    load_end = min(load_end, max(start_frame, stride_end - 1))
    stance_end = min(stance_end, max(start_frame, load_end - 1))

    load_start = min(stance_end + 1, load_end)
    stride_start = min(load_end + 1, stride_end)
    rotation_start = min(stride_end + 1, rotation_end)
    follow_start = min(impact_end + 1, end_frame)
    phases = {
        "stance": [start_frame, stance_end],
        "load": [load_start, load_end],
        "stride": [stride_start, stride_end],
        "rotation": [rotation_start, rotation_end],
        "impact": [impact_start, impact_end],
        "follow_through": [follow_start, end_frame],
    }
    durations = {
        phase: max(0.0, (end - start) / fps * 1000.0)
        for phase, (start, end) in phases.items()
    }
    return phases, durations


def _transitions_from_phase_ranges(
    phases: dict[str, list[int]],
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for from_phase, to_phase in zip(PHASE_NAMES, PHASE_NAMES[1:]):
        to_range = phases.get(to_phase)
        if not isinstance(to_range, (list, tuple)) or len(to_range) < 2:
            continue
        transitions.append(
            {
                "from_phase": from_phase,
                "to_phase": to_phase,
                "frame_index": int(to_range[0]),
                "confidence": 0.75,
            }
        )
    return transitions


def apply_conservative_phase_fallback(
    raw_classifier_metrics: dict[str, Any],
    bat_trajectory: Any,
    pose_sequence: Sequence[Any],
    fps: float,
    video_width: float = 1.0,
    video_height: float = 1.0,
) -> dict[str, Any]:
    """Use speed phases only for incomplete results backed by observed lines."""
    phase_ranges = raw_classifier_metrics.get("phase_ranges", {})
    missing_phases = [
        phase for phase in PHASE_NAMES if not phase_ranges.get(phase)
    ]
    impact_missing = (
        raw_classifier_metrics.get("estimated_impact_frame") is None
    )
    observed_line_count = sum(
        1
        for detection in getattr(bat_trajectory, "detections", [])
        if bool(getattr(detection, "detected", False))
        and not bool(getattr(detection, "is_predicted", False))
    )
    evidence = {
        "raw_classifier_impact_missing": impact_missing,
        "raw_classifier_missing_phases": missing_phases,
        "observed_non_predicted_bat_lines": observed_line_count,
        "minimum_observed_bat_lines_required": (
            MIN_OBSERVED_BAT_LINES_FOR_PHASE_FALLBACK
        ),
        "trajectory_detection_count": len(
            getattr(bat_trajectory, "detections", [])
        ),
    }
    final_metrics = dict(raw_classifier_metrics)
    final_metrics.update(
        {
            "result_source": "swing_phase_classifier",
            "fallback_applied": False,
            "fallback_reason": "raw_classifier_complete",
            "fallback_evidence": evidence,
        }
    )
    needs_fallback = impact_missing
    if not needs_fallback:
        return final_metrics
    if observed_line_count < MIN_OBSERVED_BAT_LINES_FOR_PHASE_FALLBACK:
        final_metrics["fallback_reason"] = (
            "insufficient_observed_non_predicted_bat_lines"
        )
        return final_metrics

    del pose_sequence, fps  # Contact-only fallback does not invent phase windows.
    impact_frame, impact_confidence, impact_method = (
        estimate_impact_frame_from_bat_speed(
            bat_trajectory,
            video_width,
            video_height,
        )
    )
    if impact_frame <= 0:
        final_metrics["fallback_reason"] = (
            "speed_fallback_insufficient_trajectory"
        )
        return final_metrics

    phase_ranges = dict(raw_classifier_metrics.get("phase_ranges", {}))
    phase_ranges["impact"] = [impact_frame, impact_frame]
    phase_durations = dict(
        raw_classifier_metrics.get("phase_durations_ms", {})
    )
    phase_durations["impact"] = 0.0
    result_source = (
        "mixed_pose_and_observed_bat_contact"
        if any(
            phase_ranges.get(phase)
            for phase in PHASE_NAMES
            if phase != "impact"
        )
        else "observed_bat_contact_only"
    )
    evidence["missing_phases_after_fallback"] = [
        phase for phase in PHASE_NAMES if not phase_ranges.get(phase)
    ]
    return {
        **raw_classifier_metrics,
        "phase_ranges": phase_ranges,
        "phase_durations_ms": phase_durations,
        "estimated_impact_frame": impact_frame,
        "estimated_impact_source": f"speed_fallback.{impact_method}",
        "estimated_impact_confidence": impact_confidence,
        "result_source": result_source,
        "fallback_applied": True,
        "fallback_reason": "raw_classifier_missing_impact",
        "fallback_evidence": evidence,
    }


def impact_absolute_error(
    estimated_frame: int | None, annotated_frame: int | None
) -> int | None:
    """Return absolute frame error only when both values are available."""
    if estimated_frame is None or annotated_frame is None:
        return None
    return abs(int(estimated_frame) - int(annotated_frame))


def _load_pipeline_components() -> PipelineComponents:
    """Import optional inference dependencies only when a benchmark runs."""
    from app.pipeline.pose_constrained_bat_tracker import (
        PoseConstrainedBatTracker,
    )
    from app.pipeline.pose_estimator import PoseEstimator
    from app.pipeline.pose_tracker import PoseTracker
    from app.pipeline.rtmpose_estimator import RTMPoseEstimator
    from app.pipeline.swing_classifier import SwingPhaseClassifier
    from app.pipeline.wrist_bat_estimator import WristBatEstimator

    return PipelineComponents(
        pose_estimator=PoseEstimator,
        pose_tracker=PoseTracker,
        wrist_bat_estimator=WristBatEstimator,
        pose_constrained_bat_tracker=PoseConstrainedBatTracker,
        swing_phase_classifier=SwingPhaseClassifier,
        rtmpose_estimator=RTMPoseEstimator,
    )


def _failure(stage: str, error: BaseException) -> dict[str, str]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error),
    }


def _elapsed(start: float, clock: Callable[[], float]) -> float:
    return max(0.0, float(clock() - start))


def _configuration_for_run(
    config: CandidateConfig, conditioning: str
) -> dict[str, Any]:
    payload = config.as_dict()
    payload["temporal_conditioning"] = conditioning
    return payload


def _failed_candidate(
    config: CandidateConfig,
    conditioning_modes: Sequence[str],
    failure: dict[str, str],
    processing_seconds: float,
) -> dict[str, Any]:
    return {
        "name": config.name,
        "status": "failed",
        "configuration": config.as_dict(),
        "processing_seconds": processing_seconds,
        "failure": failure,
        "runs": [
            {
                "conditioning": mode,
                "configuration": _configuration_for_run(config, mode),
                "status": "failed",
                "processing_seconds": 0.0,
                "failure": failure,
            }
            for mode in conditioning_modes
        ],
    }


def run_candidate(
    config: CandidateConfig | str,
    *,
    frames: list[np.ndarray],
    width: int,
    height: int,
    fps: float,
    batting_direction: str,
    conditioning_modes: Sequence[str],
    impact_frame: int | None = None,
    components: PipelineComponents | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one pose backend and isolate failures from other candidates."""
    if isinstance(config, str):
        config = select_candidate_configs([config])[0]
    candidate_start = clock()

    try:
        pipeline = components or _load_pipeline_components()
    except Exception as error:
        return _failed_candidate(
            config,
            conditioning_modes,
            _failure("pipeline_import", error),
            _elapsed(candidate_start, clock),
        )

    pose_start = clock()
    try:
        if config.backend == "mediapipe":
            estimator = pipeline.pose_estimator(
                min_confidence=config.min_confidence,
                static_image_mode=config.static_image_mode,
                model_complexity=config.model_complexity,
            )
        elif config.backend == "rtmpose":
            if pipeline.rtmpose_estimator is None:
                raise RuntimeError("RTMPose estimator is unavailable")
            estimator = pipeline.rtmpose_estimator(
                min_confidence=config.min_confidence,
                mode=config.rtmpose_mode,
                inference_backend="onnxruntime",
                device="cpu",
            )
        else:
            raise ValueError(f"unsupported pose backend: {config.backend!r}")
        try:
            available = getattr(estimator, "is_available", True)
            if callable(available):
                available = available()
            if not available:
                raise RuntimeError(
                    f"{config.backend} pose estimator is unavailable"
                )
            base_pose_sequence = [
                estimator.process_frame(frame, frame_index=frame_index)
                for frame_index, frame in enumerate(frames)
            ]
        finally:
            estimator.close()
    except Exception as error:
        return _failed_candidate(
            config,
            conditioning_modes,
            _failure("pose_estimation", error),
            _elapsed(candidate_start, clock),
        )
    pose_seconds = _elapsed(pose_start, clock)

    runs: list[dict[str, Any]] = []
    for conditioning in conditioning_modes:
        run_start = clock()
        stage = "pose_tracking"
        try:
            tracker = pipeline.pose_tracker(
                enable_smoothing=conditioning == "on"
            )
            tracked_pose_sequence = tracker.track_across_frames(
                base_pose_sequence
            )
            analysis_pose_sequence = [
                pose for pose in tracked_pose_sequence if pose.keypoints
            ]

            stage = "wrist_bat_estimation"
            wrist_estimator = pipeline.wrist_bat_estimator(
                dominant_hand=batting_direction
            )
            wrist_trajectory = wrist_estimator.estimate_trajectory(
                analysis_pose_sequence,
                video_width=width,
                video_height=height,
            )

            stage = "pose_constrained_bat_tracking"
            constrained_tracker = pipeline.pose_constrained_bat_tracker()
            constrained_trajectory = constrained_tracker.track(
                frames=frames,
                pose_sequence=analysis_pose_sequence,
                video_width=width,
                video_height=height,
                fps=fps,
                wrist_prior=wrist_trajectory,
            )
            has_observed_line = any(
                detection.detected and not detection.is_predicted
                for detection in constrained_trajectory.detections
            )
            if has_observed_line:
                bat_method = "pose_constrained"
                bat_trajectory = constrained_trajectory
            else:
                bat_method = "wrist_proxy"
                bat_trajectory = wrist_trajectory

            stage = "phase_classification"
            classifier = pipeline.swing_phase_classifier(
                batting_direction=batting_direction,
                video_width=width,
                video_height=height,
            )
            phase_result = classifier.classify_phases(
                analysis_pose_sequence,
                bat_trajectory,
                fps,
            )

            stage = "metric_calculation"
            pose_metrics = compute_pose_coverage(
                tracked_pose_sequence, len(frames)
            )
            pose_metrics["segment_length_stability"] = (
                compute_segment_length_cv(
                    tracked_pose_sequence,
                    width,
                    height,
                )
            )
            raw_phase_metrics = serialize_phase_metrics(phase_result)
            phase_metrics = apply_conservative_phase_fallback(
                raw_phase_metrics,
                bat_trajectory,
                analysis_pose_sequence,
                fps,
                width,
                height,
            )
            bat_metrics = compute_bat_metrics(
                bat_trajectory, len(frames), bat_method
            )
            run: dict[str, Any] = {
                "conditioning": conditioning,
                "configuration": _configuration_for_run(config, conditioning),
                "status": "completed",
                "metrics": {
                    "pose": pose_metrics,
                    "bat": bat_metrics,
                    "raw_classifier_phases": raw_phase_metrics,
                    "phases": phase_metrics,
                },
            }
            if impact_frame is not None:
                run["ground_truth_comparison"] = {
                    "annotated_impact_frame": impact_frame,
                    "estimated_impact_frame": phase_metrics[
                        "estimated_impact_frame"
                    ],
                    "estimated_impact_source": phase_metrics[
                        "estimated_impact_source"
                    ],
                    "phase_result_source": phase_metrics["result_source"],
                    "fallback_applied": phase_metrics["fallback_applied"],
                    "annotated_impact_absolute_error_frames": impact_absolute_error(
                        phase_metrics["estimated_impact_frame"], impact_frame
                    ),
                }
            run_seconds = _elapsed(run_start, clock)
            run["processing_seconds"] = run_seconds
            run["shared_pose_estimation_seconds"] = pose_seconds
            run["total_seconds_including_shared_pose"] = (
                pose_seconds + run_seconds
            )
            runs.append(run)
        except Exception as error:
            runs.append(
                {
                    "conditioning": conditioning,
                    "configuration": _configuration_for_run(
                        config, conditioning
                    ),
                    "status": "failed",
                    "processing_seconds": _elapsed(run_start, clock),
                    "shared_pose_estimation_seconds": pose_seconds,
                    "failure": _failure(stage, error),
                }
            )

    successful_runs = sum(run["status"] == "completed" for run in runs)
    if successful_runs == len(runs):
        status = "completed"
    elif successful_runs:
        status = "partial_failure"
    else:
        status = "failed"
    return {
        "name": config.name,
        "status": status,
        "configuration": config.as_dict(),
        "pose_estimation_seconds": pose_seconds,
        "processing_seconds": _elapsed(candidate_start, clock),
        "pose_frame_results": len(base_pose_sequence),
        "runs": runs,
    }


def _run_git(repository_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def repository_provenance(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Collect Git identity without requiring the directory to be a repository."""
    head = _run_git(repository_root, "rev-parse", "HEAD")
    branch = _run_git(repository_root, "branch", "--show-current")
    status = _run_git(repository_root, "status", "--porcelain")
    return {
        "root": str(repository_root.resolve()),
        "available": head is not None,
        "head": head,
        "branch": branch or None,
        "dirty": None if status is None else bool(status),
    }


def runtime_versions() -> dict[str, str | None]:
    """Collect versions while avoiding a MediaPipe module import."""
    try:
        cv2_version = str(importlib.import_module("cv2").__version__)
    except (ImportError, AttributeError, OSError):
        cv2_version = None
    try:
        mediapipe_version = metadata.version("mediapipe")
    except metadata.PackageNotFoundError:
        mediapipe_version = None
    try:
        rtmlib_version = metadata.version("rtmlib")
    except metadata.PackageNotFoundError:
        rtmlib_version = None
    try:
        onnxruntime_version = metadata.version("onnxruntime")
    except metadata.PackageNotFoundError:
        onnxruntime_version = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2_version,
        "mediapipe": mediapipe_version,
        "rtmlib": rtmlib_version,
        "onnxruntime": onnxruntime_version,
    }


def run_benchmark(
    benchmark_input: BenchmarkInput,
    *,
    candidates: Sequence[CandidateConfig],
    conditioning_modes: Sequence[str],
    batting_direction: str,
    impact_frame: int | None,
) -> dict[str, Any]:
    """Run all candidates and build one provenance-rich JSON object."""
    benchmark_start = time.perf_counter()
    analysis_frames, handedness_preprocessing = prepare_analysis_frames(
        benchmark_input,
        batting_direction,
    )
    candidate_results = [
        run_candidate(
            config,
            frames=analysis_frames,
            width=benchmark_input.width,
            height=benchmark_input.height,
            fps=benchmark_input.fps,
            batting_direction=batting_direction,
            conditioning_modes=conditioning_modes,
            impact_frame=impact_frame,
        )
        for config in candidates
    ]
    successful_pairs = sum(
        run["status"] == "completed"
        for candidate in candidate_results
        for run in candidate["runs"]
    )
    total_pairs = sum(len(candidate["runs"]) for candidate in candidate_results)
    if successful_pairs == total_pairs:
        status = "completed"
    elif successful_pairs:
        status = "partial_failure"
    else:
        status = "failed"

    annotations: dict[str, int] = {}
    if impact_frame is not None:
        annotations["impact_frame"] = impact_frame
    return {
        "schema_version": 1,
        "benchmark": "swing_accuracy_configuration_comparison",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "processing_seconds": _elapsed(benchmark_start, time.perf_counter),
        "provenance": {
            "repository": repository_provenance(),
            "runtime": runtime_versions(),
        },
        "input": {
            "mode": benchmark_input.mode,
            "path": str(benchmark_input.path),
            "sha256": benchmark_input.sha256,
            "details": benchmark_input.details,
        },
        "frames": {
            "width": benchmark_input.width,
            "height": benchmark_input.height,
            "count": len(benchmark_input.frames),
            "fps": benchmark_input.fps,
        },
        "selection": {
            "candidates": [config.name for config in candidates],
            "conditioning": list(conditioning_modes),
            "batting_direction": batting_direction,
            "handedness_preprocessing": handedness_preprocessing,
        },
        "annotations": annotations,
        "metric_interpretation": {
            "ground_truth_metrics": (
                ["annotated_impact_absolute_error_frames"]
                if impact_frame is not None
                else []
            ),
            "proxy_metrics": [
                "pose coverage",
                "critical-joint coverage",
                "low-confidence fraction",
                "mean keypoint count",
                "segment-length coefficient of variation",
                "bat trajectory coverage",
                "phase transitions",
                "speed-derived phase fallback",
            ],
            "warnings": list(PROXY_WARNINGS),
        },
        "successful_pair_count": successful_pairs,
        "requested_pair_count": total_pairs,
        "candidates": candidate_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="local video to decode")
    source.add_argument(
        "--frames-dir",
        type=Path,
        help=(
            "directory containing read-only frame_*.npy analysis arrays; "
            "left-handed inputs must already be canonical_rhb"
        ),
    )
    parser.add_argument(
        "--width",
        type=_positive_int,
        help="expected saved-frame width; inferred when omitted",
    )
    parser.add_argument(
        "--height",
        type=_positive_int,
        help="expected saved-frame height; inferred when omitted",
    )
    parser.add_argument(
        "--fps",
        type=_positive_float,
        help="saved-frame rate (default: 30); video metadata is used for videos",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        choices=tuple(CANDIDATE_CONFIGS),
        help="candidate to run; repeat to select a subset (default: all)",
    )
    parser.add_argument(
        "--conditioning",
        action="append",
        choices=("on", "off"),
        help="temporal conditioning mode; repeat as needed (default: both)",
    )
    parser.add_argument(
        "--batting-direction",
        choices=("right", "left"),
        required=True,
        help="real batter handedness passed to landmark consumers",
    )
    parser.add_argument(
        "--impact-frame",
        type=_nonnegative_int,
        help="optional annotated impact index in normalized analysis frames",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this caller-selected path instead of stdout",
    )
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output is not None:
            validate_output_path(args.output)
        candidates = select_candidate_configs(args.candidate)
        conditioning_modes = select_conditioning_modes(args.conditioning)
        if args.video is not None:
            if args.width is not None or args.height is not None or args.fps is not None:
                raise ValueError(
                    "--width, --height, and --fps are only valid with --frames-dir"
                )
            benchmark_input = load_video_input(args.video)
        else:
            benchmark_input = load_frames_input(
                args.frames_dir,
                width=args.width,
                height=args.height,
                fps=args.fps if args.fps is not None else DEFAULT_FRAMES_FPS,
            )
        if (
            args.impact_frame is not None
            and args.impact_frame >= len(benchmark_input.frames)
        ):
            raise ValueError(
                f"--impact-frame {args.impact_frame} is outside the normalized "
                f"frame range 0..{len(benchmark_input.frames) - 1}"
            )
        if (
            args.output is not None
            and args.output.expanduser().resolve() == benchmark_input.path
        ):
            raise ValueError("output path must not overwrite the benchmark input")
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    payload = run_benchmark(
        benchmark_input,
        candidates=candidates,
        conditioning_modes=conditioning_modes,
        batting_direction=args.batting_direction,
        impact_frame=args.impact_frame,
    )
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            args.output.write_text(rendered, encoding="utf-8")
    except OSError as error:
        print(f"ERROR: could not write benchmark JSON: {error}", file=sys.stderr)
        return 2
    return 0 if payload["successful_pair_count"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
