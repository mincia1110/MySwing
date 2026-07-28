"""Confidence-aware motion estimates derived from tracked body landmarks.

The helpers in this module intentionally estimate only a temporal swing anchor.
They do not turn wrist motion into a measured bat trajectory.  This distinction
lets body-relative metrics recover when the barrel detector is sparse while bat
speed and attack angle continue to require detector-observed bat evidence.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from app.models.pose import Keypoint, PoseResult

MIN_KEYPOINT_CONFIDENCE = 0.5
MIN_POSE_FRAMES = 10
MIN_SPEED_SAMPLES = 6
MIN_HAND_COVERAGE = 0.45
MIN_TORSO_COVERAGE = 0.40
MIN_PEAK_SPEED_PER_FRAME = 0.0025
MIN_ESTIMATE_CONFIDENCE = 0.55


@dataclass(frozen=True)
class PoseImpactEstimate:
    """A pose-motion estimate of the swing's contact-adjacent frame."""

    frame_index: int
    confidence: float
    method: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class _HandSample:
    frame_index: int
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class _SpeedSample:
    frame_index: int
    speed: float
    confidence: float


def _finite_positive(value: float) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError, OverflowError):
        return False


def _keypoint(
    pose: PoseResult,
    name: str,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> Keypoint | None:
    for keypoint in pose.keypoints:
        if (
            keypoint.name == name
            and keypoint.confidence >= min_confidence
            and math.isfinite(keypoint.x)
            and math.isfinite(keypoint.y)
        ):
            return keypoint
    return None


def _hand_sample(
    pose: PoseResult,
    batting_direction: str,
    x_scale: float,
) -> _HandSample | None:
    power_side = "left" if str(batting_direction).lower() == "left" else "right"
    other_side = "right" if power_side == "left" else "left"
    power = _keypoint(pose, f"{power_side}_wrist")
    other = _keypoint(pose, f"{other_side}_wrist")

    if power is None and other is None:
        return None
    shoulders = [
        keypoint
        for name in ("left_shoulder", "right_shoulder")
        if (keypoint := _keypoint(pose, name)) is not None
    ]
    hips = [
        keypoint
        for name in ("left_hip", "right_hip")
        if (keypoint := _keypoint(pose, name)) is not None
    ]
    if not shoulders or not hips:
        return None
    torso_x = (
        statistics.mean(keypoint.x for keypoint in shoulders)
        + statistics.mean(keypoint.x for keypoint in hips)
    ) / 2.0
    torso_y = (
        statistics.mean(keypoint.y for keypoint in shoulders)
        + statistics.mean(keypoint.y for keypoint in hips)
    ) / 2.0

    def relative_sample(x: float, y: float, confidence: float) -> _HandSample:
        return _HandSample(
            pose.frame_index,
            (x - torso_x) * x_scale,
            y - torso_y,
            confidence,
        )

    if power is None:
        assert other is not None
        return relative_sample(other.x, other.y, other.confidence)
    if other is None:
        return relative_sample(power.x, power.y, power.confidence)

    # Both hands normally move as one unit on the handle.  Confidence weighting
    # reduces one-sided occlusion jitter without assuming a fixed lead wrist.
    total_confidence = power.confidence + other.confidence
    power_weight = power.confidence / total_confidence
    other_weight = other.confidence / total_confidence
    return relative_sample(
        power.x * power_weight + other.x * other_weight,
        power.y * power_weight + other.y * other_weight,
        min(power.confidence, other.confidence),
    )


def _has_torso_support(pose: PoseResult) -> bool:
    shoulders = (
        _keypoint(pose, "left_shoulder"),
        _keypoint(pose, "right_shoulder"),
    )
    hips = (_keypoint(pose, "left_hip"), _keypoint(pose, "right_hip"))
    return any(shoulders) and any(hips)


def _median_smooth(samples: list[_SpeedSample]) -> list[_SpeedSample]:
    smoothed: list[_SpeedSample] = []
    for index, sample in enumerate(samples):
        window = samples[max(0, index - 1) : min(len(samples), index + 2)]
        smoothed.append(
            _SpeedSample(
                frame_index=sample.frame_index,
                speed=float(statistics.median(item.speed for item in window)),
                confidence=sample.confidence,
            )
        )
    return smoothed


def _active_segments(
    samples: list[_SpeedSample],
    threshold: float,
    max_frame_gap: int,
) -> list[list[int]]:
    active_indices = [
        index for index, sample in enumerate(samples) if sample.speed >= threshold
    ]
    if not active_indices:
        return []

    segments: list[list[int]] = [[active_indices[0]]]
    for index in active_indices[1:]:
        previous = segments[-1][-1]
        sample_gap = index - previous
        frame_gap = samples[index].frame_index - samples[previous].frame_index
        # Bridge up to two quiet samples inside one acceleration burst, but do
        # not bridge an actual landmark/time discontinuity.
        if sample_gap <= 3 and frame_gap <= max_frame_gap * 3:
            segments[-1].extend(range(previous + 1, index + 1))
        else:
            segments.append([index])
    return [segment for segment in segments if len(segment) >= 3]


def _segment_score(
    samples: list[_SpeedSample], segment: list[int], baseline: float
) -> float:
    excess_motion = sum(
        max(0.0, samples[index].speed - baseline) for index in segment
    )
    mean_confidence = statistics.mean(samples[index].confidence for index in segment)
    return excess_motion * math.sqrt(len(segment)) * mean_confidence


def _first_substantial_peak(
    samples: list[_SpeedSample], segment: list[int]
) -> int:
    segment_peak = max(samples[index].speed for index in segment)
    substantial = segment_peak * 0.82
    for position, index in enumerate(segment):
        value = samples[index].speed
        left = (
            samples[segment[position - 1]].speed
            if position > 0
            else float("-inf")
        )
        right = (
            samples[segment[position + 1]].speed
            if position + 1 < len(segment)
            else float("-inf")
        )
        if value >= substantial and value >= left and value >= right:
            return index
    return max(segment, key=lambda index: samples[index].speed)


def estimate_impact_from_pose_motion(
    pose_sequence: list[PoseResult],
    *,
    batting_direction: str = "right",
    video_width: float = 1.0,
    video_height: float = 1.0,
    fps: float = 30.0,
    earliest_frame: int | None = None,
) -> PoseImpactEstimate | None:
    """Estimate a contact-adjacent frame from a supported hand-speed burst.

    The estimate requires sustained, full-body-supported motion and exposes its
    confidence/provenance.  It is suitable as an anchor for projected body
    metrics only; it is not detector evidence for bat speed or attack angle.
    """

    if (
        len(pose_sequence) < MIN_POSE_FRAMES
        or not _finite_positive(video_width)
        or not _finite_positive(video_height)
        or not _finite_positive(fps)
    ):
        return None

    poses = sorted(pose_sequence, key=lambda pose: pose.frame_index)
    # Ignore duplicate frame entries deterministically.
    unique_poses = list({pose.frame_index: pose for pose in poses}.values())
    unique_poses.sort(key=lambda pose: pose.frame_index)
    if len(unique_poses) < MIN_POSE_FRAMES:
        return None

    x_scale = float(video_width) / float(video_height)
    hand_samples = [
        sample
        for pose in unique_poses
        if (sample := _hand_sample(pose, batting_direction, x_scale)) is not None
    ]
    hand_coverage = len(hand_samples) / len(unique_poses)
    torso_coverage = sum(_has_torso_support(pose) for pose in unique_poses) / len(
        unique_poses
    )
    if hand_coverage < MIN_HAND_COVERAGE or torso_coverage < MIN_TORSO_COVERAGE:
        return None

    max_frame_gap = max(2, int(round(float(fps) * 0.12)))
    speed_samples: list[_SpeedSample] = []
    for previous, current in zip(hand_samples, hand_samples[1:]):
        frame_gap = current.frame_index - previous.frame_index
        if frame_gap <= 0 or frame_gap > max_frame_gap:
            continue
        speed = math.hypot(current.x - previous.x, current.y - previous.y) / frame_gap
        if not math.isfinite(speed):
            continue
        speed_samples.append(
            _SpeedSample(
                frame_index=current.frame_index,
                speed=speed,
                confidence=min(previous.confidence, current.confidence),
            )
        )

    if earliest_frame is not None:
        speed_samples = [
            sample for sample in speed_samples if sample.frame_index >= earliest_frame
        ]
    if len(speed_samples) < MIN_SPEED_SAMPLES:
        return None

    smoothed = _median_smooth(speed_samples)
    speeds = [sample.speed for sample in smoothed]
    sorted_speeds = sorted(speeds)
    quiet_count = max(3, len(sorted_speeds) // 2)
    baseline = float(statistics.median(sorted_speeds[:quiet_count]))
    deviations = [abs(speed - baseline) for speed in speeds]
    mad = float(statistics.median(deviations))
    peak_speed = max(speeds)
    required_peak = max(
        MIN_PEAK_SPEED_PER_FRAME,
        baseline + max(0.0015, 3.0 * mad),
    )
    if peak_speed < required_peak:
        return None

    threshold = max(
        MIN_PEAK_SPEED_PER_FRAME * 0.65,
        baseline + 1.5 * mad,
        peak_speed * 0.22,
    )
    threshold = min(threshold, peak_speed * 0.72)
    segments = _active_segments(smoothed, threshold, max_frame_gap)
    if not segments:
        return None

    ranked_segments = sorted(
        (
            (_segment_score(smoothed, segment, baseline), segment)
            for segment in segments
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    strongest_score, strongest_segment = ranked_segments[0]
    if strongest_score <= 0.0:
        return None
    strongest_peak = max(smoothed[index].speed for index in strongest_segment)
    qualified_segments = [
        (score, segment)
        for score, segment in ranked_segments
        if score >= strongest_score * 0.50
        and max(smoothed[index].speed for index in segment)
        >= strongest_peak * 0.68
    ]
    # In a single-swing upload, a later high-energy burst is commonly the
    # follow-through/reset rather than a second contact. Prefer the earliest
    # substantial burst while rejecting weak setup motion by score and peak.
    selected_score, best_segment = min(
        qualified_segments,
        key=lambda item: smoothed[item[1][0]].frame_index,
    )
    peak_index = _first_substantial_peak(smoothed, best_segment)
    peak_sample = smoothed[peak_index]

    before = [
        smoothed[index].speed
        for index in best_segment
        if index < peak_index
    ][-3:]
    after = [
        smoothed[index].speed
        for index in best_segment
        if index > peak_index
    ][:3]
    post_speed = float(statistics.median(after)) if after else peak_sample.speed
    post_drop = max(
        0.0,
        min(1.0, (peak_sample.speed - post_speed) / max(peak_sample.speed, 1e-9)),
    )
    prominence = max(
        0.0,
        min(1.0, (peak_sample.speed - baseline) / max(peak_sample.speed, 1e-9)),
    )
    segment_support = min(1.0, len(best_segment) / max(float(fps) * 0.20, 4.0))
    temporal_support = 0.5 * min(1.0, len(before) / 2.0) + 0.5 * min(
        1.0, len(after) / 2.0
    )
    alternative_scores = [
        score for score, segment in ranked_segments if segment is not best_segment
    ]
    ambiguity_ratio = (
        min(1.0, max(alternative_scores) / selected_score)
        if alternative_scores and selected_score > 0.0
        else 0.0
    )
    mean_segment_confidence = float(
        statistics.mean(smoothed[index].confidence for index in best_segment)
    )

    confidence = (
        0.18
        + 0.22 * prominence
        + 0.16 * hand_coverage
        + 0.12 * torso_coverage
        + 0.12 * segment_support
        + 0.10 * temporal_support
        + 0.06 * post_drop
        + 0.04 * mean_segment_confidence
    )
    # Similar-strength bursts usually mean a repeated swing or an ambiguous clip.
    confidence -= 0.10 * ambiguity_ratio
    confidence = max(0.0, min(0.95, confidence))
    if confidence < MIN_ESTIMATE_CONFIDENCE:
        return None

    return PoseImpactEstimate(
        frame_index=peak_sample.frame_index,
        confidence=confidence,
        method="pose_hand_speed_peak",
        evidence={
            "hand_frame_coverage": round(hand_coverage, 4),
            "torso_frame_coverage": round(torso_coverage, 4),
            "speed_sample_count": len(smoothed),
            "motion_segment_count": len(segments),
            "substantial_motion_segment_count": len(qualified_segments),
            "segment_selection": "earliest_substantial_burst",
            "selected_segment_start_frame": smoothed[best_segment[0]].frame_index,
            "selected_segment_end_frame": smoothed[best_segment[-1]].frame_index,
            "peak_speed_frame_height_per_frame": round(peak_sample.speed, 6),
            "quiet_speed_baseline": round(baseline, 6),
            "peak_prominence": round(prominence, 4),
            "post_peak_drop": round(post_drop, 4),
            "competing_segment_ratio": round(ambiguity_ratio, 4),
            "supports_bat_metrics": False,
        },
    )
