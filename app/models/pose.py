"""Pose estimation data models."""

from dataclasses import dataclass, field


@dataclass
class Keypoint:
    """A single body keypoint detected in a frame."""

    x: float  # normalized 0-1
    y: float  # normalized 0-1
    z: float  # depth estimate
    confidence: float  # 0-1
    name: str  # e.g., "left_hip", "right_wrist"


@dataclass
class PoseResult:
    """Pose estimation result for a single person in a single frame."""

    frame_index: int
    keypoints: list[Keypoint]  # 33 MediaPipe landmarks → 17+ essential keypoints
    person_id: int
    is_primary_batter: bool
    overall_confidence: float
    is_low_confidence: bool  # True when ≥40% of keypoints are occluded


@dataclass
class MultiPoseResult:
    """Pose estimation results for multiple persons in a single frame."""

    frame_index: int
    persons: list[PoseResult] = field(default_factory=list)
