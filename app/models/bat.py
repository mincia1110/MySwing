"""Bat detection and tracking data models."""

from dataclasses import dataclass, field


@dataclass
class BatDetectionResult:
    """Bat detection result for a single frame (Requirement 4.1)."""

    frame_index: int
    detected: bool
    position: tuple[float, float]  # center coordinates in the declared coordinate_space
    orientation_angle: float  # 0-360 degrees
    length_pixels: float
    confidence: float
    is_predicted: bool  # True if generated from trajectory prediction
    coordinate_space: str = "pixel"  # "pixel" | "normalized"
    bat_head_position: tuple[float, float] | None = None  # optional barrel/head point


@dataclass
class BatTrajectory:
    """Bat trajectory tracking result across multiple frames."""

    detections: list[BatDetectionResult] = field(default_factory=list)
    bat_speed_pixels_per_frame: list[float] = field(default_factory=list)
    tracking_accuracy: float = 0.0  # 0-1
    tracking_failures: list[tuple[int, int]] = field(
        default_factory=list
    )  # (start_frame, end_frame)
