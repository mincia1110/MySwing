"""Video-related data models."""

from dataclasses import dataclass, field

from app.models.enums import QualityStatus


@dataclass
class VideoMetadata:
    """Metadata extracted from an uploaded video file."""

    file_key: str
    file_name: str
    file_size_bytes: int
    duration_seconds: float
    resolution_width: int
    resolution_height: int
    frame_rate: float
    codec: str
    format: str  # mp4, mov, avi


@dataclass
class VideoValidationResult:
    """Result of video file validation checks."""

    is_valid: bool
    format_ok: bool
    size_ok: bool
    resolution_ok: bool
    frame_rate_ok: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class VideoInputPolicyResult:
    """Structured result for the short single-swing input policy."""

    accepted: bool
    severity: str
    reason: str | None
    duration_sec: float | None
    ideal_duration_sec: float
    recommended_min_duration_sec: float
    recommended_max_duration_sec: float
    max_duration_sec: float
    message: str
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        """Return an API-serializable dictionary."""
        return {
            "accepted": self.accepted,
            "severity": self.severity,
            "reason": self.reason,
            "duration_sec": self.duration_sec,
            "ideal_duration_sec": self.ideal_duration_sec,
            "recommended_min_duration_sec": self.recommended_min_duration_sec,
            "recommended_max_duration_sec": self.recommended_max_duration_sec,
            "max_duration_sec": self.max_duration_sec,
            "message": self.message,
            "recommendation": self.recommendation,
        }


@dataclass
class QualityCheckResult:
    """Result of video quality checks (brightness, framing, resolution, frame rate stability)."""

    brightness_status: QualityStatus
    framing_status: QualityStatus
    resolution_status: QualityStatus
    frame_rate_stability_status: QualityStatus
    brightness_value: float  # lux equivalent
    swing_arc_visibility_percent: float
    frame_rate_variation_percent: float
    warnings: list[str] = field(default_factory=list)
