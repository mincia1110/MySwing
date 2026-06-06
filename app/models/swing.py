"""Swing phase classification data models."""

from dataclasses import dataclass, field
from typing import Optional

from app.models.enums import SwingPhase


@dataclass
class TransitionBoundary:
    """Boundary between two consecutive swing phases."""

    from_phase: SwingPhase
    to_phase: SwingPhase
    frame_index: int
    confidence: float


@dataclass
class PhaseAnomaly:
    """Anomaly detected in a swing phase (missing or abnormally short)."""

    phase: SwingPhase
    anomaly_type: str  # "missing" | "abnormally_short"
    duration_ms: Optional[float] = None


@dataclass
class SwingPhaseResult:
    """Complete swing phase classification result."""

    phases: dict[SwingPhase, tuple[int, int]] = field(
        default_factory=dict
    )  # phase → (start_frame, end_frame)
    transitions: list[TransitionBoundary] = field(default_factory=list)
    phase_durations_ms: dict[SwingPhase, float] = field(default_factory=dict)
    anomalies: list[PhaseAnomaly] = field(default_factory=list)
    classification_failures: list[str] = field(default_factory=list)
