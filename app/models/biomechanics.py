"""Biomechanics analysis data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BatSpeedResult:
    """Bat speed measurement at the impact zone (Requirement 6.3)."""

    speed_kmh: float
    precision: float  # ±1 km/h
    measurement_frame: int


@dataclass
class ImpactAngleResult:
    """Impact attack-angle measurement at impact (2-frame window, Requirement 6.4)."""

    angle_degrees: float
    precision: float  # ±0.5 degrees
    impact_frame: int


# Backward-compat alias: historical name used across tests/API layers.
LaunchAngleResult = ImpactAngleResult


@dataclass
class JointAngularVelocity:
    """Peak angular velocity measurement for a single joint."""

    joint_name: str
    peak_velocity_dps: float  # degrees per second
    peak_frame: int
    peak_time_ms: float  # time relative to swing start


@dataclass
class KinematicChainResult:
    """Kinematic chain analysis result (Requirement 6.5)."""

    joint_peak_angular_velocities: dict[str, JointAngularVelocity] = field(
        default_factory=dict
    )  # keys: "hips", "shoulders", "elbows", "wrists"
    sequence_correct: bool = False  # proximal-to-distal order compliance
    timing_gaps_ms: dict[str, float] = field(
        default_factory=dict
    )  # timing difference between joints


@dataclass
class RotationResult:
    """Hip and shoulder rotation analysis result (Requirement 6.6)."""

    hip_rotation_speed_dps: float  # degrees per second
    shoulder_rotation_speed_dps: float  # degrees per second
    hip_shoulder_separation_degrees: float
    rotation_phase_start_frame: int
    rotation_phase_end_frame: int


@dataclass
class AttackAngleResult:
    """Attack angle measurement through the hitting zone (Requirement 6.8)."""

    angle_degrees: float
    precision: float  # ±0.5 degrees
    hitting_zone_start_frame: int
    hitting_zone_end_frame: int


@dataclass
class UnmeasurableMetric:
    """A metric that could not be computed, with reason (Requirement 6.9)."""

    metric_name: str
    reason: str


@dataclass
class BiomechanicsResult:
    """Complete biomechanics analysis result."""

    bat_speed: Optional[BatSpeedResult] = None
    attack_angle: Optional[ImpactAngleResult] = None  # 2-frame window
    kinematic_chain: Optional[KinematicChainResult] = None
    rotation: Optional[RotationResult] = None
    hand_path_efficiency: Optional[float] = None  # 0.0 - 1.0
    # Swing quality metrics (optional — depend on phase classification)
    stride_length_cm: Optional[float] = None
    cog_sway_cm: Optional[float] = None
    cog_drop_cm: Optional[float] = None
    head_stability_cm: Optional[float] = None
    front_knee_flexion_degrees: Optional[float] = None
    spine_angle_degrees: Optional[float] = None
    unmeasurable_metrics: list[UnmeasurableMetric] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    timeout_occurred: bool = False
