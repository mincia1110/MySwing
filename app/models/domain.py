"""Domain data models - consolidated re-exports for convenient access.

This module re-exports all core domain dataclasses and enums from their
respective modules, providing a single import point for the analysis pipeline.
"""

# Enums
from app.models.enums import (
    BattingDirection,
    MetricRating,
    QualityStatus,
    SwingPhase,
)

# Video models
from app.models.video import (
    QualityCheckResult,
    VideoMetadata,
    VideoValidationResult,
)

# Pose estimation models
from app.models.pose import (
    Keypoint,
    MultiPoseResult,
    PoseResult,
)

# Bat detection models
from app.models.bat import (
    BatDetectionResult,
    BatTrajectory,
)

# Swing phase classification models
from app.models.swing import (
    PhaseAnomaly,
    SwingPhaseResult,
    TransitionBoundary,
)

# Biomechanics analysis models
from app.models.biomechanics import (
    AttackAngleResult,
    BatSpeedResult,
    BiomechanicsResult,
    JointAngularVelocity,
    KinematicChainResult,
    LaunchAngleResult,
    RotationResult,
    UnmeasurableMetric,
)

# Evaluation models
from app.models.evaluation import (
    DrillRecommendation,
    ImprovementArea,
    MetricEvaluation,
    WeightTransferResult,
)

# Report models
from app.models.report import (
    AnalysisReport,
    MetricDataPoint,
    TrendData,
)

# User profile model
from app.models.user_profile import UserProfile

__all__ = [
    # Enums
    "BattingDirection",
    "MetricRating",
    "QualityStatus",
    "SwingPhase",
    # Video
    "VideoMetadata",
    "VideoValidationResult",
    "QualityCheckResult",
    # Pose
    "Keypoint",
    "PoseResult",
    "MultiPoseResult",
    # Bat
    "BatDetectionResult",
    "BatTrajectory",
    # Swing
    "TransitionBoundary",
    "SwingPhaseResult",
    "PhaseAnomaly",
    # Biomechanics
    "BatSpeedResult",
    "LaunchAngleResult",
    "JointAngularVelocity",
    "KinematicChainResult",
    "RotationResult",
    "AttackAngleResult",
    "BiomechanicsResult",
    "UnmeasurableMetric",
    # Evaluation
    "MetricEvaluation",
    "WeightTransferResult",
    "ImprovementArea",
    "DrillRecommendation",
    # Report
    "AnalysisReport",
    "TrendData",
    "MetricDataPoint",
    # User Profile
    "UserProfile",
]
