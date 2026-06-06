"""Pydantic schemas for analysis API endpoints."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class AnalysisCreateRequest(BaseModel):
    """Request schema for creating a new analysis job."""

    file_key: str = Field(..., description="S3 object key of the uploaded video")
    user_id: str = Field(..., description="User ID to associate with the analysis")


class AnalysisStatusResponse(BaseModel):
    """Response schema for analysis status polling."""

    analysis_id: str
    status: Literal[
        "pending",
        "preprocessing",
        "analyzing",
        "evaluating",
        "generating_report",
        "completed",
        "failed",
    ]
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime


class MetricEvaluationResponse(BaseModel):
    """Response schema for a single metric evaluation."""

    metric_name: str
    measured_value: float
    unit: str
    reference_min: float
    reference_max: float
    deviation_percent: float
    rating: Literal["below_range", "within_range", "above_range"]
    color_code: Literal["green", "yellow", "red"]


class ImprovementAreaResponse(BaseModel):
    """Response schema for an improvement area."""

    metric_name: str
    deviation_percent: float
    current_value: float
    target_range_min: float
    target_range_max: float
    rank: int = Field(..., ge=1, le=3)
    rating: Optional[str] = None  # "above_range" or "below_range"


class DrillRecommendationResponse(BaseModel):
    """Response schema for a drill recommendation."""

    drill_name: str
    target_metric: str
    description: str
    # 방향성 정보. "below" = 기준 미달, "above" = 기준 초과, "generic" = 전용 매핑 부재.
    direction: Literal["below", "above", "generic"] = "generic"


class MetricDataPointResponse(BaseModel):
    """Response schema for a single metric data point in trend data."""

    analysis_id: str
    recorded_at: datetime
    value: float
    rating: Literal["below_range", "within_range", "above_range"]


class TrendDataResponse(BaseModel):
    """Response schema for trend data (Requirement 8.7)."""

    metrics_history: dict[str, list[MetricDataPointResponse]] = Field(default_factory=dict)
    total_recordings: int
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    message: Optional[str] = None  # Message when insufficient recordings


class SwingPhaseResponse(BaseModel):
    """Response schema for swing phase data."""

    phase: str
    start_frame: int
    end_frame: int
    duration_ms: float


class BatSpeedResponse(BaseModel):
    """Response schema for bat speed measurement."""

    speed_kmh: float
    precision: float


class LaunchAngleResponse(BaseModel):
    """Response schema for launch angle measurement."""

    angle_degrees: float
    precision: float


class BiomechanicsResponse(BaseModel):
    """Response schema for biomechanics analysis results."""

    bat_speed: Optional[BatSpeedResponse] = None
    attack_angle: Optional[LaunchAngleResponse] = None
    hand_path_efficiency: Optional[float] = None
    # Swing quality metrics
    stride_length_cm: Optional[float] = None
    cog_sway_cm: Optional[float] = None
    cog_drop_cm: Optional[float] = None
    head_stability_cm: Optional[float] = None
    front_knee_flexion_degrees: Optional[float] = None
    spine_angle_degrees: Optional[float] = None
    processing_time_seconds: Optional[float] = None


class AnalysisReportResponse(BaseModel):
    """Response schema for the complete analysis report."""

    analysis_id: str
    user_id: str
    created_at: datetime
    status: str
    video_metadata: dict
    quality_check: dict
    analysis_metadata: dict = Field(default_factory=dict)
    swing_phases: list[SwingPhaseResponse] = Field(default_factory=list)
    biomechanics: Optional[BiomechanicsResponse] = None
    metric_evaluations: list[MetricEvaluationResponse] = Field(default_factory=list)
    improvements: list[ImprovementAreaResponse] = Field(default_factory=list)
    drill_recommendations: list[DrillRecommendationResponse] = Field(default_factory=list)
    overlay_video_url: Optional[str] = None
    trend_data: Optional[TrendDataResponse] = None
