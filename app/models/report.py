"""Analysis report data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from app.models.enums import MetricRating

if TYPE_CHECKING:
    from app.models.biomechanics import BiomechanicsResult
    from app.models.evaluation import DrillRecommendation, ImprovementArea, MetricEvaluation
    from app.models.swing import SwingPhaseResult
    from app.models.video import QualityCheckResult, VideoMetadata


@dataclass
class MetricDataPoint:
    """A single data point in a metric's trend history."""

    analysis_id: str
    recorded_at: datetime
    value: float
    rating: MetricRating


@dataclass
class TrendData:
    """Trend data across multiple analysis sessions (Requirement 8.7)."""

    metrics_history: dict[str, list[MetricDataPoint]] = field(default_factory=dict)
    total_recordings: int = 0
    date_range: tuple[datetime, datetime] = field(
        default_factory=lambda: (datetime.min, datetime.max)
    )


@dataclass
class AnalysisReport:
    """Complete analysis report combining all analysis results."""

    analysis_id: str
    user_id: str
    created_at: datetime
    video_metadata: VideoMetadata
    quality_check: QualityCheckResult
    swing_phases: SwingPhaseResult
    biomechanics: BiomechanicsResult
    metric_evaluations: list[MetricEvaluation] = field(default_factory=list)
    improvements: list[ImprovementArea] = field(default_factory=list)
    drill_recommendations: list[DrillRecommendation] = field(default_factory=list)
    overlay_video_url: str = ""
    comparison_view: Optional[dict] = None
    trend_data: Optional[TrendData] = None
