"""Swing evaluation data models."""

from dataclasses import dataclass

from app.models.enums import MetricRating


@dataclass
class MetricEvaluation:
    """Evaluation of a single metric against reference ranges."""

    metric_name: str
    measured_value: float
    unit: str
    reference_min: float
    reference_max: float
    deviation_percent: float  # deviation from range boundary as percentage
    rating: MetricRating
    color_code: str  # "green" | "yellow" | "red"


@dataclass
class WeightTransferResult:
    """Weight transfer timing analysis result (Requirement 7.4)."""

    stride_to_foot_plant_ms: float
    foot_plant_to_hip_rotation_ms: float


@dataclass
class ImprovementArea:
    """An identified area for improvement, ranked by deviation magnitude."""

    metric_name: str
    deviation_percent: float
    current_value: float
    target_range: tuple[float, float]
    rank: int  # 1-3
    rating: str = ""  # "above_range" or "below_range"


@dataclass
class DrillRecommendation:
    """A drill recommendation for addressing a weakness.

    `direction`:
        - "below": 사용 지표가 기준 미만이라 below 드릴이 추천됨
        - "above": 사용 지표가 기준 초과라 above 드릴이 추천됨
        - "generic": 이 metric/direction에 전용 매핑이 없어 일반 안내로 폴백됨
    """

    drill_name: str
    target_metric: str
    description: str
    direction: str = "generic"
