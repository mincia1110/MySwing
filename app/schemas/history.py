"""Pydantic schemas for user history API endpoints (Requirements 8.7, 8.8).

Provides request/response schemas for:
- GET /api/v1/users/{id}/analyses - Analysis history list (paginated)
- GET /api/v1/users/{id}/trends - Trend data retrieval
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AnalysisHistoryItem(BaseModel):
    """Response schema for a single analysis in the history list."""

    analysis_id: str
    video_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    video_file_name: Optional[str] = None
    processing_time_seconds: Optional[float] = None


class AnalysisHistoryResponse(BaseModel):
    """Paginated response for analysis history list."""

    items: list[AnalysisHistoryItem] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20
    has_next: bool = False


class MetricTrendPoint(BaseModel):
    """A single data point in a metric's trend history."""

    analysis_id: str
    recorded_at: datetime
    value: float
    rating: str  # "below_range", "within_range", "above_range"


class TrendResponse(BaseModel):
    """Response schema for trend data (Requirements 8.7, 8.8).

    If the user has >= 2 recordings, metrics_history is populated.
    If the user has < 2 recordings, metrics_history is empty and message is set.
    """

    metrics_history: dict[str, list[MetricTrendPoint]] = Field(default_factory=dict)
    total_recordings: int = 0
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    message: Optional[str] = None
