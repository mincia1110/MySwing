"""Pydantic schemas for API request/response models."""

from app.schemas.user_profile import (
    UserProfileCreate,
    UserProfileResponse,
    UserProfileUpdate,
)
from app.schemas.video import (
    PresignedUrlRequest,
    PresignedUrlResponse,
    QualityCheckResponse,
    ResolutionResponse,
    VideoMetadataResponse,
    VideoMetadataWithThumbnailResponse,
    VideoValidationResponse,
)
from app.schemas.analysis import (
    AnalysisCreateRequest,
    AnalysisStatusResponse,
    AnalysisReportResponse,
    MetricEvaluationResponse,
    ImprovementAreaResponse,
    DrillRecommendationResponse,
    TrendDataResponse,
)

__all__ = [
    # User Profile
    "UserProfileCreate",
    "UserProfileUpdate",
    "UserProfileResponse",
    # Video
    "PresignedUrlRequest",
    "PresignedUrlResponse",
    "ResolutionResponse",
    "VideoMetadataResponse",
    "VideoMetadataWithThumbnailResponse",
    "VideoValidationResponse",
    "QualityCheckResponse",
    # Analysis
    "AnalysisCreateRequest",
    "AnalysisStatusResponse",
    "AnalysisReportResponse",
    "MetricEvaluationResponse",
    "ImprovementAreaResponse",
    "DrillRecommendationResponse",
    "TrendDataResponse",
]
