"""Pydantic schemas for video upload API endpoints."""

from typing import Optional

from pydantic import BaseModel, Field


class PresignedUrlRequest(BaseModel):
    """Request schema for generating a presigned upload URL."""

    file_name: str = Field(..., description="Original file name")
    content_type: str = Field(
        ...,
        pattern=r"^video/(mp4|quicktime|x-msvideo)$",
        description="MIME type of the video file",
    )


class PresignedUrlResponse(BaseModel):
    """Response schema for presigned upload URL."""

    upload_url: str = Field(..., description="Presigned S3 upload URL")
    file_key: str = Field(..., description="S3 object key for the uploaded file")
    expires_in: int = Field(..., description="URL expiration time in seconds")


class ResolutionResponse(BaseModel):
    """Resolution sub-schema with width and height."""

    width: int
    height: int


class VideoMetadataResponse(BaseModel):
    """Response schema for video metadata after upload."""

    file_key: str
    file_name: str
    file_size_bytes: int
    duration_seconds: float
    resolution: ResolutionResponse
    frame_rate: float
    codec: str
    format: str
    thumbnail_url: Optional[str] = None


class VideoMetadataWithThumbnailResponse(BaseModel):
    """Response schema for video metadata with thumbnail (Requirement 1.7).

    Returns file_name, duration_seconds, resolution, file_size_bytes, and thumbnail_url.
    """

    file_name: str
    duration_seconds: float
    resolution: ResolutionResponse
    file_size_bytes: int
    thumbnail_url: Optional[str] = None


class VideoValidationResponse(BaseModel):
    """Response schema for video validation result."""

    is_valid: bool
    format_ok: bool
    size_ok: bool
    resolution_ok: bool
    frame_rate_ok: bool
    errors: list[str] = Field(default_factory=list)


class QualityCheckResponse(BaseModel):
    """Response schema for video quality check result."""

    brightness_status: str  # "pass" | "warning"
    framing_status: str  # "pass" | "warning"
    resolution_status: str  # "pass" | "warning"
    frame_rate_stability_status: str  # "pass" | "warning"
    brightness_value: float
    swing_arc_visibility_percent: float
    frame_rate_variation_percent: float
    warnings: list[str] = Field(default_factory=list)
