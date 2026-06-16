"""Analysis job API endpoints (Requirements 1.1, 8.1).

Provides endpoints for:
- Creating analysis jobs (POST /analyses)
- Querying analysis status (GET /analyses/{id}/status)
- Retrieving analysis reports (GET /analyses/{id}/report)
- Getting overlay video URLs (GET /analyses/{id}/overlay)
- Getting metrics data (GET /analyses/{id}/metrics)
"""

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id
from app.db.models import (
    AnalysisResultTable,
    AnalysisTable,
    QualityCheckTable,
    UserProfileTable,
    VideoTable,
)
from app.db.session import get_async_db
from app.schemas.analysis import (
    AnalysisCreateRequest,
    AnalysisCreateResponse,
    AnalysisReportResponse,
    AnalysisStatusResponse,
    DrillRecommendationResponse,
    ImprovementAreaResponse,
    MetricEvaluationResponse,
    SwingPhaseResponse,
    TrendDataResponse,
)
from app.services.localization import (
    localize_drill_recommendations,
    normalize_locale,
)
from app.services.s3_client import get_s3_client
from app.services.trend_service import build_user_trends
from app.services.video_validator import validate_single_swing_input_policy
from app.tasks.pipeline import analyze_swing_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyses", tags=["analyses"])


def _quality_check_to_response(quality_check: Any) -> dict:
    """Return API-facing quality check data from a QualityCheckTable row."""
    if quality_check is None:
        return {}
    details = quality_check.details or {}
    return {
        "brightness_status": quality_check.brightness_status,
        "framing_status": quality_check.framing_status,
        "resolution_status": quality_check.resolution_status,
        "frame_rate_stability_status": quality_check.frame_rate_stability_status,
        "brightness_value": details.get("brightness_value"),
        "swing_arc_visibility_percent": details.get("swing_arc_visibility_percent"),
        "frame_rate_variation_percent": details.get("frame_rate_variation_percent"),
        "warnings": details.get("warnings", []),
        "checked_at": quality_check.checked_at,
    }


def _extract_analysis_metadata(analysis_result: Any) -> dict:
    """Extract API-facing analysis metadata from stored JSONB result data."""
    biomechanics_data = analysis_result.biomechanics_data or {}
    if not isinstance(biomechanics_data, dict):
        return {}
    metadata = biomechanics_data.get("analysis_metadata", {})
    return metadata if isinstance(metadata, dict) else {}


@router.post(
    "",
    response_model=AnalysisCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create analysis job",
    description="Create a new swing analysis job. Enqueues the analysis pipeline "
    "and returns 202 Accepted with the analysis_id.",
    responses={
        202: {"description": "Analysis job created and enqueued"},
        400: {"description": "Invalid request - missing user profile or video not found"},
        404: {"description": "Video or user profile not found"},
    },
)
async def create_analysis(
    request: AnalysisCreateRequest,
    current_user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_async_db),
) -> dict:
    """Create a new analysis job.

    Validates that the video file exists and the user has a profile,
    creates an analysis record with 'pending' status, and enqueues
    the Celery analysis pipeline task.

    Returns 202 Accepted with the analysis_id.
    """
    user_id = request.user_id or current_user_id

    # Verify user profile exists (Requirement 2.1 - profile required before analysis)
    profile_result = await db.execute(
        select(UserProfileTable).where(UserProfileTable.user_id == user_id)
    )
    profile = profile_result.scalar_one_or_none()
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User profile is required before starting analysis. "
            "Please create a profile with height, bat_length, and batting_direction.",
        )

    # Verify the video exists and belongs to the requesting user. The file_key is
    # client-provided, so it must never be sufficient by itself.
    video_result = await db.execute(
        select(VideoTable).where(
            VideoTable.file_key == request.file_key, VideoTable.user_id == user_id
        )
    )
    video = video_result.scalar_one_or_none()
    if video is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Video not found for file_key: {request.file_key}. Metadata may be unavailable; "
                "upload a valid short single-swing video and fetch metadata before "
                "starting analysis."
            ),
        )

    input_validation = validate_single_swing_input_policy(video.duration_seconds)
    if not input_validation.accepted:
        detail_status = input_validation.reason or "metadata_unavailable"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": detail_status,
                "message": input_validation.message,
                "recommendation": input_validation.recommendation,
                "input_validation": input_validation.to_dict(),
            },
        )

    # Create analysis record
    analysis = AnalysisTable(
        user_id=user_id,
        video_id=video.id,
        status="pending",
    )
    db.add(analysis)
    await db.flush()

    # Enqueue Celery task
    analyze_swing_task.delay(str(analysis.id))

    logger.info(
        "Analysis job created: analysis_id=%s, user_id=%s, video_id=%s",
        analysis.id,
        user_id,
        video.id,
    )

    return {
        "analysis_id": str(analysis.id),
        "status": "pending",
        "message": "Analysis job created and enqueued for processing.",
        "input_validation": input_validation.to_dict(),
    }


@router.get(
    "/{analysis_id}/status",
    response_model=AnalysisStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get analysis status",
    description="Query the current status of an analysis job. "
    "Status transitions: pending → preprocessing → analyzing → "
    "evaluating → generating_report → completed/failed.",
)
async def get_analysis_status(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_async_db),
) -> AnalysisStatusResponse:
    """Get the current status of an analysis job.

    Returns the analysis status along with timestamps for tracking progress.
    """
    result = await db.execute(
        select(AnalysisTable).where(AnalysisTable.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis not found: {analysis_id}",
        )

    return AnalysisStatusResponse(
        analysis_id=str(analysis.id),
        status=analysis.status,
        error_message=analysis.error_message,
        started_at=analysis.started_at,
        completed_at=analysis.completed_at,
        created_at=analysis.created_at,
    )


@router.get(
    "/{analysis_id}/report",
    response_model=AnalysisReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Get analysis report",
    description="Retrieve the complete analysis report including biomechanics, "
    "evaluations, improvements, and drill recommendations.",
    responses={
        200: {"description": "Analysis report retrieved successfully"},
        404: {"description": "Analysis not found"},
        409: {"description": "Analysis not yet completed"},
    },
)
async def get_analysis_report(
    analysis_id: UUID,
    locale: str = Query(default="ko", description="Response language: ko or en"),
    db: AsyncSession = Depends(get_async_db),
) -> AnalysisReportResponse:
    """Retrieve the complete analysis report.

    Returns the full report only when the analysis status is 'completed'.
    Returns 409 if the analysis is still in progress or has failed.
    """
    result = await db.execute(
        select(AnalysisTable).where(AnalysisTable.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis not found: {analysis_id}",
        )

    if analysis.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Analysis is not yet completed. Current status: {analysis.status}",
        )

    # Get analysis result
    result_query = await db.execute(
        select(AnalysisResultTable).where(
            AnalysisResultTable.analysis_id == analysis_id
        )
    )
    analysis_result = result_query.scalar_one_or_none()

    if analysis_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis result data not found.",
        )

    # Get video metadata
    video_query = await db.execute(
        select(VideoTable).where(VideoTable.id == analysis.video_id)
    )
    video = video_query.scalar_one_or_none()
    video_metadata = {}
    quality_check = {}
    if video:
        quality_query = await db.execute(
            select(QualityCheckTable)
            .where(QualityCheckTable.video_id == video.id)
            .order_by(desc(QualityCheckTable.checked_at))
        )
        quality_check = _quality_check_to_response(
            quality_query.scalars().first()
        )

        video_metadata = {
            "file_key": video.file_key,
            "file_name": video.file_name,
            "file_size_bytes": video.file_size_bytes,
            "duration_seconds": video.duration_seconds,
            "resolution_width": video.resolution_width,
            "resolution_height": video.resolution_height,
            "frame_rate": video.frame_rate,
            "format": video.format,
        }

    # Build overlay video URL if available
    overlay_video_url = None
    if analysis_result.overlay_video_key:
        try:
            s3_client = get_s3_client()
            overlay_video_url = s3_client.generate_presigned_download_url(
                analysis_result.overlay_video_key, expires_in=3600
            )
        except Exception as e:
            logger.warning("Failed to generate overlay URL: %s", e)

    # Build metric evaluations from JSONB data
    metric_evaluations = []
    evaluations_data = analysis_result.evaluations_data or {}
    if isinstance(evaluations_data, list):
        for eval_item in evaluations_data:
            metric_evaluations.append(MetricEvaluationResponse(**eval_item))
    elif isinstance(evaluations_data, dict) and "evaluations" in evaluations_data:
        for eval_item in evaluations_data["evaluations"]:
            metric_evaluations.append(MetricEvaluationResponse(**eval_item))

    # Build improvements list (validate through Pydantic model)
    improvements: list[ImprovementAreaResponse] = []
    improvements_data = analysis_result.improvements_data or {}
    raw_improvements = []
    if isinstance(improvements_data, list):
        raw_improvements = improvements_data
    elif isinstance(improvements_data, dict) and "improvements" in improvements_data:
        raw_improvements = improvements_data["improvements"]
    for imp_item in raw_improvements:
        try:
            # Handle serialized format: target_range is [min, max] list
            item = dict(imp_item)
            if "target_range" in item and isinstance(item["target_range"], (list, tuple)):
                item["target_range_min"] = item["target_range"][0]
                item["target_range_max"] = item["target_range"][1]
                del item["target_range"]
            improvements.append(ImprovementAreaResponse(**item))
        except Exception:
            logger.warning("Skipping invalid improvement item: %s", imp_item)

    # Build drill recommendations (validate through Pydantic model)
    drill_recommendations: list[DrillRecommendationResponse] = []
    drill_data = analysis_result.drill_recommendations or {}
    raw_drills = []
    if isinstance(drill_data, list):
        raw_drills = drill_data
    elif isinstance(drill_data, dict) and "recommendations" in drill_data:
        raw_drills = drill_data["recommendations"]
    for drill_item in raw_drills:
        try:
            drill_recommendations.append(DrillRecommendationResponse(**drill_item))
        except Exception:
            logger.warning("Skipping invalid drill item: %s", drill_item)
    drill_recommendations = localize_drill_recommendations(
        drill_recommendations,
        normalize_locale(locale),
    )

    # Build swing phases (validate through Pydantic model)
    swing_phases: list[SwingPhaseResponse] = []
    phases_data = analysis_result.swing_phases_data or {}
    raw_phases_dict = {}
    phase_durations = {}

    if isinstance(phases_data, dict):
        raw_phases_dict = phases_data.get("phases", {})
        phase_durations = phases_data.get("phase_durations_ms", {})
    elif isinstance(phases_data, list):
        # Legacy format: list of phase dicts
        for phase_item in phases_data:
            try:
                swing_phases.append(SwingPhaseResponse(**phase_item))
            except Exception:
                logger.warning("Skipping invalid phase item: %s", phase_item)

    # Convert phases dict {"stance": [0, 34], ...} to SwingPhaseResponse list
    if raw_phases_dict and isinstance(raw_phases_dict, dict):
        fps = 30.0  # default fps for duration calculation
        for phase_name, frame_range in raw_phases_dict.items():
            if isinstance(frame_range, (list, tuple)) and len(frame_range) >= 2:
                start_frame = int(frame_range[0])
                end_frame = int(frame_range[1])
                # Get duration from phase_durations_ms if available
                duration_ms = phase_durations.get(
                    phase_name,
                    (end_frame - start_frame) / fps * 1000.0,
                )
                try:
                    swing_phases.append(SwingPhaseResponse(
                        phase=phase_name,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        duration_ms=float(duration_ms),
                    ))
                except Exception:
                    logger.warning("Skipping invalid phase: %s=%s", phase_name, frame_range)

    # Build biomechanics
    biomechanics = None
    bio_data = analysis_result.biomechanics_data or {}
    if bio_data and isinstance(bio_data, dict) and "bat_speed" in bio_data:
        biomechanics = bio_data

    # Build trend data for this user (Requirement 8.7)
    trend_data: TrendDataResponse | None = None
    try:
        trend_response = await build_user_trends(db, analysis.user_id)
        if trend_response.total_recordings >= 2:
            from app.schemas.analysis import MetricDataPointResponse
            trend_data = TrendDataResponse(
                metrics_history={
                    metric: [MetricDataPointResponse(**pt.model_dump()) for pt in points]
                    for metric, points in trend_response.metrics_history.items()
                },
                total_recordings=trend_response.total_recordings,
                date_range_start=trend_response.date_range_start,
                date_range_end=trend_response.date_range_end,
            )
    except Exception as e:
        logger.warning("Failed to build trend data: %s", e)

    return AnalysisReportResponse(
        analysis_id=str(analysis.id),
        user_id=str(analysis.user_id),
        created_at=analysis.created_at,
        status=analysis.status,
        video_metadata=video_metadata,
        quality_check=quality_check,
        analysis_metadata=_extract_analysis_metadata(analysis_result),
        swing_phases=swing_phases,
        biomechanics=biomechanics,
        metric_evaluations=metric_evaluations,
        improvements=improvements,
        drill_recommendations=drill_recommendations,
        overlay_video_url=overlay_video_url,
        trend_data=trend_data,
    )


@router.get(
    "/{analysis_id}/overlay",
    status_code=status.HTTP_200_OK,
    summary="Get overlay video URL",
    description="Retrieve the presigned URL for the overlay video "
    "showing pose skeleton and bat trajectory on the original video.",
    responses={
        200: {"description": "Overlay video URL retrieved"},
        404: {"description": "Analysis or overlay not found"},
        409: {"description": "Analysis not yet completed"},
    },
)
async def get_analysis_overlay(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_async_db),
) -> dict:
    """Retrieve the overlay video URL.

    Returns a presigned download URL for the overlay video that shows
    the detected pose skeleton and bat trajectory on the original video.
    """
    result = await db.execute(
        select(AnalysisTable).where(AnalysisTable.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis not found: {analysis_id}",
        )

    if analysis.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Analysis is not yet completed. Current status: {analysis.status}",
        )

    # Get analysis result for overlay key
    result_query = await db.execute(
        select(AnalysisResultTable).where(
            AnalysisResultTable.analysis_id == analysis_id
        )
    )
    analysis_result = result_query.scalar_one_or_none()

    if analysis_result is None or not analysis_result.overlay_video_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Overlay video not available for this analysis.",
        )

    s3_client = get_s3_client()
    overlay_url = s3_client.generate_presigned_download_url(
        analysis_result.overlay_video_key, expires_in=3600
    )

    return {
        "analysis_id": str(analysis_id),
        "overlay_video_url": overlay_url,
        "expires_in": 3600,
    }


@router.get(
    "/{analysis_id}/metrics",
    status_code=status.HTTP_200_OK,
    summary="Get analysis metrics",
    description="Retrieve the metrics data including biomechanics measurements "
    "and metric evaluations with color-coded ratings.",
    responses={
        200: {"description": "Metrics data retrieved"},
        404: {"description": "Analysis not found"},
        409: {"description": "Analysis not yet completed"},
    },
)
async def get_analysis_metrics(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_async_db),
) -> dict:
    """Retrieve metrics data for an analysis.

    Returns biomechanics measurements and metric evaluations with
    color-coded ratings (green/yellow/red).
    """
    result = await db.execute(
        select(AnalysisTable).where(AnalysisTable.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis not found: {analysis_id}",
        )

    if analysis.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Analysis is not yet completed. Current status: {analysis.status}",
        )

    # Get analysis result
    result_query = await db.execute(
        select(AnalysisResultTable).where(
            AnalysisResultTable.analysis_id == analysis_id
        )
    )
    analysis_result = result_query.scalar_one_or_none()

    if analysis_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis result data not found.",
        )

    # Build evaluations list
    evaluations = []
    evaluations_data = analysis_result.evaluations_data or {}
    if isinstance(evaluations_data, list):
        evaluations = evaluations_data
    elif isinstance(evaluations_data, dict) and "evaluations" in evaluations_data:
        evaluations = evaluations_data["evaluations"]

    return {
        "analysis_id": str(analysis_id),
        "biomechanics": analysis_result.biomechanics_data or {},
        "evaluations": evaluations,
        "analysis_metadata": _extract_analysis_metadata(analysis_result),
        "processing_time_seconds": analysis_result.processing_time_seconds,
    }
