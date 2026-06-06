"""User history API endpoints (Requirements 8.7, 8.8).

Provides endpoints for:
- GET /api/v1/users/{id}/analyses - Analysis history list (paginated, most recent first)
- GET /api/v1/users/{id}/trends - Trend data retrieval
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AnalysisResultTable, AnalysisTable, VideoTable
from app.db.session import get_async_db
from app.pipeline.report_generator import MAX_TREND_RECORDINGS, MIN_RECORDINGS_FOR_TREND
from app.schemas.history import (
    AnalysisHistoryItem,
    AnalysisHistoryResponse,
    MetricTrendPoint,
    TrendResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["history"])


@router.get(
    "/{user_id}/analyses",
    response_model=AnalysisHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get user analysis history",
    description=(
        "Retrieve paginated analysis history for a user, "
        "ordered by most recent first."
    ),
)
async def get_user_analyses(
    user_id: UUID,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_async_db),
) -> AnalysisHistoryResponse:
    """Retrieve paginated analysis history for a user.

    Returns analyses ordered by creation date descending (most recent first).
    Includes video file name and processing time when available.
    """
    offset = (page - 1) * page_size

    # Count total analyses for this user
    count_stmt = select(func.count()).select_from(AnalysisTable).where(
        AnalysisTable.user_id == user_id
    )
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # Fetch paginated analyses with video info, most recent first
    stmt = (
        select(AnalysisTable, VideoTable.file_name, AnalysisResultTable.processing_time_seconds)
        .outerjoin(VideoTable, AnalysisTable.video_id == VideoTable.id)
        .outerjoin(AnalysisResultTable, AnalysisTable.id == AnalysisResultTable.analysis_id)
        .where(AnalysisTable.user_id == user_id)
        .order_by(AnalysisTable.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    rows = result.all()

    items = []
    for row in rows:
        analysis = row[0]
        video_file_name = row[1]
        processing_time = row[2]

        items.append(
            AnalysisHistoryItem(
                analysis_id=str(analysis.id),
                video_id=str(analysis.video_id),
                status=analysis.status,
                created_at=analysis.created_at,
                completed_at=analysis.completed_at,
                video_file_name=video_file_name,
                processing_time_seconds=processing_time,
            )
        )

    has_next = (offset + page_size) < total

    return AnalysisHistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=has_next,
    )


@router.get(
    "/{user_id}/trends",
    response_model=TrendResponse,
    status_code=status.HTTP_200_OK,
    summary="Get user trend data",
    description=(
        "Retrieve trend data showing metric changes over time. "
        "Requires at least 2 completed analyses. "
        "Returns at most the 30 most recent recordings in chronological order."
    ),
)
async def get_user_trends(
    user_id: UUID,
    db: AsyncSession = Depends(get_async_db),
) -> TrendResponse:
    """Retrieve trend data for a user's swing metrics over time.

    Requirements 8.7: If user has >= 2 completed analyses, display trend data
    showing metric changes over the most recent 30 recordings in chronological order.

    Requirements 8.8: If user has < 2 completed analyses, omit trend data and
    display a message indicating the minimum number of recordings required.
    """
    # Fetch completed analyses with results, ordered by creation date
    stmt = (
        select(AnalysisTable, AnalysisResultTable)
        .join(AnalysisResultTable, AnalysisTable.id == AnalysisResultTable.analysis_id)
        .where(
            AnalysisTable.user_id == user_id,
            AnalysisTable.status == "completed",
        )
        .order_by(AnalysisTable.created_at.desc())
        .limit(MAX_TREND_RECORDINGS)
    )
    result = await db.execute(stmt)
    rows = result.all()

    total_recordings = len(rows)

    # Requirement 8.8: If fewer than 2 recordings, return message
    if total_recordings < MIN_RECORDINGS_FOR_TREND:
        return TrendResponse(
            metrics_history={},
            total_recordings=total_recordings,
            message=(
                f"트렌드 분석을 위해 최소 {MIN_RECORDINGS_FOR_TREND}회의 "
                f"완료된 분석이 필요합니다. 현재 {total_recordings}회 완료."
            ),
        )

    # Reverse to chronological order (oldest first)
    rows = list(reversed(rows))

    # Build metrics history from evaluations_data JSONB
    metrics_history: dict[str, list[MetricTrendPoint]] = {}
    date_range_start = None
    date_range_end = None

    for row in rows:
        analysis = row[0]
        analysis_result = row[1]

        recorded_at = analysis.completed_at or analysis.created_at

        if date_range_start is None or recorded_at < date_range_start:
            date_range_start = recorded_at
        if date_range_end is None or recorded_at > date_range_end:
            date_range_end = recorded_at

        # Extract metrics from evaluations_data
        evaluations_data = analysis_result.evaluations_data
        if isinstance(evaluations_data, list):
            for evaluation in evaluations_data:
                metric_name = evaluation.get("metric_name", "")
                value = evaluation.get("measured_value")
                rating = evaluation.get("rating", "within_range")

                if not metric_name or value is None:
                    continue

                point = MetricTrendPoint(
                    analysis_id=str(analysis.id),
                    recorded_at=recorded_at,
                    value=float(value),
                    rating=rating,
                )

                if metric_name not in metrics_history:
                    metrics_history[metric_name] = []
                metrics_history[metric_name].append(point)

    return TrendResponse(
        metrics_history=metrics_history,
        total_recordings=total_recordings,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
    )
