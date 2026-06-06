"""Shared trend-building service used by both /users/{id}/trends and the
report endpoint (Requirements 8.7, 8.8).

Extracts trend-data assembly into a single place so the analysis report
and the dedicated trends endpoint stay consistent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AnalysisResultTable, AnalysisTable
from app.pipeline.report_generator import (
    MAX_TREND_RECORDINGS,
    MIN_RECORDINGS_FOR_TREND,
)
from app.schemas.history import MetricTrendPoint, TrendResponse


async def build_user_trends(db: AsyncSession, user_id: UUID) -> TrendResponse:
    """Build a TrendResponse for the user's most recent completed analyses.

    Returns a TrendResponse with `metrics_history` populated when the user has
    at least MIN_RECORDINGS_FOR_TREND completed analyses, otherwise returns
    an empty history with an explanatory `message` (Req 8.8). Limits the
    history to the most recent MAX_TREND_RECORDINGS in chronological order.
    """
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

    if total_recordings < MIN_RECORDINGS_FOR_TREND:
        return TrendResponse(
            metrics_history={},
            total_recordings=total_recordings,
            message=(
                f"트렌드 분석을 위해 최소 {MIN_RECORDINGS_FOR_TREND}회의 "
                f"완료된 분석이 필요합니다. 현재 {total_recordings}회 완료."
            ),
        )

    # Chronological order (oldest first)
    rows = list(reversed(rows))

    metrics_history: dict[str, list[MetricTrendPoint]] = {}
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None

    for row in rows:
        analysis = row[0]
        analysis_result = row[1]

        recorded_at = analysis.completed_at or analysis.created_at

        if date_range_start is None or recorded_at < date_range_start:
            date_range_start = recorded_at
        if date_range_end is None or recorded_at > date_range_end:
            date_range_end = recorded_at

        evaluations_data = analysis_result.evaluations_data
        if not isinstance(evaluations_data, list):
            continue

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
            metrics_history.setdefault(metric_name, []).append(point)

    return TrendResponse(
        metrics_history=metrics_history,
        total_recordings=total_recordings,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
    )
