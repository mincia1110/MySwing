"""API-level regression tests for report abstention localization."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.analyses import get_analysis_report


def _scalar_result(value):
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locale", "expected_reasons"),
    [
        (
            "ko",
            [
                "사용 가능한 포즈 데이터가 없습니다.",
                "검출기로 관측한 배트 근거나 명시적인 임팩트 구간을 사용할 수 없습니다.",
            ],
        ),
        (
            "en",
            [
                "No pose data available",
                "No detector-observed bat evidence or explicit impact phase was available",
            ],
        ),
    ],
)
@patch("app.api.analyses.build_user_trends", new_callable=AsyncMock)
async def test_report_api_localizes_confirmed_abstentions_without_changing_ids(
    mock_build_user_trends: AsyncMock,
    locale: str,
    expected_reasons: list[str],
) -> None:
    mock_build_user_trends.side_effect = RuntimeError("trend data unavailable")
    analysis_id = uuid.uuid4()
    user_id = uuid.uuid4()
    analysis = MagicMock(
        id=analysis_id,
        user_id=user_id,
        video_id=uuid.uuid4(),
        status="completed",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    analysis_result = MagicMock(
        biomechanics_data={
            "bat_speed": None,
            "unmeasurable_metrics": [
                {"metric_name": "all", "reason": "No pose data available"},
                {
                    "metric_name": "impact_anchor",
                    "reason": (
                        "No detector-observed bat evidence or explicit impact phase "
                        "was available"
                    ),
                },
            ],
        },
        evaluations_data=[],
        improvements_data=[],
        drill_recommendations=[],
        swing_phases_data=[],
        overlay_video_key=None,
    )
    db = AsyncMock()
    db.execute.side_effect = [
        _scalar_result(analysis),
        _scalar_result(analysis_result),
        _scalar_result(None),
    ]

    report = await get_analysis_report(
        analysis_id=analysis_id,
        locale=locale,
        current_user_id=user_id,
        db=db,
    )

    assert report.biomechanics is not None
    assert [item.metric_name for item in report.biomechanics.unmeasurable_metrics] == [
        "all",
        "impact_anchor",
    ]
    assert [item.reason for item in report.biomechanics.unmeasurable_metrics] == expected_reasons
