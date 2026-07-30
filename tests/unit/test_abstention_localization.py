"""Regression tests for confirmed abstention-reason localization gaps."""

import pytest

from app.schemas.analysis import UnmeasurableMetricResponse
from app.services.localization import localize_unmeasurable_metrics


@pytest.mark.parametrize(
    ("metric_name", "reason", "expected"),
    [
        (
            "all",
            "No pose data available",
            "사용 가능한 포즈 데이터가 없습니다.",
        ),
        (
            "impact_anchor",
            "No detector-observed bat evidence or explicit impact phase was available",
            "검출기로 관측한 배트 근거나 명시적인 임팩트 구간을 사용할 수 없습니다.",
        ),
    ],
)
def test_confirmed_abstention_reasons_are_localized_exactly_in_korean(
    metric_name: str,
    reason: str,
    expected: str,
) -> None:
    metric = UnmeasurableMetricResponse(metric_name=metric_name, reason=reason)

    [localized] = localize_unmeasurable_metrics([metric], "ko")

    assert localized.metric_name == metric_name
    assert localized.reason == expected


@pytest.mark.parametrize(
    "reason",
    [
        "No pose data available",
        "No detector-observed bat evidence or explicit impact phase was available",
    ],
)
def test_confirmed_abstention_reasons_remain_english_for_english_locale(
    reason: str,
) -> None:
    metric = UnmeasurableMetricResponse(metric_name="all", reason=reason)

    assert localize_unmeasurable_metrics([metric], "en") == [metric]
