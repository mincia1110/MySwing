"""Tests for API-facing localization helpers."""

from app.schemas.analysis import DrillRecommendationResponse, UnmeasurableMetricResponse
from app.services.localization import (
    localize_drill_recommendations,
    localize_unmeasurable_metrics,
    normalize_locale,
)


def test_normalize_locale_defaults_to_korean() -> None:
    assert normalize_locale(None) == "ko"
    assert normalize_locale("ko") == "ko"
    assert normalize_locale("fr") == "ko"


def test_normalize_locale_accepts_english_prefix() -> None:
    assert normalize_locale("en") == "en"
    assert normalize_locale("en-US") == "en"


def test_localize_drill_recommendations_keeps_korean_by_default() -> None:
    drill = DrillRecommendationResponse(
        drill_name="로우 티 드릴",
        target_metric="attack_angle",
        description="한국어 설명",
        direction="below",
    )

    assert localize_drill_recommendations([drill], "ko") == [drill]


def test_localize_drill_recommendations_translates_known_drill_to_english() -> None:
    drill = DrillRecommendationResponse(
        drill_name="로우 티 드릴",
        target_metric="attack_angle",
        description="한국어 설명",
        direction="below",
    )

    [localized] = localize_drill_recommendations([drill], "en")

    assert localized.drill_name == "Low Tee Drill"
    assert localized.target_metric == "attack_angle"
    assert localized.direction == "below"
    assert "positive upward swing path" in localized.description


def test_localize_drill_recommendations_translates_generic_drill_to_english() -> None:
    drill = DrillRecommendationResponse(
        drill_name="맞춤 훈련 설계 필요",
        target_metric="bat_speed",
        description="한국어 설명",
        direction="above",
    )

    [localized] = localize_drill_recommendations([drill], "en")

    assert localized.drill_name == "Custom Training Plan Needed"
    assert "Bat Speed is above the reference range" in localized.description


def test_localize_unmeasurable_metrics_translates_known_reason_to_korean() -> None:
    metric = UnmeasurableMetricResponse(
        metric_name="bat_speed",
        reason="Bat barrel was not observed near impact",
    )

    [localized] = localize_unmeasurable_metrics([metric], "ko")

    assert localized.metric_name == "bat_speed"
    assert localized.reason == "임팩트 부근에서 배트 배럴이 관측되지 않았습니다."


def test_localize_unmeasurable_metrics_preserves_unknown_reason() -> None:
    metric = UnmeasurableMetricResponse(
        metric_name="custom_metric",
        reason="Provider-specific diagnostic",
    )

    assert localize_unmeasurable_metrics([metric], "ko") == [metric]
    assert localize_unmeasurable_metrics([metric], "en") == [metric]
