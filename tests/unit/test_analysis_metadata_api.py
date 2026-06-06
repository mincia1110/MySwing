"""Unit tests for analysis metadata API extraction."""

from datetime import datetime, timezone
from types import SimpleNamespace

from app.api.analyses import _extract_analysis_metadata
from app.schemas.analysis import AnalysisReportResponse


def test_extract_analysis_metadata_from_biomechanics_jsonb():
    metadata = {
        "video_normalization": {
            "normalization_applied": True,
            "normalization_target_fps": 30.0,
            "normalization_crop_box": [656, 0, 608, 1080],
            "original_fps": 59.95,
            "analysis_fps": 30.0,
        },
        "analysis_coordinate_system": "canonical_rhb",
        "canonical_batting_direction": "right",
    }
    analysis_result = SimpleNamespace(
        biomechanics_data={"bat_speed": None, "analysis_metadata": metadata}
    )

    assert _extract_analysis_metadata(analysis_result) == metadata


def test_extract_analysis_metadata_handles_legacy_results_without_metadata():
    analysis_result = SimpleNamespace(biomechanics_data={"bat_speed": None})

    assert _extract_analysis_metadata(analysis_result) == {}


def test_analysis_report_response_defaults_analysis_metadata_for_legacy_payloads():
    response = AnalysisReportResponse(
        analysis_id="analysis-1",
        user_id="user-1",
        created_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
        status="completed",
        video_metadata={},
        quality_check={},
        swing_phases=[],
        biomechanics=None,
        metric_evaluations=[],
        improvements=[],
        drill_recommendations=[],
    )

    assert response.analysis_metadata == {}
