"""Unit tests for TrendAnalyzer (Task 11.4).

Tests cover:
- 0 previous recordings → returns None
- 1 previous recording (total=2 with current) → returns TrendData
- 30+ recordings → limits to 30 most recent
- Chronological ordering
- metrics_history contains all measured metrics
- MetricDataPoint fields are populated correctly
"""

from datetime import datetime, timedelta

import pytest

from app.models.enums import MetricRating
from app.models.report import MetricDataPoint, TrendData
from app.pipeline.report_generator import TrendAnalyzer


@pytest.fixture
def analyzer() -> TrendAnalyzer:
    """Create a TrendAnalyzer instance."""
    return TrendAnalyzer()


def _make_analysis(
    analysis_id: str,
    recorded_at: datetime,
    metrics: dict[str, dict] | None = None,
) -> dict:
    """Helper to create an analysis dict with sensible defaults."""
    if metrics is None:
        metrics = {
            "bat_speed": {"value": 120.0, "rating": "within_range"},
            "attack_angle": {"value": 12.0, "rating": "within_range"},
        }
    return {
        "analysis_id": analysis_id,
        "recorded_at": recorded_at,
        "metrics": metrics,
    }


class TestZeroPreviousRecordings:
    """Tests for when there are 0 previous recordings (total=1 with current)."""

    def test_no_history_returns_none(self, analyzer: TrendAnalyzer):
        """With 0 previous recordings, total is 1 (only current), should return None."""
        current = _make_analysis("current-1", datetime(2024, 1, 1))

        result = analyzer.build_trend_data([], current)

        assert result is None

    def test_empty_history_list_returns_none(self, analyzer: TrendAnalyzer):
        """Explicitly empty history list with current should return None."""
        current = _make_analysis("current-1", datetime(2024, 6, 15, 10, 0))

        result = analyzer.build_trend_data(analysis_history=[], current_analysis=current)

        assert result is None


class TestMinimumRecordings:
    """Tests for 1 previous recording (total=2 with current) → returns TrendData."""

    def test_one_previous_returns_trend_data(self, analyzer: TrendAnalyzer):
        """With 1 previous recording + current = 2 total, should return TrendData."""
        history = [_make_analysis("prev-1", datetime(2024, 1, 1))]
        current = _make_analysis("current-1", datetime(2024, 1, 15))

        result = analyzer.build_trend_data(history, current)

        assert result is not None
        assert isinstance(result, TrendData)

    def test_total_recordings_is_2(self, analyzer: TrendAnalyzer):
        """total_recordings should reflect the combined count."""
        history = [_make_analysis("prev-1", datetime(2024, 1, 1))]
        current = _make_analysis("current-1", datetime(2024, 1, 15))

        result = analyzer.build_trend_data(history, current)

        assert result.total_recordings == 2

    def test_two_previous_returns_trend_data(self, analyzer: TrendAnalyzer):
        """With 2 previous recordings + current = 3 total, should return TrendData."""
        history = [
            _make_analysis("prev-1", datetime(2024, 1, 1)),
            _make_analysis("prev-2", datetime(2024, 1, 10)),
        ]
        current = _make_analysis("current-1", datetime(2024, 1, 20))

        result = analyzer.build_trend_data(history, current)

        assert result is not None
        assert result.total_recordings == 3


class TestLimitTo30Recordings:
    """Tests for 30+ recordings → limits to 30 most recent."""

    def test_35_recordings_limited_to_30(self, analyzer: TrendAnalyzer):
        """With 35 total recordings, should only include the 30 most recent."""
        base_time = datetime(2024, 1, 1)
        # 34 previous + 1 current = 35 total
        history = [
            _make_analysis(f"prev-{i}", base_time + timedelta(days=i))
            for i in range(34)
        ]
        current = _make_analysis("current-1", base_time + timedelta(days=34))

        result = analyzer.build_trend_data(history, current)

        assert result is not None
        # Each recording has 2 metrics, so each metric should have 30 data points
        for metric_name, data_points in result.metrics_history.items():
            assert len(data_points) == 30

    def test_30_recordings_includes_all(self, analyzer: TrendAnalyzer):
        """With exactly 30 total recordings, should include all."""
        base_time = datetime(2024, 1, 1)
        history = [
            _make_analysis(f"prev-{i}", base_time + timedelta(days=i))
            for i in range(29)
        ]
        current = _make_analysis("current-1", base_time + timedelta(days=29))

        result = analyzer.build_trend_data(history, current)

        assert result is not None
        for metric_name, data_points in result.metrics_history.items():
            assert len(data_points) == 30

    def test_oldest_recordings_excluded(self, analyzer: TrendAnalyzer):
        """The oldest recordings should be excluded when limiting to 30."""
        base_time = datetime(2024, 1, 1)
        # 34 previous + 1 current = 35 total; oldest 5 should be excluded
        history = [
            _make_analysis(f"prev-{i}", base_time + timedelta(days=i))
            for i in range(34)
        ]
        current = _make_analysis("current-1", base_time + timedelta(days=34))

        result = analyzer.build_trend_data(history, current)

        # The first included recording should be day 5 (prev-5)
        first_points = result.metrics_history["bat_speed"]
        assert first_points[0].analysis_id == "prev-5"
        # The last should be the current
        assert first_points[-1].analysis_id == "current-1"


class TestChronologicalOrdering:
    """Tests for chronological ordering of data points."""

    def test_data_points_in_chronological_order(self, analyzer: TrendAnalyzer):
        """Data points should be sorted by recorded_at ascending."""
        # Provide history in non-chronological order
        history = [
            _make_analysis("prev-3", datetime(2024, 3, 1)),
            _make_analysis("prev-1", datetime(2024, 1, 1)),
            _make_analysis("prev-2", datetime(2024, 2, 1)),
        ]
        current = _make_analysis("current-1", datetime(2024, 4, 1))

        result = analyzer.build_trend_data(history, current)

        data_points = result.metrics_history["bat_speed"]
        for i in range(len(data_points) - 1):
            assert data_points[i].recorded_at <= data_points[i + 1].recorded_at

    def test_current_analysis_appears_last_when_most_recent(self, analyzer: TrendAnalyzer):
        """Current analysis should appear last if it has the most recent timestamp."""
        history = [
            _make_analysis("prev-1", datetime(2024, 1, 1)),
            _make_analysis("prev-2", datetime(2024, 2, 1)),
        ]
        current = _make_analysis("current-1", datetime(2024, 3, 1))

        result = analyzer.build_trend_data(history, current)

        data_points = result.metrics_history["bat_speed"]
        assert data_points[-1].analysis_id == "current-1"

    def test_date_range_reflects_included_recordings(self, analyzer: TrendAnalyzer):
        """date_range should be (earliest, latest) of included recordings."""
        history = [_make_analysis("prev-1", datetime(2024, 1, 10))]
        current = _make_analysis("current-1", datetime(2024, 6, 20))

        result = analyzer.build_trend_data(history, current)

        assert result.date_range == (datetime(2024, 1, 10), datetime(2024, 6, 20))


class TestMetricsHistoryContainsAllMetrics:
    """Tests for metrics_history containing all measured metrics."""

    def test_all_metrics_present_in_history(self, analyzer: TrendAnalyzer):
        """metrics_history should contain keys for all metrics in the recordings."""
        metrics = {
            "bat_speed": {"value": 120.0, "rating": "within_range"},
            "attack_angle": {"value": 12.0, "rating": "above_range"},
            "hand_path_efficiency": {"value": 0.85, "rating": "within_range"},
            "attack_angle": {"value": 10.0, "rating": "below_range"},
        }
        history = [_make_analysis("prev-1", datetime(2024, 1, 1), metrics)]
        current = _make_analysis("current-1", datetime(2024, 2, 1), metrics)

        result = analyzer.build_trend_data(history, current)

        assert set(result.metrics_history.keys()) == {
            "bat_speed", "attack_angle", "hand_path_efficiency", "attack_angle"
        }

    def test_metric_with_varying_presence_across_recordings(self, analyzer: TrendAnalyzer):
        """Metrics not present in all recordings should still be tracked."""
        history = [
            _make_analysis("prev-1", datetime(2024, 1, 1), {
                "bat_speed": {"value": 110.0, "rating": "within_range"},
            }),
        ]
        current = _make_analysis("current-1", datetime(2024, 2, 1), {
            "bat_speed": {"value": 120.0, "rating": "within_range"},
            "attack_angle": {"value": 12.0, "rating": "above_range"},
        })

        result = analyzer.build_trend_data(history, current)

        assert "bat_speed" in result.metrics_history
        assert "attack_angle" in result.metrics_history
        # bat_speed has 2 data points, launch_angle has 1
        assert len(result.metrics_history["bat_speed"]) == 2
        assert len(result.metrics_history["attack_angle"]) == 1


class TestMetricDataPointFields:
    """Tests for MetricDataPoint fields being populated correctly."""

    def test_analysis_id_populated(self, analyzer: TrendAnalyzer):
        """Each data point should have the correct analysis_id."""
        history = [_make_analysis("prev-1", datetime(2024, 1, 1))]
        current = _make_analysis("current-1", datetime(2024, 2, 1))

        result = analyzer.build_trend_data(history, current)

        data_points = result.metrics_history["bat_speed"]
        assert data_points[0].analysis_id == "prev-1"
        assert data_points[1].analysis_id == "current-1"

    def test_recorded_at_populated(self, analyzer: TrendAnalyzer):
        """Each data point should have the correct recorded_at datetime."""
        t1 = datetime(2024, 1, 1, 10, 30)
        t2 = datetime(2024, 2, 15, 14, 0)
        history = [_make_analysis("prev-1", t1)]
        current = _make_analysis("current-1", t2)

        result = analyzer.build_trend_data(history, current)

        data_points = result.metrics_history["bat_speed"]
        assert data_points[0].recorded_at == t1
        assert data_points[1].recorded_at == t2

    def test_value_populated(self, analyzer: TrendAnalyzer):
        """Each data point should have the correct metric value."""
        history = [_make_analysis("prev-1", datetime(2024, 1, 1), {
            "bat_speed": {"value": 115.5, "rating": "within_range"},
        })]
        current = _make_analysis("current-1", datetime(2024, 2, 1), {
            "bat_speed": {"value": 122.3, "rating": "above_range"},
        })

        result = analyzer.build_trend_data(history, current)

        data_points = result.metrics_history["bat_speed"]
        assert data_points[0].value == 115.5
        assert data_points[1].value == 122.3

    def test_rating_populated_from_string(self, analyzer: TrendAnalyzer):
        """Rating should be parsed correctly from string values."""
        history = [_make_analysis("prev-1", datetime(2024, 1, 1), {
            "bat_speed": {"value": 100.0, "rating": "below_range"},
        })]
        current = _make_analysis("current-1", datetime(2024, 2, 1), {
            "bat_speed": {"value": 130.0, "rating": "above_range"},
        })

        result = analyzer.build_trend_data(history, current)

        data_points = result.metrics_history["bat_speed"]
        assert data_points[0].rating == MetricRating.BELOW_RANGE
        assert data_points[1].rating == MetricRating.ABOVE_RANGE

    def test_rating_populated_from_enum(self, analyzer: TrendAnalyzer):
        """Rating should work when passed as MetricRating enum directly."""
        history = [_make_analysis("prev-1", datetime(2024, 1, 1), {
            "bat_speed": {"value": 100.0, "rating": MetricRating.WITHIN_RANGE},
        })]
        current = _make_analysis("current-1", datetime(2024, 2, 1), {
            "bat_speed": {"value": 130.0, "rating": MetricRating.ABOVE_RANGE},
        })

        result = analyzer.build_trend_data(history, current)

        data_points = result.metrics_history["bat_speed"]
        assert data_points[0].rating == MetricRating.WITHIN_RANGE
        assert data_points[1].rating == MetricRating.ABOVE_RANGE

    def test_data_point_is_metric_data_point_instance(self, analyzer: TrendAnalyzer):
        """Each data point should be an instance of MetricDataPoint."""
        history = [_make_analysis("prev-1", datetime(2024, 1, 1))]
        current = _make_analysis("current-1", datetime(2024, 2, 1))

        result = analyzer.build_trend_data(history, current)

        for data_points in result.metrics_history.values():
            for dp in data_points:
                assert isinstance(dp, MetricDataPoint)
