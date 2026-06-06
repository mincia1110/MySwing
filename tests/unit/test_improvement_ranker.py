"""Unit tests for ImprovementRanker (Task 10.3).

Tests cover:
- Top 3 selection from 5 metrics
- Descending order by deviation
- Fewer than 3 metrics with deviation
- All metrics within range (no improvements)
- Rank assignment (1, 2, 3)
- target_range contains correct reference values
"""

import pytest

from app.models.enums import MetricRating
from app.models.evaluation import ImprovementArea, MetricEvaluation
from app.pipeline.swing_evaluator import ImprovementRanker


@pytest.fixture
def ranker() -> ImprovementRanker:
    """Create an ImprovementRanker instance."""
    return ImprovementRanker()


def _make_evaluation(
    metric_name: str,
    measured_value: float,
    deviation_percent: float,
    rating: MetricRating,
    reference_min: float = 70.0,
    reference_max: float = 100.0,
) -> MetricEvaluation:
    """Helper to create a MetricEvaluation with sensible defaults."""
    return MetricEvaluation(
        metric_name=metric_name,
        measured_value=measured_value,
        unit="km/h",
        reference_min=reference_min,
        reference_max=reference_max,
        deviation_percent=deviation_percent,
        rating=rating,
        color_code="red" if rating != MetricRating.WITHIN_RANGE else "green",
    )


class TestTop3Selection:
    """Tests for selecting top 3 improvement areas from multiple metrics."""

    def test_top_3_from_5_metrics_with_deviation(self, ranker: ImprovementRanker):
        """Should select exactly 3 metrics with largest deviations from 5."""
        evaluations = [
            _make_evaluation("bat_speed", 50.0, 28.6, MetricRating.BELOW_RANGE),
            _make_evaluation("attack_angle", 20.0, 33.3, MetricRating.ABOVE_RANGE, 5.0, 15.0),
            _make_evaluation("hand_path_efficiency", 0.5, 16.7, MetricRating.BELOW_RANGE, 0.6, 0.9),
            _make_evaluation("attack_angle", 18.0, 20.0, MetricRating.ABOVE_RANGE, 5.0, 15.0),
            _make_evaluation("hip_shoulder_separation", 20.0, 14.3, MetricRating.BELOW_RANGE, 25.0, 45.0),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert len(improvements) == 3
        # Top 3 by deviation: launch_angle (33.3), bat_speed (28.6), attack_angle (20.0)
        assert improvements[0].metric_name == "attack_angle"
        assert improvements[1].metric_name == "bat_speed"
        assert improvements[2].metric_name == "attack_angle"

    def test_excludes_within_range_metrics(self, ranker: ImprovementRanker):
        """Metrics within range (deviation=0) should not appear in improvements."""
        evaluations = [
            _make_evaluation("bat_speed", 85.0, 0.0, MetricRating.WITHIN_RANGE),
            _make_evaluation("attack_angle", 20.0, 33.3, MetricRating.ABOVE_RANGE, 5.0, 15.0),
            _make_evaluation("attack_angle", 18.0, 20.0, MetricRating.ABOVE_RANGE, 5.0, 15.0),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert len(improvements) == 2
        metric_names = [imp.metric_name for imp in improvements]
        assert "bat_speed" not in metric_names


class TestDescendingOrder:
    """Tests for descending order by deviation magnitude."""

    def test_sorted_descending_by_deviation(self, ranker: ImprovementRanker):
        """Improvements should be sorted from largest to smallest deviation."""
        evaluations = [
            _make_evaluation("metric_a", 50.0, 10.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_b", 50.0, 30.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_c", 50.0, 20.0, MetricRating.BELOW_RANGE),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert improvements[0].deviation_percent == 30.0
        assert improvements[1].deviation_percent == 20.0
        assert improvements[2].deviation_percent == 10.0

    def test_input_order_does_not_affect_result(self, ranker: ImprovementRanker):
        """Result should be the same regardless of input order."""
        evaluations_a = [
            _make_evaluation("metric_a", 50.0, 10.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_b", 50.0, 30.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_c", 50.0, 20.0, MetricRating.BELOW_RANGE),
        ]
        evaluations_b = [
            _make_evaluation("metric_b", 50.0, 30.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_c", 50.0, 20.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_a", 50.0, 10.0, MetricRating.BELOW_RANGE),
        ]

        improvements_a = ranker.rank_improvements(evaluations_a)
        improvements_b = ranker.rank_improvements(evaluations_b)

        assert [i.metric_name for i in improvements_a] == [i.metric_name for i in improvements_b]


class TestFewerThan3Metrics:
    """Tests for cases with fewer than 3 metrics having deviation."""

    def test_two_metrics_with_deviation(self, ranker: ImprovementRanker):
        """Should return only 2 improvements when only 2 have deviation."""
        evaluations = [
            _make_evaluation("bat_speed", 50.0, 28.6, MetricRating.BELOW_RANGE),
            _make_evaluation("attack_angle", 12.0, 0.0, MetricRating.WITHIN_RANGE, 5.0, 15.0),
            _make_evaluation("attack_angle", 18.0, 20.0, MetricRating.ABOVE_RANGE, 5.0, 15.0),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert len(improvements) == 2

    def test_one_metric_with_deviation(self, ranker: ImprovementRanker):
        """Should return only 1 improvement when only 1 has deviation."""
        evaluations = [
            _make_evaluation("bat_speed", 50.0, 28.6, MetricRating.BELOW_RANGE),
            _make_evaluation("attack_angle", 12.0, 0.0, MetricRating.WITHIN_RANGE, 5.0, 15.0),
            _make_evaluation("attack_angle", 10.0, 0.0, MetricRating.WITHIN_RANGE, 5.0, 15.0),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert len(improvements) == 1
        assert improvements[0].metric_name == "bat_speed"


class TestAllWithinRange:
    """Tests for when all metrics are within range (no improvements needed)."""

    def test_all_within_range_returns_empty(self, ranker: ImprovementRanker):
        """Should return empty list when all metrics are within range."""
        evaluations = [
            _make_evaluation("bat_speed", 85.0, 0.0, MetricRating.WITHIN_RANGE),
            _make_evaluation("attack_angle", 10.0, 0.0, MetricRating.WITHIN_RANGE, 5.0, 15.0),
            _make_evaluation("attack_angle", 10.0, 0.0, MetricRating.WITHIN_RANGE, 5.0, 15.0),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert improvements == []

    def test_empty_evaluations_returns_empty(self, ranker: ImprovementRanker):
        """Should return empty list when no evaluations are provided."""
        improvements = ranker.rank_improvements([])

        assert improvements == []


class TestRankAssignment:
    """Tests for correct rank assignment (1, 2, 3)."""

    def test_rank_1_is_largest_deviation(self, ranker: ImprovementRanker):
        """Rank 1 should be assigned to the metric with the largest deviation."""
        evaluations = [
            _make_evaluation("metric_a", 50.0, 10.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_b", 50.0, 30.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_c", 50.0, 20.0, MetricRating.BELOW_RANGE),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert improvements[0].rank == 1
        assert improvements[0].metric_name == "metric_b"

    def test_ranks_are_sequential_1_2_3(self, ranker: ImprovementRanker):
        """Ranks should be 1, 2, 3 for top 3 improvements."""
        evaluations = [
            _make_evaluation("metric_a", 50.0, 10.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_b", 50.0, 30.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_c", 50.0, 20.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_d", 50.0, 25.0, MetricRating.BELOW_RANGE),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert [imp.rank for imp in improvements] == [1, 2, 3]

    def test_single_improvement_has_rank_1(self, ranker: ImprovementRanker):
        """A single improvement should have rank 1."""
        evaluations = [
            _make_evaluation("bat_speed", 50.0, 28.6, MetricRating.BELOW_RANGE),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert len(improvements) == 1
        assert improvements[0].rank == 1

    def test_two_improvements_have_ranks_1_2(self, ranker: ImprovementRanker):
        """Two improvements should have ranks 1 and 2."""
        evaluations = [
            _make_evaluation("metric_a", 50.0, 10.0, MetricRating.BELOW_RANGE),
            _make_evaluation("metric_b", 50.0, 30.0, MetricRating.BELOW_RANGE),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert [imp.rank for imp in improvements] == [1, 2]


class TestTargetRange:
    """Tests for correct target_range values in ImprovementArea."""

    def test_target_range_contains_reference_min_max(self, ranker: ImprovementRanker):
        """target_range should be (reference_min, reference_max) from the evaluation."""
        evaluations = [
            _make_evaluation(
                "bat_speed", 50.0, 28.6, MetricRating.BELOW_RANGE,
                reference_min=70.0, reference_max=100.0,
            ),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert improvements[0].target_range == (70.0, 100.0)

    def test_target_range_varies_per_metric(self, ranker: ImprovementRanker):
        """Each improvement should have its own correct target_range."""
        evaluations = [
            _make_evaluation(
                "bat_speed", 50.0, 28.6, MetricRating.BELOW_RANGE,
                reference_min=70.0, reference_max=100.0,
            ),
            _make_evaluation(
                "attack_angle", 20.0, 33.3, MetricRating.ABOVE_RANGE,
                reference_min=5.0, reference_max=15.0,
            ),
        ]

        improvements = ranker.rank_improvements(evaluations)

        # launch_angle has larger deviation, so it's rank 1
        assert improvements[0].target_range == (5.0, 15.0)
        assert improvements[1].target_range == (70.0, 100.0)

    def test_current_value_matches_measured_value(self, ranker: ImprovementRanker):
        """current_value should match the measured_value from the evaluation."""
        evaluations = [
            _make_evaluation("bat_speed", 55.5, 20.7, MetricRating.BELOW_RANGE),
        ]

        improvements = ranker.rank_improvements(evaluations)

        assert improvements[0].current_value == 55.5
