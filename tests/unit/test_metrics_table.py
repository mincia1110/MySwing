"""Unit tests for MetricsTableBuilder (Task 11.2).

Tests cover:
- Table has one row per metric
- Value formatting with units
- Color codes are preserved
- Empty evaluations returns empty table
"""

import pytest

from app.models.enums import MetricRating
from app.models.evaluation import MetricEvaluation
from app.pipeline.report_generator import MetricsTableBuilder


@pytest.fixture
def builder() -> MetricsTableBuilder:
    """Create a MetricsTableBuilder instance."""
    return MetricsTableBuilder()


def _make_evaluation(
    metric_name: str = "bat_speed",
    measured_value: float = 108.5,
    unit: str = "km/h",
    color_code: str = "green",
    rating: MetricRating = MetricRating.WITHIN_RANGE,
    reference_min: float = 90.0,
    reference_max: float = 120.0,
    deviation_percent: float = 0.0,
) -> MetricEvaluation:
    """Helper to create a MetricEvaluation with sensible defaults."""
    return MetricEvaluation(
        metric_name=metric_name,
        measured_value=measured_value,
        unit=unit,
        reference_min=reference_min,
        reference_max=reference_max,
        deviation_percent=deviation_percent,
        rating=rating,
        color_code=color_code,
    )


class TestTableRowCount:
    """Tests that the table has exactly one row per metric."""

    def test_single_metric_produces_one_row(self, builder: MetricsTableBuilder):
        """A single evaluation should produce exactly one row."""
        evaluations = [_make_evaluation()]

        table = builder.build_metrics_table(evaluations)

        assert len(table) == 1

    def test_multiple_metrics_produce_matching_rows(self, builder: MetricsTableBuilder):
        """N evaluations should produce exactly N rows."""
        evaluations = [
            _make_evaluation(metric_name="bat_speed", measured_value=108.5, unit="km/h"),
            _make_evaluation(metric_name="attack_angle", measured_value=12.3, unit="degrees"),
            _make_evaluation(metric_name="hip_shoulder_separation", measured_value=45.0, unit="degrees"),
            _make_evaluation(metric_name="kinematic_chain_timing", measured_value=35.2, unit="ms"),
        ]

        table = builder.build_metrics_table(evaluations)

        assert len(table) == 4

    def test_row_metric_names_match_input_order(self, builder: MetricsTableBuilder):
        """Rows should preserve the order of input evaluations."""
        evaluations = [
            _make_evaluation(metric_name="bat_speed"),
            _make_evaluation(metric_name="attack_angle"),
            _make_evaluation(metric_name="attack_angle"),
        ]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["metric_name"] == "bat_speed"
        assert table[1]["metric_name"] == "attack_angle"
        assert table[2]["metric_name"] == "attack_angle"


class TestValueFormatting:
    """Tests for value formatting with units."""

    def test_bat_speed_formatted_with_kmh(self, builder: MetricsTableBuilder):
        """Bat speed should be formatted as '108.5 km/h'."""
        evaluations = [_make_evaluation(measured_value=108.5, unit="km/h")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["measured_value"] == "108.5 km/h"

    def test_launch_angle_formatted_with_degree_symbol(self, builder: MetricsTableBuilder):
        """Launch angle should be formatted as '12.3°'."""
        evaluations = [_make_evaluation(metric_name="attack_angle", measured_value=12.3, unit="degrees")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["measured_value"] == "12.3°"

    def test_degree_unit_symbol_formatted(self, builder: MetricsTableBuilder):
        """Unit '°' should also format correctly as '45.0°'."""
        evaluations = [_make_evaluation(metric_name="separation", measured_value=45.0, unit="°")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["measured_value"] == "45.0°"

    def test_milliseconds_formatted_with_ms(self, builder: MetricsTableBuilder):
        """Timing values should be formatted as '35.2 ms'."""
        evaluations = [_make_evaluation(metric_name="timing", measured_value=35.2, unit="ms")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["measured_value"] == "35.2 ms"

    def test_ratio_formatted_without_unit(self, builder: MetricsTableBuilder):
        """Ratio values should be formatted as '0.85' with 2 decimal places."""
        evaluations = [_make_evaluation(metric_name="hand_path_efficiency", measured_value=0.85, unit="ratio")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["measured_value"] == "0.85"

    def test_dps_formatted_with_degree_per_second(self, builder: MetricsTableBuilder):
        """Degrees per second should be formatted as '450.0°/s'."""
        evaluations = [_make_evaluation(metric_name="hip_rotation", measured_value=450.0, unit="dps")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["measured_value"] == "450.0°/s"

    def test_percentage_formatted_with_symbol(self, builder: MetricsTableBuilder):
        """Percentage values should be formatted as '15.2%'."""
        evaluations = [_make_evaluation(metric_name="deviation", measured_value=15.2, unit="%")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["measured_value"] == "15.2%"


class TestColorCodePreservation:
    """Tests that color codes are preserved from evaluations."""

    def test_green_color_code_preserved(self, builder: MetricsTableBuilder):
        """Green color code should be preserved in the table row."""
        evaluations = [_make_evaluation(color_code="green")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["color_code"] == "green"

    def test_yellow_color_code_preserved(self, builder: MetricsTableBuilder):
        """Yellow color code should be preserved in the table row."""
        evaluations = [_make_evaluation(color_code="yellow")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["color_code"] == "yellow"

    def test_red_color_code_preserved(self, builder: MetricsTableBuilder):
        """Red color code should be preserved in the table row."""
        evaluations = [_make_evaluation(color_code="red")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["color_code"] == "red"

    def test_green_maps_to_optimal_rating(self, builder: MetricsTableBuilder):
        """Green color code should map to 'optimal' rating label."""
        evaluations = [_make_evaluation(color_code="green")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["rating"] == "optimal"

    def test_yellow_maps_to_acceptable_rating(self, builder: MetricsTableBuilder):
        """Yellow color code should map to 'acceptable' rating label."""
        evaluations = [_make_evaluation(color_code="yellow")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["rating"] == "acceptable"

    def test_red_maps_to_outside_rating(self, builder: MetricsTableBuilder):
        """Red color code should map to 'outside' rating label."""
        evaluations = [_make_evaluation(color_code="red")]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["rating"] == "outside"

    def test_mixed_color_codes_in_table(self, builder: MetricsTableBuilder):
        """Multiple evaluations with different colors should all be preserved."""
        evaluations = [
            _make_evaluation(metric_name="bat_speed", color_code="green"),
            _make_evaluation(metric_name="attack_angle", color_code="yellow"),
            _make_evaluation(metric_name="attack_angle", color_code="red"),
        ]

        table = builder.build_metrics_table(evaluations)

        assert table[0]["color_code"] == "green"
        assert table[1]["color_code"] == "yellow"
        assert table[2]["color_code"] == "red"


class TestEmptyEvaluations:
    """Tests for empty evaluations input."""

    def test_empty_list_returns_empty_table(self, builder: MetricsTableBuilder):
        """Empty evaluations list should return an empty table."""
        table = builder.build_metrics_table([])

        assert table == []

    def test_empty_table_is_a_list(self, builder: MetricsTableBuilder):
        """Empty result should be a list type."""
        table = builder.build_metrics_table([])

        assert isinstance(table, list)


class TestRowStructure:
    """Tests that each row has the required keys."""

    def test_row_contains_all_required_keys(self, builder: MetricsTableBuilder):
        """Each row should contain metric_name, measured_value, color_code, and rating."""
        evaluations = [_make_evaluation()]

        table = builder.build_metrics_table(evaluations)

        row = table[0]
        assert "metric_name" in row
        assert "measured_value" in row
        assert "color_code" in row
        assert "rating" in row

    def test_measured_value_is_string(self, builder: MetricsTableBuilder):
        """measured_value should be a formatted string, not a number."""
        evaluations = [_make_evaluation(measured_value=108.5, unit="km/h")]

        table = builder.build_metrics_table(evaluations)

        assert isinstance(table[0]["measured_value"], str)
