"""Unit tests for ReferenceComparator (Task 10.1).

Tests cover:
- Deviation calculation for below/within/above range
- MetricRating classification
- Color code assignment (green/yellow/red)
- Default level selection (recreational adult)
- Specific level selection (professional, college, etc.)
- All 5 metrics are evaluated when biomechanics data is complete
"""

import pytest

from app.models.biomechanics import (
    LaunchAngleResult,
    BatSpeedResult,
    BiomechanicsResult,
    KinematicChainResult,
    LaunchAngleResult,
    RotationResult,
)
from app.models.enums import MetricRating
from app.models.evaluation import MetricEvaluation
from app.pipeline.swing_evaluator import (
    DEFAULT_AGE_GROUP,
    DEFAULT_LEVEL,
    ReferenceComparator,
    ReferenceRange,
)


# --- Fixtures ---


@pytest.fixture
def comparator() -> ReferenceComparator:
    """Create a ReferenceComparator instance."""
    return ReferenceComparator()


@pytest.fixture
def full_biomechanics() -> BiomechanicsResult:
    """Create a complete BiomechanicsResult with all metrics populated."""
    return BiomechanicsResult(
        bat_speed=BatSpeedResult(speed_kmh=90.0, precision=1.0, measurement_frame=50),
        kinematic_chain=KinematicChainResult(),
        rotation=RotationResult(
            hip_rotation_speed_dps=600.0,
            shoulder_rotation_speed_dps=500.0,
            hip_shoulder_separation_degrees=35.0,
            rotation_phase_start_frame=30,
            rotation_phase_end_frame=50,
        ),
        hand_path_efficiency=0.70,
        attack_angle=LaunchAngleResult(
            angle_degrees=12.0, precision=0.5, impact_frame=50
        ),
    )


# --- Deviation Calculation Tests ---


class TestDeviationCalculation:
    """Tests for _calculate_deviation method."""

    def test_below_range_deviation(self, comparator: ReferenceComparator):
        """Value below min should produce positive deviation with BELOW_RANGE rating."""
        # value=60, min=80 → deviation = (80-60)/80 * 100 = 25%
        deviation, rating = comparator._calculate_deviation(60.0, 80.0, 100.0)
        assert rating == MetricRating.BELOW_RANGE
        assert deviation == pytest.approx(25.0)

    def test_above_range_deviation(self, comparator: ReferenceComparator):
        """Value above max should produce positive deviation with ABOVE_RANGE rating."""
        # value=120, max=100 → deviation = (120-100)/100 * 100 = 20%
        deviation, rating = comparator._calculate_deviation(120.0, 80.0, 100.0)
        assert rating == MetricRating.ABOVE_RANGE
        assert deviation == pytest.approx(20.0)

    def test_within_range_deviation(self, comparator: ReferenceComparator):
        """Value within range should produce 0 deviation with WITHIN_RANGE rating."""
        deviation, rating = comparator._calculate_deviation(90.0, 80.0, 100.0)
        assert rating == MetricRating.WITHIN_RANGE
        assert deviation == 0.0

    def test_at_min_boundary(self, comparator: ReferenceComparator):
        """Value exactly at min should be within range."""
        deviation, rating = comparator._calculate_deviation(80.0, 80.0, 100.0)
        assert rating == MetricRating.WITHIN_RANGE
        assert deviation == 0.0

    def test_at_max_boundary(self, comparator: ReferenceComparator):
        """Value exactly at max should be within range."""
        deviation, rating = comparator._calculate_deviation(100.0, 80.0, 100.0)
        assert rating == MetricRating.WITHIN_RANGE
        assert deviation == 0.0

    def test_slightly_below_min(self, comparator: ReferenceComparator):
        """Value just below min should be below_range with small deviation."""
        # value=79, min=80 → deviation = (80-79)/80 * 100 = 1.25%
        deviation, rating = comparator._calculate_deviation(79.0, 80.0, 100.0)
        assert rating == MetricRating.BELOW_RANGE
        assert deviation == pytest.approx(1.25)

    def test_slightly_above_max(self, comparator: ReferenceComparator):
        """Value just above max should be above_range with small deviation."""
        # value=101, max=100 → deviation = (101-100)/100 * 100 = 1%
        deviation, rating = comparator._calculate_deviation(101.0, 80.0, 100.0)
        assert rating == MetricRating.ABOVE_RANGE
        assert deviation == pytest.approx(1.0)


# --- MetricRating Classification Tests ---


class TestMetricRatingClassification:
    """Tests for MetricRating assignment in compare_with_reference."""

    def test_below_range_rating(self, comparator: ReferenceComparator):
        """Bat speed below recreational min (70) should be BELOW_RANGE."""
        biomechanics = BiomechanicsResult(
            bat_speed=BatSpeedResult(speed_kmh=50.0, precision=1.0, measurement_frame=50),
        )
        evaluations = comparator.compare_with_reference(biomechanics)
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        assert bat_speed_eval.rating == MetricRating.BELOW_RANGE

    def test_within_range_rating(self, comparator: ReferenceComparator):
        """Bat speed within recreational range (70-100) should be WITHIN_RANGE."""
        biomechanics = BiomechanicsResult(
            bat_speed=BatSpeedResult(speed_kmh=85.0, precision=1.0, measurement_frame=50),
        )
        evaluations = comparator.compare_with_reference(biomechanics)
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        assert bat_speed_eval.rating == MetricRating.WITHIN_RANGE

    def test_above_range_rating(self, comparator: ReferenceComparator):
        """Bat speed above recreational max (100) should be ABOVE_RANGE."""
        biomechanics = BiomechanicsResult(
            bat_speed=BatSpeedResult(speed_kmh=115.0, precision=1.0, measurement_frame=50),
        )
        evaluations = comparator.compare_with_reference(biomechanics)
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        assert bat_speed_eval.rating == MetricRating.ABOVE_RANGE


# --- Color Code Tests ---


class TestColorCodeAssignment:
    """Tests for _determine_color_code method."""

    def test_green_within_optimal(self, comparator: ReferenceComparator):
        """Value within optimal range should be green."""
        # recreational bat_speed: optimal 80-95, acceptable 70-100
        color = comparator._determine_color_code(
            MetricRating.WITHIN_RANGE, 85.0, 80.0, 95.0, 70.0, 100.0
        )
        assert color == "green"

    def test_yellow_between_acceptable_and_optimal_low(self, comparator: ReferenceComparator):
        """Value between acceptable min and optimal min should be yellow."""
        # value=75, optimal_min=80, ref_min=70
        color = comparator._determine_color_code(
            MetricRating.WITHIN_RANGE, 75.0, 80.0, 95.0, 70.0, 100.0
        )
        assert color == "yellow"

    def test_yellow_between_optimal_and_acceptable_high(self, comparator: ReferenceComparator):
        """Value between optimal max and acceptable max should be yellow."""
        # value=98, optimal_max=95, ref_max=100
        color = comparator._determine_color_code(
            MetricRating.WITHIN_RANGE, 98.0, 80.0, 95.0, 70.0, 100.0
        )
        assert color == "yellow"

    def test_red_below_acceptable(self, comparator: ReferenceComparator):
        """Value below acceptable min should be red."""
        color = comparator._determine_color_code(
            MetricRating.BELOW_RANGE, 60.0, 80.0, 95.0, 70.0, 100.0
        )
        assert color == "red"

    def test_red_above_acceptable(self, comparator: ReferenceComparator):
        """Value above acceptable max should be red."""
        color = comparator._determine_color_code(
            MetricRating.ABOVE_RANGE, 110.0, 80.0, 95.0, 70.0, 100.0
        )
        assert color == "red"

    def test_green_at_optimal_min_boundary(self, comparator: ReferenceComparator):
        """Value exactly at optimal_min should be green."""
        color = comparator._determine_color_code(
            MetricRating.WITHIN_RANGE, 80.0, 80.0, 95.0, 70.0, 100.0
        )
        assert color == "green"

    def test_green_at_optimal_max_boundary(self, comparator: ReferenceComparator):
        """Value exactly at optimal_max should be green."""
        color = comparator._determine_color_code(
            MetricRating.WITHIN_RANGE, 95.0, 80.0, 95.0, 70.0, 100.0
        )
        assert color == "green"


# --- Default Level Selection Tests ---


class TestDefaultLevelSelection:
    """Tests for default level/age_group selection when not specified."""

    def test_none_level_defaults_to_recreational(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """None level should default to recreational."""
        evaluations = comparator.compare_with_reference(full_biomechanics, level=None)
        # Recreational adult bat_speed range: 70-100
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        assert bat_speed_eval.reference_min == 70.0
        assert bat_speed_eval.reference_max == 100.0

    def test_empty_string_level_defaults_to_recreational(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """Empty string level should default to recreational."""
        evaluations = comparator.compare_with_reference(full_biomechanics, level="")
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        assert bat_speed_eval.reference_min == 70.0
        assert bat_speed_eval.reference_max == 100.0

    def test_none_age_group_defaults_to_adult(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """None age_group should default to adult."""
        evaluations = comparator.compare_with_reference(
            full_biomechanics, level="recreational", age_group=None
        )
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        assert bat_speed_eval.reference_min == 70.0
        assert bat_speed_eval.reference_max == 100.0

    def test_no_arguments_uses_recreational_adult(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """No level/age_group arguments should use recreational adult defaults."""
        evaluations = comparator.compare_with_reference(full_biomechanics)
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        assert bat_speed_eval.reference_min == 70.0
        assert bat_speed_eval.reference_max == 100.0


# --- Specific Level Tests ---


class TestSpecificLevelSelection:
    """Tests for specific level/age_group selection."""

    def test_professional_level(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """Professional level should use professional reference data."""
        evaluations = comparator.compare_with_reference(
            full_biomechanics, level="professional", age_group="adult"
        )
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        # Professional bat_speed range: 110-130
        assert bat_speed_eval.reference_min == 110.0
        assert bat_speed_eval.reference_max == 130.0

    def test_college_level(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """College level should use college reference data."""
        evaluations = comparator.compare_with_reference(
            full_biomechanics, level="college", age_group="adult"
        )
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        # College bat_speed range: 100-120
        assert bat_speed_eval.reference_min == 100.0
        assert bat_speed_eval.reference_max == 120.0

    def test_high_school_level(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """High school level should use high school reference data."""
        evaluations = comparator.compare_with_reference(
            full_biomechanics, level="high_school", age_group="adult"
        )
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        # High school adult bat_speed range: 85-110
        assert bat_speed_eval.reference_min == 85.0
        assert bat_speed_eval.reference_max == 110.0

    def test_high_school_youth(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """High school youth should use youth-specific reference data."""
        evaluations = comparator.compare_with_reference(
            full_biomechanics, level="high_school", age_group="youth"
        )
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        # High school youth bat_speed range: 70-95
        assert bat_speed_eval.reference_min == 70.0
        assert bat_speed_eval.reference_max == 95.0

    def test_unknown_level_falls_back_to_recreational(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """Unknown level should fall back to recreational adult."""
        evaluations = comparator.compare_with_reference(
            full_biomechanics, level="unknown_level", age_group="adult"
        )
        bat_speed_eval = next(e for e in evaluations if e.metric_name == "bat_speed")
        assert bat_speed_eval.reference_min == 70.0
        assert bat_speed_eval.reference_max == 100.0


# --- All Metrics Evaluated Tests ---


class TestAllMetricsEvaluated:
    """Tests that all 5 metrics are evaluated when data is complete."""

    def test_all_five_metrics_present(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """All 5 metrics should be evaluated when biomechanics data is complete."""
        evaluations = comparator.compare_with_reference(full_biomechanics)
        metric_names = {e.metric_name for e in evaluations}
        expected_metrics = {
            "bat_speed",
            "attack_angle",
            "hip_shoulder_separation",
            "hand_path_efficiency",
            "attack_angle",
        }
        assert metric_names == expected_metrics

    def test_partial_biomechanics_only_evaluates_available(self, comparator: ReferenceComparator):
        """Only available metrics should be evaluated when data is partial."""
        biomechanics = BiomechanicsResult(
            bat_speed=BatSpeedResult(speed_kmh=90.0, precision=1.0, measurement_frame=50),
        )
        evaluations = comparator.compare_with_reference(biomechanics)
        metric_names = {e.metric_name for e in evaluations}
        assert metric_names == {"bat_speed"}

    def test_empty_biomechanics_returns_empty(self, comparator: ReferenceComparator):
        """Empty biomechanics should return no evaluations."""
        biomechanics = BiomechanicsResult()
        evaluations = comparator.compare_with_reference(biomechanics)
        assert evaluations == []

    def test_evaluation_contains_correct_units(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """Each evaluation should have the correct unit."""
        evaluations = comparator.compare_with_reference(full_biomechanics)
        units_map = {e.metric_name: e.unit for e in evaluations}
        assert units_map["bat_speed"] == "km/h"
        assert units_map["attack_angle"] == "degrees"
        assert units_map["hip_shoulder_separation"] == "degrees"
        assert units_map["hand_path_efficiency"] == "ratio"
        assert units_map["attack_angle"] == "degrees"

    def test_evaluation_measured_values_match_input(self, comparator: ReferenceComparator, full_biomechanics: BiomechanicsResult):
        """Measured values in evaluations should match the input biomechanics."""
        evaluations = comparator.compare_with_reference(full_biomechanics)
        values_map = {e.metric_name: e.measured_value for e in evaluations}
        assert values_map["bat_speed"] == 90.0
        assert values_map["attack_angle"] == 12.0
        assert values_map["hip_shoulder_separation"] == 35.0
        assert values_map["hand_path_efficiency"] == 0.70

    def test_front_knee_extension_near_180_is_within_range(
        self, comparator: ReferenceComparator
    ):
        """Lead-leg bracing uses extension angle, so near-180° is valid."""
        biomechanics = BiomechanicsResult(front_knee_extension_degrees=178.0)

        evaluations = comparator.compare_with_reference(biomechanics)
        knee_eval = next(
            e for e in evaluations
            if e.metric_name == "front_knee_extension_degrees"
        )

        assert knee_eval.rating == MetricRating.WITHIN_RANGE
        assert knee_eval.color_code == "green"
        assert knee_eval.reference_min == 155.0
        assert knee_eval.reference_max == 180.0

    def test_front_knee_extension_falls_back_to_legacy_field(
        self, comparator: ReferenceComparator
    ):
        """Legacy API field is still interpreted as the same extension angle."""
        biomechanics = BiomechanicsResult(front_knee_flexion_degrees=166.0)

        evaluations = comparator.compare_with_reference(biomechanics)
        knee_eval = next(
            e for e in evaluations
            if e.metric_name == "front_knee_extension_degrees"
        )

        assert knee_eval.measured_value == 166.0
        assert knee_eval.rating == MetricRating.WITHIN_RANGE
