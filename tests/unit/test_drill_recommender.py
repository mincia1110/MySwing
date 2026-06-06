"""Unit tests for DrillRecommender and ComparisonViewBuilder (Task 11.3).

Tests cover:
- 1-3 drills per improvement area
- drill_name and target_metric are populated
- All 5 metrics have drill mappings
- Comparison view structure
- Significant difference highlighting

Validates: Requirements 8.4, 8.5
"""

import pytest

from app.models.enums import SwingPhase
from app.models.evaluation import ImprovementArea
from app.models.swing import SwingPhaseResult
from app.pipeline.report_generator import (
    DRILL_DATABASE,
    PROTECTED_DIRECTIONAL_METRICS,
    ComparisonViewBuilder,
    DrillRecommender,
)


@pytest.fixture
def recommender() -> DrillRecommender:
    """Create a DrillRecommender instance."""
    return DrillRecommender()


@pytest.fixture
def comparison_builder() -> ComparisonViewBuilder:
    """Create a ComparisonViewBuilder instance."""
    return ComparisonViewBuilder()


def _make_improvement(
    metric_name: str,
    deviation_percent: float = 25.0,
    current_value: float = 50.0,
    target_range: tuple[float, float] = (70.0, 100.0),
    rank: int = 1,
    rating: str = "below_range",
) -> ImprovementArea:
    """Helper to create an ImprovementArea with sensible defaults."""
    return ImprovementArea(
        metric_name=metric_name,
        deviation_percent=deviation_percent,
        current_value=current_value,
        target_range=target_range,
        rank=rank,
        rating=rating,
    )


class TestDrillCountPerImprovement:
    """Tests for 1-3 drills per improvement area."""

    def test_high_deviation_gets_3_drills(self, recommender: DrillRecommender):
        """Deviation >= 30% should recommend 3 drills."""
        improvements = [_make_improvement("bat_speed", deviation_percent=35.0)]

        drills = recommender.recommend_drills(improvements)

        bat_speed_drills = [d for d in drills if d.target_metric == "bat_speed"]
        assert len(bat_speed_drills) == 3

    def test_medium_deviation_gets_2_drills(self, recommender: DrillRecommender):
        """Deviation between 15% and 30% should recommend 2 drills."""
        improvements = [_make_improvement("bat_speed", deviation_percent=20.0)]

        drills = recommender.recommend_drills(improvements)

        bat_speed_drills = [d for d in drills if d.target_metric == "bat_speed"]
        assert len(bat_speed_drills) == 2

    def test_low_deviation_gets_1_drill(self, recommender: DrillRecommender):
        """Deviation < 15% should recommend 1 drill."""
        improvements = [_make_improvement("bat_speed", deviation_percent=10.0)]

        drills = recommender.recommend_drills(improvements)

        bat_speed_drills = [d for d in drills if d.target_metric == "bat_speed"]
        assert len(bat_speed_drills) == 1

    def test_multiple_improvements_each_get_drills(self, recommender: DrillRecommender):
        """Each improvement area should get its own set of drills."""
        improvements = [
            _make_improvement("bat_speed", deviation_percent=35.0, rank=1),
            _make_improvement("attack_angle", deviation_percent=20.0, rank=2),
            _make_improvement("attack_angle", deviation_percent=10.0, rank=3),
        ]

        drills = recommender.recommend_drills(improvements)

        bat_speed_drills = [d for d in drills if d.target_metric == "bat_speed"]
        attack_angle_drills = [d for d in drills if d.target_metric == "attack_angle"]

        assert len(bat_speed_drills) == 3
        assert len(attack_angle_drills) == 3

    def test_empty_improvements_returns_empty(self, recommender: DrillRecommender):
        """No improvements should return no drills."""
        drills = recommender.recommend_drills([])
        assert drills == []


class TestDrillFieldsPopulated:
    """Tests for drill_name and target_metric being populated."""

    def test_drill_name_is_populated(self, recommender: DrillRecommender):
        """Every drill recommendation should have a non-empty drill_name."""
        improvements = [_make_improvement("bat_speed", deviation_percent=35.0)]

        drills = recommender.recommend_drills(improvements)

        for drill in drills:
            assert drill.drill_name
            assert len(drill.drill_name) > 0

    def test_target_metric_is_populated(self, recommender: DrillRecommender):
        """Every drill recommendation should have a non-empty target_metric."""
        improvements = [_make_improvement("attack_angle", deviation_percent=25.0)]

        drills = recommender.recommend_drills(improvements)

        for drill in drills:
            assert drill.target_metric == "attack_angle"

    def test_description_is_populated(self, recommender: DrillRecommender):
        """Every drill recommendation should have a non-empty description."""
        improvements = [_make_improvement("hip_shoulder_separation", deviation_percent=35.0)]

        drills = recommender.recommend_drills(improvements)

        for drill in drills:
            assert drill.description
            assert len(drill.description) > 0

    def test_target_metric_matches_improvement_metric(self, recommender: DrillRecommender):
        """Each drill's target_metric should match the improvement area's metric_name."""
        improvements = [
            _make_improvement("bat_speed", deviation_percent=35.0, rank=1),
            _make_improvement("hand_path_efficiency", deviation_percent=20.0, rank=2),
        ]

        drills = recommender.recommend_drills(improvements)

        for drill in drills:
            assert drill.target_metric in ["bat_speed", "hand_path_efficiency"]


class TestAllMetricsHaveDrillMappings:
    """Tests that all reported metrics have drill mappings in the database."""

    EXPECTED_METRICS = [
        "bat_speed",
        "attack_angle",
        "hip_shoulder_separation",
        "hand_path_efficiency",
        "stride_length_cm",
        "cog_sway_cm",
        "cog_drop_cm",
        "head_stability_cm",
        "front_knee_flexion_degrees",
        "spine_angle_degrees",
    ]

    def test_all_metrics_exist_in_database(self):
        """The drill database should contain entries for all reported metrics."""
        assert len(set(self.EXPECTED_METRICS)) == len(self.EXPECTED_METRICS)
        for metric in self.EXPECTED_METRICS:
            assert metric in DRILL_DATABASE, f"Missing drill mapping for {metric}"

    def test_each_metric_has_below_and_above_keys(self):
        """Each metric should declare both below/above directions explicitly."""
        for metric in self.EXPECTED_METRICS:
            assert "below" in DRILL_DATABASE[metric]
            assert "above" in DRILL_DATABASE[metric]

    def test_each_direction_has_at_least_1_drill(self):
        """Each direction of a reported metric should have at least 1 drill."""
        for metric in self.EXPECTED_METRICS:
            for direction in ("below", "above"):
                assert (
                    len(DRILL_DATABASE[metric][direction]) >= 1
                ), f"{metric}/{direction} has no drill"

    def test_each_direction_has_at_most_3_drills(self):
        """Each direction of a reported metric should have at most 3 drills."""
        for metric in self.EXPECTED_METRICS:
            for direction in ("below", "above"):
                assert (
                    len(DRILL_DATABASE[metric][direction]) <= 3
                ), f"{metric}/{direction} has more than 3 drills"

    def test_all_metrics_produce_recommendations(self, recommender: DrillRecommender):
        """Recommending drills for each metric should produce results."""
        for metric in self.EXPECTED_METRICS:
            improvements = [_make_improvement(metric, deviation_percent=35.0)]
            drills = recommender.recommend_drills(improvements)
            assert len(drills) >= 1, f"No drills recommended for {metric}"

    def test_unknown_metric_gets_generic_drill(self, recommender: DrillRecommender):
        """An unknown metric should still get a generic Korean recommendation."""
        improvements = [_make_improvement("unknown_metric", deviation_percent=25.0)]

        drills = recommender.recommend_drills(improvements)

        assert len(drills) == 1
        assert drills[0].target_metric == "unknown_metric"
        assert drills[0].direction == "generic"
        assert drills[0].drill_name
        assert "전용 드릴 데이터" in drills[0].description


class TestDirectionalSelection:
    """Tests that below/above ratings surface distinct drill sets."""

    @pytest.mark.parametrize(
        "metric",
        [
            "bat_speed",
            "attack_angle",
            "hip_shoulder_separation",
            "hand_path_efficiency",
            "stride_length_cm",
            "cog_sway_cm",
            "cog_drop_cm",
            "head_stability_cm",
            "front_knee_flexion_degrees",
            "spine_angle_degrees",
        ],
    )
    def test_below_and_above_return_distinct_drill_names(
        self, recommender: DrillRecommender, metric: str
    ):
        """Same metric with below vs above rating should yield different drill names."""
        below = recommender.recommend_drills(
            [
                _make_improvement(
                    metric, deviation_percent=40.0, rating="below_range"
                )
            ]
        )
        above = recommender.recommend_drills(
            [
                _make_improvement(
                    metric, deviation_percent=40.0, rating="above_range"
                )
            ]
        )
        assert below and above
        below_names = {d.drill_name for d in below}
        above_names = {d.drill_name for d in above}
        assert below_names.isdisjoint(above_names), (
            f"{metric}: same drill name across below/above -> "
            f"below={below_names}, above={above_names}"
        )
        assert {d.direction for d in below} == {"below"}
        assert {d.direction for d in above} == {"above"}

    def test_protected_metric_missing_direction_does_not_fall_back(
        self, recommender: DrillRecommender
    ):
        """Protected metrics with one missing direction must not silently use the other."""
        # hand_path_efficiency is protected; clear one direction and ensure no fallback.
        original_above = DRILL_DATABASE["hand_path_efficiency"].pop("above", None)
        try:
            drills = recommender.recommend_drills(
                [
                    _make_improvement(
                        "hand_path_efficiency",
                        deviation_percent=35.0,
                        rating="above_range",
                    )
                ]
            )
            assert len(drills) == 1
            assert drills[0].direction == "above"
            assert drills[0].drill_name == "맞춤 훈련 설계 필요"
            assert "전용 드릴 미정" in drills[0].description
        finally:
            DRILL_DATABASE["hand_path_efficiency"]["above"] = original_above or []

    def test_protected_set_covers_all_directional_metrics(self):
        """Every directional metric in the database should be in the protected set."""
        for metric in DRILL_DATABASE:
            assert metric in PROTECTED_DIRECTIONAL_METRICS, (
                f"metric {metric!r} is missing from PROTECTED_DIRECTIONAL_METRICS; "
                "this could silently recommend the wrong-direction drill."
            )


class TestKoreanDrillContent:
    """Tests for Korean localization of drill names and descriptions."""

    KOREAN_HANGUL_RE = __import__("re").compile(r"[가-힣]")

    @pytest.mark.parametrize(
        "metric",
        [
            "bat_speed",
            "attack_angle",
            "hip_shoulder_separation",
            "hand_path_efficiency",
            "stride_length_cm",
            "cog_sway_cm",
            "cog_drop_cm",
            "head_stability_cm",
            "front_knee_flexion_degrees",
            "spine_angle_degrees",
        ],
    )
    def test_drill_database_is_korean(self, metric: str):
        """Default database drill names/descriptions should be in Korean."""
        for direction in ("below", "above"):
            for entry in DRILL_DATABASE[metric][direction]:
                name = entry["drill_name"]
                desc = entry["description"]
                assert not name.isascii() or " " in name, (
                    f"{metric}/{direction} drill name {name!r} looks English"
                )
                hangul = self.KOREAN_HANGUL_RE.findall(desc)
                assert len(hangul) >= max(8, len(desc) // 4), (
                    f"{metric}/{direction} description is not mostly Korean: {desc!r}"
                )

    def test_recommendation_output_never_returns_english_generic(
        self, recommender: DrillRecommender
    ):
        """Even the generic fallback should not surface the old English 'General ... Drill' name."""
        drills = recommender.recommend_drills(
            [_make_improvement("unknown_metric", deviation_percent=20.0)]
        )
        assert drills, "unknown metrics must still surface at least one item"
        for drill in drills:
            assert "General" not in drill.drill_name
            assert "Drill" not in drill.drill_name or "드릴" in drill.drill_name

    def test_drill_copy_avoids_translation_artifacts(self):
        """Product-facing Korean copy should not expose awkward translation artifacts."""
        banned_terms = [
            "伸展",
            "토레소",
            "어퍼힐",
            "코그",
            "컨트롤드",
            "힙 트랩",
            "워치 더 백",
        ]

        for metric, directions in DRILL_DATABASE.items():
            for direction, entries in directions.items():
                for entry in entries:
                    text = f"{entry['drill_name']} {entry['description']}"
                    for term in banned_terms:
                        assert term not in text, (
                            f"{metric}/{direction} contains translation artifact "
                            f"{term!r}: {text}"
                        )


class TestComparisonViewStructure:
    """Tests for comparison view structure and content."""

    def _make_user_phases(self) -> SwingPhaseResult:
        """Create a sample user SwingPhaseResult."""
        return SwingPhaseResult(
            phases={
                SwingPhase.STANCE: (0, 30),
                SwingPhase.LOAD: (30, 50),
                SwingPhase.STRIDE: (50, 70),
                SwingPhase.ROTATION: (70, 85),
                SwingPhase.IMPACT: (85, 90),
                SwingPhase.FOLLOW_THROUGH: (90, 120),
            },
            transitions=[],
            phase_durations_ms={
                SwingPhase.STANCE: 500.0,
                SwingPhase.LOAD: 333.3,
                SwingPhase.STRIDE: 333.3,
                SwingPhase.ROTATION: 250.0,
                SwingPhase.IMPACT: 83.3,
                SwingPhase.FOLLOW_THROUGH: 500.0,
            },
            anomalies=[],
        )

    def _make_reference_phases(self) -> SwingPhaseResult:
        """Create a sample reference (pro) SwingPhaseResult."""
        return SwingPhaseResult(
            phases={
                SwingPhase.STANCE: (0, 25),
                SwingPhase.LOAD: (25, 45),
                SwingPhase.STRIDE: (45, 60),
                SwingPhase.ROTATION: (60, 75),
                SwingPhase.IMPACT: (75, 78),
                SwingPhase.FOLLOW_THROUGH: (78, 110),
            },
            transitions=[],
            phase_durations_ms={
                SwingPhase.STANCE: 416.7,
                SwingPhase.LOAD: 333.3,
                SwingPhase.STRIDE: 250.0,
                SwingPhase.ROTATION: 250.0,
                SwingPhase.IMPACT: 50.0,
                SwingPhase.FOLLOW_THROUGH: 533.3,
            },
            anomalies=[],
        )

    def test_comparison_has_phase_comparisons(self, comparison_builder: ComparisonViewBuilder):
        """Comparison view should contain phase_comparisons list."""
        user = self._make_user_phases()
        reference = self._make_reference_phases()

        result = comparison_builder.build_comparison(user, reference)

        assert "phase_comparisons" in result
        assert isinstance(result["phase_comparisons"], list)

    def test_comparison_has_all_phases(self, comparison_builder: ComparisonViewBuilder):
        """Comparison should include all 6 swing phases."""
        user = self._make_user_phases()
        reference = self._make_reference_phases()

        result = comparison_builder.build_comparison(user, reference)

        phase_names = [pc["phase"] for pc in result["phase_comparisons"]]
        assert len(phase_names) == 6

    def test_each_phase_has_required_fields(self, comparison_builder: ComparisonViewBuilder):
        """Each phase comparison should have user_duration, reference_duration, and difference."""
        user = self._make_user_phases()
        reference = self._make_reference_phases()

        result = comparison_builder.build_comparison(user, reference)

        for pc in result["phase_comparisons"]:
            assert "phase" in pc
            assert "user_duration_ms" in pc
            assert "reference_duration_ms" in pc
            assert "difference_ms" in pc
            assert "difference_percent" in pc
            assert "is_significant" in pc

    def test_comparison_has_total_durations(self, comparison_builder: ComparisonViewBuilder):
        """Comparison should include total duration for both user and reference."""
        user = self._make_user_phases()
        reference = self._make_reference_phases()

        result = comparison_builder.build_comparison(user, reference)

        assert "user_total_duration_ms" in result
        assert "reference_total_duration_ms" in result
        assert "total_difference_ms" in result

    def test_comparison_has_significant_differences(self, comparison_builder: ComparisonViewBuilder):
        """Comparison should highlight significant differences (>20%)."""
        user = self._make_user_phases()
        reference = self._make_reference_phases()

        result = comparison_builder.build_comparison(user, reference)

        assert "significant_differences" in result
        # Stride: user 333.3 vs ref 250.0 = +33.3% → significant
        # Impact: user 83.3 vs ref 50.0 = +66.6% → significant
        assert len(result["significant_differences"]) >= 1

    def test_difference_ms_is_user_minus_reference(self, comparison_builder: ComparisonViewBuilder):
        """difference_ms should be user_duration - reference_duration."""
        user = self._make_user_phases()
        reference = self._make_reference_phases()

        result = comparison_builder.build_comparison(user, reference)

        for pc in result["phase_comparisons"]:
            if pc["user_duration_ms"] is not None and pc["reference_duration_ms"] is not None:
                expected_diff = pc["user_duration_ms"] - pc["reference_duration_ms"]
                assert pc["difference_ms"] == pytest.approx(expected_diff, abs=0.01)

    def test_empty_user_phases(self, comparison_builder: ComparisonViewBuilder):
        """Should handle empty user phases gracefully."""
        user = SwingPhaseResult()
        reference = self._make_reference_phases()

        result = comparison_builder.build_comparison(user, reference)

        assert "phase_comparisons" in result
        assert result["user_total_duration_ms"] == 0.0

    def test_identical_phases_show_no_difference(self, comparison_builder: ComparisonViewBuilder):
        """Identical phases should show zero difference."""
        user = self._make_user_phases()

        result = comparison_builder.build_comparison(user, user)

        for pc in result["phase_comparisons"]:
            if pc["difference_ms"] is not None:
                assert pc["difference_ms"] == pytest.approx(0.0, abs=0.01)
                assert pc["is_significant"] is False
