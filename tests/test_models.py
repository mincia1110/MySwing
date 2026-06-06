"""Unit tests for core data models and enums."""

import pytest

from app.models.enums import BattingDirection, MetricRating, QualityStatus, SwingPhase
from app.models.video import QualityCheckResult, VideoMetadata, VideoValidationResult
from app.models.pose import Keypoint, MultiPoseResult, PoseResult
from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.swing import PhaseAnomaly, SwingPhaseResult, TransitionBoundary
from app.models.biomechanics import (
    LaunchAngleResult,
    BatSpeedResult,
    BiomechanicsResult,
    JointAngularVelocity,
    KinematicChainResult,
    LaunchAngleResult,
    RotationResult,
    UnmeasurableMetric,
)
from app.models.evaluation import (
    DrillRecommendation,
    ImprovementArea,
    MetricEvaluation,
    WeightTransferResult,
)
from app.models.report import MetricDataPoint, TrendData
from app.models.user_profile import UserProfile


class TestSwingPhaseEnum:
    """Tests for SwingPhase enum (Requirement 5.1)."""

    def test_has_six_phases(self):
        assert len(SwingPhase) == 6

    def test_phase_values(self):
        assert SwingPhase.STANCE.value == "stance"
        assert SwingPhase.LOAD.value == "load"
        assert SwingPhase.STRIDE.value == "stride"
        assert SwingPhase.ROTATION.value == "rotation"
        assert SwingPhase.IMPACT.value == "impact"
        assert SwingPhase.FOLLOW_THROUGH.value == "follow_through"

    def test_phase_from_value(self):
        assert SwingPhase("stance") == SwingPhase.STANCE
        assert SwingPhase("follow_through") == SwingPhase.FOLLOW_THROUGH


class TestMetricRatingEnum:
    def test_has_three_ratings(self):
        assert len(MetricRating) == 3

    def test_rating_values(self):
        assert MetricRating.BELOW_RANGE.value == "below_range"
        assert MetricRating.WITHIN_RANGE.value == "within_range"
        assert MetricRating.ABOVE_RANGE.value == "above_range"


class TestQualityStatusEnum:
    def test_has_two_statuses(self):
        assert len(QualityStatus) == 2

    def test_status_values(self):
        assert QualityStatus.PASS.value == "pass"
        assert QualityStatus.WARNING.value == "warning"


class TestBattingDirectionEnum:
    """Tests for BattingDirection enum (Requirement 2.1)."""

    def test_has_two_directions(self):
        assert len(BattingDirection) == 2

    def test_direction_values(self):
        assert BattingDirection.LEFT.value == "left"
        assert BattingDirection.RIGHT.value == "right"

    def test_direction_from_value(self):
        assert BattingDirection("left") == BattingDirection.LEFT
        assert BattingDirection("right") == BattingDirection.RIGHT


class TestVideoMetadata:
    def test_creation(self):
        meta = VideoMetadata(
            file_key="uploads/abc123.mp4",
            file_name="swing.mp4",
            file_size_bytes=50_000_000,
            duration_seconds=3.5,
            resolution_width=1920,
            resolution_height=1080,
            frame_rate=60.0,
            codec="h264",
            format="mp4",
        )
        assert meta.file_key == "uploads/abc123.mp4"
        assert meta.resolution_width == 1920
        assert meta.format == "mp4"


class TestVideoValidationResult:
    def test_valid_result(self):
        result = VideoValidationResult(
            is_valid=True,
            format_ok=True,
            size_ok=True,
            resolution_ok=True,
            frame_rate_ok=True,
        )
        assert result.is_valid is True
        assert result.errors == []

    def test_invalid_result_with_errors(self):
        result = VideoValidationResult(
            is_valid=False,
            format_ok=False,
            size_ok=True,
            resolution_ok=True,
            frame_rate_ok=True,
            errors=["Unsupported format: wmv"],
        )
        assert result.is_valid is False
        assert len(result.errors) == 1


class TestQualityCheckResult:
    def test_all_pass(self):
        result = QualityCheckResult(
            brightness_status=QualityStatus.PASS,
            framing_status=QualityStatus.PASS,
            resolution_status=QualityStatus.PASS,
            frame_rate_stability_status=QualityStatus.PASS,
            brightness_value=60.0,
            swing_arc_visibility_percent=95.0,
            frame_rate_variation_percent=3.0,
        )
        assert result.brightness_status == QualityStatus.PASS
        assert result.warnings == []

    def test_with_warnings(self):
        result = QualityCheckResult(
            brightness_status=QualityStatus.WARNING,
            framing_status=QualityStatus.PASS,
            resolution_status=QualityStatus.PASS,
            frame_rate_stability_status=QualityStatus.PASS,
            brightness_value=30.0,
            swing_arc_visibility_percent=90.0,
            frame_rate_variation_percent=5.0,
            warnings=["Insufficient lighting"],
        )
        assert result.brightness_status == QualityStatus.WARNING
        assert len(result.warnings) == 1


class TestKeypoint:
    def test_creation(self):
        kp = Keypoint(x=0.5, y=0.3, z=0.1, confidence=0.95, name="left_hip")
        assert kp.x == 0.5
        assert kp.name == "left_hip"
        assert kp.confidence == 0.95


class TestPoseResult:
    def test_creation(self):
        keypoints = [Keypoint(x=0.5, y=0.3, z=0.1, confidence=0.9, name="nose")]
        result = PoseResult(
            frame_index=0,
            keypoints=keypoints,
            person_id=1,
            is_primary_batter=True,
            overall_confidence=0.9,
            is_low_confidence=False,
        )
        assert result.frame_index == 0
        assert result.is_primary_batter is True
        assert len(result.keypoints) == 1


class TestBatDetectionResult:
    def test_detected(self):
        result = BatDetectionResult(
            frame_index=10,
            detected=True,
            position=(640.0, 360.0),
            orientation_angle=45.0,
            length_pixels=200.0,
            confidence=0.95,
            is_predicted=False,
        )
        assert result.detected is True
        assert result.orientation_angle == 45.0
        assert result.position == (640.0, 360.0)

    def test_predicted(self):
        result = BatDetectionResult(
            frame_index=15,
            detected=False,
            position=(650.0, 365.0),
            orientation_angle=50.0,
            length_pixels=200.0,
            confidence=0.7,
            is_predicted=True,
        )
        assert result.is_predicted is True


class TestSwingPhaseResult:
    def test_creation_with_phases(self):
        result = SwingPhaseResult(
            phases={
                SwingPhase.STANCE: (0, 30),
                SwingPhase.LOAD: (31, 50),
            },
            phase_durations_ms={
                SwingPhase.STANCE: 500.0,
                SwingPhase.LOAD: 333.0,
            },
        )
        assert SwingPhase.STANCE in result.phases
        assert result.phase_durations_ms[SwingPhase.STANCE] == 500.0

    def test_anomaly(self):
        anomaly = PhaseAnomaly(
            phase=SwingPhase.STRIDE,
            anomaly_type="abnormally_short",
            duration_ms=30.0,
        )
        assert anomaly.anomaly_type == "abnormally_short"
        assert anomaly.duration_ms == 30.0


class TestBiomechanicsResult:
    def test_creation(self):
        result = BiomechanicsResult(
            bat_speed=BatSpeedResult(speed_kmh=120.0, precision=1.0, measurement_frame=100),
            kinematic_chain=KinematicChainResult(),
            rotation=RotationResult(
                hip_rotation_speed_dps=800.0,
                shoulder_rotation_speed_dps=1200.0,
                hip_shoulder_separation_degrees=45.0,
                rotation_phase_start_frame=60,
                rotation_phase_end_frame=100,
            ),
            hand_path_efficiency=0.85,
            attack_angle=LaunchAngleResult(
                angle_degrees=10.0,
                precision=0.5,
                impact_frame=100,
            ),
        )
        assert result.bat_speed.speed_kmh == 120.0
        assert result.hand_path_efficiency == 0.85
        assert result.unmeasurable_metrics == []

    def test_with_unmeasurable_metrics(self):
        result = BiomechanicsResult(
            bat_speed=BatSpeedResult(speed_kmh=0.0, precision=1.0, measurement_frame=0),
            kinematic_chain=KinematicChainResult(),
            rotation=RotationResult(
                hip_rotation_speed_dps=0.0,
                shoulder_rotation_speed_dps=0.0,
                hip_shoulder_separation_degrees=0.0,
                rotation_phase_start_frame=0,
                rotation_phase_end_frame=0,
            ),
            hand_path_efficiency=0.0,
            attack_angle=LaunchAngleResult(
                angle_degrees=0.0, precision=0.5, impact_frame=0
            ),
            unmeasurable_metrics=[
                UnmeasurableMetric(metric_name="bat_speed", reason="Impact not detected"),
            ],
        )
        assert len(result.unmeasurable_metrics) == 1
        assert result.unmeasurable_metrics[0].reason == "Impact not detected"


class TestUserProfile:
    def test_required_fields_only(self):
        profile = UserProfile(
            height=175.0,
            bat_length=33.0,
            batting_direction=BattingDirection.RIGHT,
        )
        assert profile.height == 175.0
        assert profile.bat_length == 33.0
        assert profile.batting_direction == BattingDirection.RIGHT
        assert profile.weight is None
        assert profile.level is None

    def test_all_fields(self):
        profile = UserProfile(
            height=180.0,
            bat_length=34.0,
            batting_direction=BattingDirection.LEFT,
            weight=80.0,
            camera_direction="side",
            age_group="adult",
            level="college",
            bat_weight=30.0,
        )
        assert profile.batting_direction == BattingDirection.LEFT
        assert profile.weight == 80.0
        assert profile.camera_direction == "side"
        assert profile.level == "college"
        assert profile.bat_weight == 30.0


class TestMetricEvaluation:
    def test_within_range(self):
        evaluation = MetricEvaluation(
            metric_name="bat_speed",
            measured_value=120.0,
            unit="km/h",
            reference_min=110.0,
            reference_max=130.0,
            deviation_percent=0.0,
            rating=MetricRating.WITHIN_RANGE,
            color_code="green",
        )
        assert evaluation.rating == MetricRating.WITHIN_RANGE
        assert evaluation.color_code == "green"

    def test_below_range(self):
        evaluation = MetricEvaluation(
            metric_name="bat_speed",
            measured_value=95.0,
            unit="km/h",
            reference_min=110.0,
            reference_max=130.0,
            deviation_percent=13.6,
            rating=MetricRating.BELOW_RANGE,
            color_code="red",
        )
        assert evaluation.rating == MetricRating.BELOW_RANGE


class TestImprovementArea:
    def test_creation(self):
        area = ImprovementArea(
            metric_name="bat_speed",
            deviation_percent=15.0,
            current_value=95.0,
            target_range=(110.0, 130.0),
            rank=1,
        )
        assert area.rank == 1
        assert area.target_range == (110.0, 130.0)


class TestDrillRecommendation:
    def test_creation(self):
        drill = DrillRecommendation(
            drill_name="Tee Work - High Tee",
            target_metric="attack_angle",
            description="Practice hitting off a high tee to improve launch angle.",
        )
        assert drill.target_metric == "attack_angle"


class TestDomainModuleImports:
    """Tests that the domain module correctly re-exports all models."""

    def test_import_all_enums(self):
        from app.models.domain import (
            BattingDirection,
            MetricRating,
            QualityStatus,
            SwingPhase,
        )
        assert SwingPhase.STANCE.value == "stance"
        assert MetricRating.WITHIN_RANGE.value == "within_range"
        assert QualityStatus.PASS.value == "pass"
        assert BattingDirection.LEFT.value == "left"

    def test_import_all_video_models(self):
        from app.models.domain import (
            QualityCheckResult,
            VideoMetadata,
            VideoValidationResult,
        )
        assert VideoMetadata is not None
        assert VideoValidationResult is not None
        assert QualityCheckResult is not None

    def test_import_all_pose_models(self):
        from app.models.domain import Keypoint, MultiPoseResult, PoseResult
        assert Keypoint is not None
        assert PoseResult is not None
        assert MultiPoseResult is not None

    def test_import_all_bat_models(self):
        from app.models.domain import BatDetectionResult, BatTrajectory
        assert BatDetectionResult is not None
        assert BatTrajectory is not None

    def test_import_all_swing_models(self):
        from app.models.domain import PhaseAnomaly, SwingPhaseResult, TransitionBoundary
        assert TransitionBoundary is not None
        assert SwingPhaseResult is not None
        assert PhaseAnomaly is not None

    def test_import_all_biomechanics_models(self):
        from app.models.domain import (
            LaunchAngleResult,
            BatSpeedResult,
            BiomechanicsResult,
            JointAngularVelocity,
            KinematicChainResult,
            LaunchAngleResult,
            RotationResult,
            UnmeasurableMetric,
        )
        assert BatSpeedResult is not None
        assert BiomechanicsResult is not None

    def test_import_all_evaluation_models(self):
        from app.models.domain import (
            DrillRecommendation,
            ImprovementArea,
            MetricEvaluation,
            WeightTransferResult,
        )
        assert MetricEvaluation is not None
        assert ImprovementArea is not None

    def test_import_all_report_models(self):
        from app.models.domain import AnalysisReport, MetricDataPoint, TrendData
        assert AnalysisReport is not None
        assert TrendData is not None
        assert MetricDataPoint is not None

    def test_import_user_profile(self):
        from app.models.domain import UserProfile
        assert UserProfile is not None
