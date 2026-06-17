"""Unit tests for pixel calibration and bat speed calculation (Task 9.1).

Tests PixelCalibrator and BatSpeedCalculator classes covering:
- Pixel-to-meter ratio calculation
- Bat length verification (within 15%, exceeding 15%)
- Bat speed calculation at various speeds
- Bat speed unit conversion (m/s to km/h)
- Graceful handling of missing keypoints
"""

import math

import pytest

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.pose import Keypoint, PoseResult
from app.pipeline.biomechanics_analyzer import (
    BAT_LENGTH_DISCREPANCY_THRESHOLD,
    BAT_SPEED_PRECISION_KMH,
    BatSpeedCalculator,
    BiomechanicsOrchestrator,
    CalibrationError,
    PixelCalibrator,
)

# --- Helper functions ---


def _make_keypoint(
    name: str,
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
    confidence: float = 0.9,
) -> Keypoint:
    """Create a Keypoint with given parameters."""
    return Keypoint(x=x, y=y, z=z, confidence=confidence, name=name)


def _make_pose_result(
    keypoints: list[Keypoint],
    frame_index: int = 0,
) -> PoseResult:
    """Create a PoseResult with given keypoints."""
    return PoseResult(
        frame_index=frame_index,
        keypoints=keypoints,
        person_id=0,
        is_primary_batter=True,
        overall_confidence=0.9,
        is_low_confidence=False,
    )


def _make_detection(
    frame_index: int,
    position: tuple[float, float] = (100.0, 100.0),
    detected: bool = True,
    length_pixels: float = 150.0,
    confidence: float = 0.95,
    coordinate_space: str = "pixel",
    bat_head_position: tuple[float, float] | None = None,
) -> BatDetectionResult:
    """Create a BatDetectionResult with given parameters."""
    return BatDetectionResult(
        frame_index=frame_index,
        detected=detected,
        position=position,
        orientation_angle=45.0,
        length_pixels=length_pixels,
        confidence=confidence,
        is_predicted=False,
        coordinate_space=coordinate_space,
        bat_head_position=bat_head_position,
    )


def _make_standard_pose(
    head_y: float = 0.1,
    left_ankle_y: float = 0.9,
    right_ankle_y: float = 0.9,
) -> PoseResult:
    """Create a standard pose with head at top and ankles at bottom.

    Default: head at y=0.1, ankles at y=0.9 → distance = 0.8 normalized.
    """
    keypoints = [
        _make_keypoint("nose", x=0.5, y=head_y),
        _make_keypoint("left_ankle", x=0.48, y=left_ankle_y),
        _make_keypoint("right_ankle", x=0.52, y=right_ankle_y),
    ]
    return _make_pose_result(keypoints)


# --- PixelCalibrator Tests ---


class TestPixelCalibratorCalibrate:
    """Test PixelCalibrator.calibrate() method."""

    def test_basic_calibration(self):
        """Basic pixel-to-meter ratio calculation with standard pose."""
        calibrator = PixelCalibrator()
        # Head at y=0.1, ankles at y=0.9 → distance ≈ 0.8 (normalized)
        pose = _make_standard_pose(head_y=0.1, left_ankle_y=0.9, right_ankle_y=0.9)
        user_height_cm = 180.0  # 1.8 meters

        ratio = calibrator.calibrate(pose, user_height_cm)

        # Expected: 1.8m / 0.8 normalized pixels = 2.25 m/normalized_pixel
        expected_distance = math.sqrt((0.5 - 0.5) ** 2 + (0.1 - 0.9) ** 2)
        expected_ratio = 1.8 / expected_distance
        assert abs(ratio - expected_ratio) < 0.001

    def test_calibration_with_different_heights(self):
        """Ratio scales linearly with user height."""
        calibrator = PixelCalibrator()
        pose = _make_standard_pose()

        ratio_170 = calibrator.calibrate(pose, 170.0)
        ratio_180 = calibrator.calibrate(pose, 180.0)

        # Ratio should be proportional to height
        assert abs(ratio_180 / ratio_170 - 180.0 / 170.0) < 0.001

    def test_calibration_with_single_left_ankle(self):
        """Calibration works with only left ankle available."""
        calibrator = PixelCalibrator()
        keypoints = [
            _make_keypoint("nose", x=0.5, y=0.1),
            _make_keypoint("left_ankle", x=0.5, y=0.9),
            _make_keypoint("right_ankle", x=0.5, y=0.9, confidence=0.1),  # low confidence
        ]
        pose = _make_pose_result(keypoints)

        ratio = calibrator.calibrate(pose, 180.0)

        expected_distance = 0.8  # |0.1 - 0.9|
        expected_ratio = 1.8 / expected_distance
        assert abs(ratio - expected_ratio) < 0.001

    def test_calibration_with_single_right_ankle(self):
        """Calibration works with only right ankle available."""
        calibrator = PixelCalibrator()
        keypoints = [
            _make_keypoint("nose", x=0.5, y=0.1),
            _make_keypoint("left_ankle", x=0.5, y=0.9, confidence=0.2),  # low confidence
            _make_keypoint("right_ankle", x=0.5, y=0.9),
        ]
        pose = _make_pose_result(keypoints)

        ratio = calibrator.calibrate(pose, 180.0)

        expected_distance = 0.8
        expected_ratio = 1.8 / expected_distance
        assert abs(ratio - expected_ratio) < 0.001

    def test_calibration_uses_ankle_midpoint(self):
        """When both ankles available, uses their midpoint."""
        calibrator = PixelCalibrator()
        keypoints = [
            _make_keypoint("nose", x=0.5, y=0.1),
            _make_keypoint("left_ankle", x=0.4, y=0.9),
            _make_keypoint("right_ankle", x=0.6, y=0.9),
        ]
        pose = _make_pose_result(keypoints)

        ratio = calibrator.calibrate(pose, 180.0)

        # Midpoint of ankles: (0.5, 0.9)
        # Distance from (0.5, 0.1) to (0.5, 0.9) = 0.8
        expected_ratio = 1.8 / 0.8
        assert abs(ratio - expected_ratio) < 0.001


class TestPixelCalibratorMissingKeypoints:
    """Test PixelCalibrator graceful handling of missing keypoints."""

    def test_missing_head_keypoint(self):
        """Raises CalibrationError when head keypoint is missing."""
        calibrator = PixelCalibrator()
        keypoints = [
            _make_keypoint("left_ankle", x=0.5, y=0.9),
            _make_keypoint("right_ankle", x=0.5, y=0.9),
        ]
        pose = _make_pose_result(keypoints)

        with pytest.raises(CalibrationError, match="Head keypoint"):
            calibrator.calibrate(pose, 180.0)

    def test_missing_both_ankles(self):
        """Raises CalibrationError when both ankle keypoints are missing."""
        calibrator = PixelCalibrator()
        keypoints = [
            _make_keypoint("nose", x=0.5, y=0.1),
        ]
        pose = _make_pose_result(keypoints)

        with pytest.raises(CalibrationError, match="ankle"):
            calibrator.calibrate(pose, 180.0)

    def test_low_confidence_head(self):
        """Raises CalibrationError when head has low confidence."""
        calibrator = PixelCalibrator()
        keypoints = [
            _make_keypoint("nose", x=0.5, y=0.1, confidence=0.3),
            _make_keypoint("left_ankle", x=0.5, y=0.9),
            _make_keypoint("right_ankle", x=0.5, y=0.9),
        ]
        pose = _make_pose_result(keypoints)

        with pytest.raises(CalibrationError, match="Head keypoint"):
            calibrator.calibrate(pose, 180.0)

    def test_low_confidence_both_ankles(self):
        """Raises CalibrationError when both ankles have low confidence."""
        calibrator = PixelCalibrator()
        keypoints = [
            _make_keypoint("nose", x=0.5, y=0.1),
            _make_keypoint("left_ankle", x=0.5, y=0.9, confidence=0.2),
            _make_keypoint("right_ankle", x=0.5, y=0.9, confidence=0.3),
        ]
        pose = _make_pose_result(keypoints)

        with pytest.raises(CalibrationError, match="ankle"):
            calibrator.calibrate(pose, 180.0)

    def test_zero_height_raises_error(self):
        """Raises CalibrationError when user height is zero."""
        calibrator = PixelCalibrator()
        pose = _make_standard_pose()

        with pytest.raises(CalibrationError, match="height must be positive"):
            calibrator.calibrate(pose, 0.0)

    def test_negative_height_raises_error(self):
        """Raises CalibrationError when user height is negative."""
        calibrator = PixelCalibrator()
        pose = _make_standard_pose()

        with pytest.raises(CalibrationError, match="height must be positive"):
            calibrator.calibrate(pose, -170.0)

    def test_empty_keypoints(self):
        """Raises CalibrationError when keypoints list is empty."""
        calibrator = PixelCalibrator()
        pose = _make_pose_result(keypoints=[])

        with pytest.raises(CalibrationError):
            calibrator.calibrate(pose, 180.0)


class TestPixelCalibratorVerifyWithBat:
    """Test PixelCalibrator.verify_with_bat() method."""

    def test_valid_calibration_within_threshold(self):
        """Verification passes when discrepancy is within 15%."""
        calibrator = PixelCalibrator()
        # Bat detected at 150 pixels, pixel_to_meter = 0.005 m/px
        # Detected length in meters: 150 * 0.005 = 0.75m
        # Actual bat length: 0.76m
        # Discrepancy: |0.75 - 0.76| / 0.76 ≈ 1.3%
        bat_detection = _make_detection(0, length_pixels=150.0)
        pixel_to_meter = 0.005
        bat_length_actual = 0.76  # meters

        is_valid, discrepancy = calibrator.verify_with_bat(
            bat_detection, bat_length_actual, pixel_to_meter
        )

        assert is_valid is True
        assert discrepancy < BAT_LENGTH_DISCREPANCY_THRESHOLD

    def test_invalid_calibration_exceeds_threshold(self):
        """Verification fails when discrepancy exceeds 15%."""
        calibrator = PixelCalibrator()
        # Bat detected at 150 pixels, pixel_to_meter = 0.005 m/px
        # Detected length in meters: 150 * 0.005 = 0.75m
        # Actual bat length: 0.60m
        # Discrepancy: |0.75 - 0.60| / 0.60 = 25%
        bat_detection = _make_detection(0, length_pixels=150.0)
        pixel_to_meter = 0.005
        bat_length_actual = 0.60  # meters

        is_valid, discrepancy = calibrator.verify_with_bat(
            bat_detection, bat_length_actual, pixel_to_meter
        )

        assert is_valid is False
        assert discrepancy > BAT_LENGTH_DISCREPANCY_THRESHOLD

    def test_exact_match(self):
        """Verification passes with zero discrepancy."""
        calibrator = PixelCalibrator()
        # Bat detected at 150 pixels, pixel_to_meter = 0.005 m/px
        # Detected length: 0.75m, Actual: 0.75m
        bat_detection = _make_detection(0, length_pixels=150.0)
        pixel_to_meter = 0.005
        bat_length_actual = 0.75  # meters

        is_valid, discrepancy = calibrator.verify_with_bat(
            bat_detection, bat_length_actual, pixel_to_meter
        )

        assert is_valid is True
        assert abs(discrepancy) < 0.001

    def test_at_threshold_boundary(self):
        """Verification passes at exactly 15% discrepancy."""
        calibrator = PixelCalibrator()
        # Set up exactly 15% discrepancy
        bat_length_actual = 0.80  # meters
        # We want detected = actual * 1.15 = 0.92m
        # detected_pixels * pixel_to_meter = 0.92
        # 184 * 0.005 = 0.92
        bat_detection = _make_detection(0, length_pixels=184.0)
        pixel_to_meter = 0.005

        is_valid, discrepancy = calibrator.verify_with_bat(
            bat_detection, bat_length_actual, pixel_to_meter
        )

        assert is_valid is True
        assert abs(discrepancy - 0.15) < 0.001

    def test_bat_not_detected_raises_error(self):
        """Raises CalibrationError when bat is not detected."""
        calibrator = PixelCalibrator()
        bat_detection = _make_detection(0, detected=False)

        with pytest.raises(CalibrationError, match="not detected"):
            calibrator.verify_with_bat(bat_detection, 0.75, 0.005)

    def test_zero_bat_length_raises_error(self):
        """Raises CalibrationError when actual bat length is zero."""
        calibrator = PixelCalibrator()
        bat_detection = _make_detection(0, length_pixels=150.0)

        with pytest.raises(CalibrationError, match="bat length must be positive"):
            calibrator.verify_with_bat(bat_detection, 0.0, 0.005)

    def test_zero_pixel_to_meter_raises_error(self):
        """Raises CalibrationError when pixel_to_meter is zero."""
        calibrator = PixelCalibrator()
        bat_detection = _make_detection(0, length_pixels=150.0)

        with pytest.raises(CalibrationError, match="Pixel-to-meter ratio must be positive"):
            calibrator.verify_with_bat(bat_detection, 0.75, 0.0)

    def test_zero_length_pixels_raises_error(self):
        """Raises CalibrationError when detected bat length in pixels is zero."""
        calibrator = PixelCalibrator()
        bat_detection = _make_detection(0, length_pixels=0.0)

        with pytest.raises(CalibrationError, match="zero or negative"):
            calibrator.verify_with_bat(bat_detection, 0.75, 0.005)

    def test_non_pixel_coordinate_space_raises_error(self):
        """Normalized-coordinate bat detections are not valid for pixel-length verification."""
        calibrator = PixelCalibrator()
        bat_detection = _make_detection(0, length_pixels=0.25, coordinate_space="normalized")

        with pytest.raises(CalibrationError, match="requires pixel coordinate space"):
            calibrator.verify_with_bat(bat_detection, 0.75, 0.005)


class TestImpactMetricFrameSelection:
    """Test robust metric-frame selection within the impact zone."""

    def test_prefers_high_speed_plausible_angle_candidate_near_impact(self):
        """Metric frame can differ from phase midpoint when local motion is stronger."""
        orchestrator = BiomechanicsOrchestrator()
        positions = {
            52: (0.0, 0.0),
            53: (10.0, 0.0),
            54: (60.0, -10.0),  # high speed, plausible upward angle
            55: (110.0, -20.0),
            56: (120.0, -20.0),
            62: (130.0, -20.0),
            63: (135.0, -20.0),
            64: (140.0, -20.0),  # phase midpoint, but lower local speed
            65: (145.0, -20.0),
            66: (150.0, -20.0),
        }
        trajectory = BatTrajectory(
            detections=[_make_detection(frame, position=pos) for frame, pos in positions.items()]
        )

        frame = orchestrator._select_impact_metric_frame(
            trajectory,
            impact_frame=64,
            pixel_to_meter=1.0,
            fps=1.0,
            video_width=1,
            video_height=1,
        )

        assert frame in {54, 55}
        assert frame != 64


class TestBatSpeedCalculator:
    """Test BatSpeedCalculator.calculate_bat_speed() method."""

    def test_basic_speed_calculation(self):
        """Basic bat speed calculation with known displacement."""
        calculator = BatSpeedCalculator()
        # 5 frames around impact (frames 8-12), impact at frame 10
        # Bat moves 50 pixels per frame horizontally
        detections = [
            _make_detection(8, position=(100.0, 200.0)),
            _make_detection(9, position=(150.0, 200.0)),
            _make_detection(10, position=(200.0, 200.0)),
            _make_detection(11, position=(250.0, 200.0)),
            _make_detection(12, position=(300.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        # pixel_to_meter = 0.005 m/px, fps = 120
        # displacement per frame = 50 px/frame
        # speed_m_s = 50 * 0.005 * 120 = 30 m/s
        # speed_kmh = 30 * 3.6 = 108 km/h
        result = calculator.calculate_bat_speed(
            trajectory, impact_frame=10, pixel_to_meter=0.005, fps=120.0
        )

        assert abs(result.speed_kmh - 108.0) < BAT_SPEED_PRECISION_KMH
        assert result.precision == BAT_SPEED_PRECISION_KMH
        assert result.measurement_frame == 10

    def test_speed_prefers_bat_head_position_when_available(self):
        """Speed should use bat_head_position over center position when provided."""
        calculator = BatSpeedCalculator()
        # Center is static, but bat head moves 10 px/frame -> expected non-zero speed.
        detections = [
            _make_detection(8, position=(100.0, 200.0), bat_head_position=(50.0, 100.0)),
            _make_detection(9, position=(100.0, 200.0), bat_head_position=(60.0, 100.0)),
            _make_detection(10, position=(100.0, 200.0), bat_head_position=(70.0, 100.0)),
            _make_detection(11, position=(100.0, 200.0), bat_head_position=(80.0, 100.0)),
            _make_detection(12, position=(100.0, 200.0), bat_head_position=(90.0, 100.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_bat_speed(
            trajectory, impact_frame=10, pixel_to_meter=0.01, fps=100.0
        )

        # 10 px/frame * 0.01 m/px * 100 fps = 10 m/s = 36 km/h
        assert abs(result.speed_kmh - 36.0) < BAT_SPEED_PRECISION_KMH

    def test_barrel_speed_multiplier_applies_to_proxy_trajectories(self):
        """Optional multiplier should scale proxy bat-head speeds."""
        calculator = BatSpeedCalculator()
        detections = [
            _make_detection(8, position=(100.0, 200.0), bat_head_position=(50.0, 100.0)),
            _make_detection(9, position=(100.0, 200.0), bat_head_position=(60.0, 100.0)),
            _make_detection(10, position=(100.0, 200.0), bat_head_position=(70.0, 100.0)),
            _make_detection(11, position=(100.0, 200.0), bat_head_position=(80.0, 100.0)),
            _make_detection(12, position=(100.0, 200.0), bat_head_position=(90.0, 100.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_bat_speed(
            trajectory,
            impact_frame=10,
            pixel_to_meter=0.01,
            fps=100.0,
            barrel_speed_multiplier=1.7,
        )

        assert abs(result.speed_kmh - 61.2) < BAT_SPEED_PRECISION_KMH

    def test_speed_unit_conversion_m_s_to_kmh(self):
        """Verify correct m/s to km/h conversion (×3.6)."""
        calculator = BatSpeedCalculator()
        # Set up: 1 pixel displacement per frame, 1 m/px, 1 fps
        # speed_m_s = 1 * 1 * 1 = 1 m/s
        # speed_kmh = 1 * 3.6 = 3.6 km/h
        detections = [
            _make_detection(8, position=(0.0, 0.0)),
            _make_detection(9, position=(1.0, 0.0)),
            _make_detection(10, position=(2.0, 0.0)),
            _make_detection(11, position=(3.0, 0.0)),
            _make_detection(12, position=(4.0, 0.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_bat_speed(
            trajectory, impact_frame=10, pixel_to_meter=1.0, fps=1.0
        )

        assert abs(result.speed_kmh - 3.6) < 0.01

    def test_high_speed_swing(self):
        """Test with high-speed swing (typical pro bat speed ~130-150 km/h)."""
        calculator = BatSpeedCalculator()
        # Target: ~140 km/h = 38.89 m/s
        # With fps=240, pixel_to_meter=0.003:
        # displacement_per_frame = 38.89 / (0.003 * 240) = 54.01 px/frame
        detections = [
            _make_detection(8, position=(100.0, 200.0)),
            _make_detection(9, position=(154.0, 200.0)),
            _make_detection(10, position=(208.0, 200.0)),
            _make_detection(11, position=(262.0, 200.0)),
            _make_detection(12, position=(316.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_bat_speed(
            trajectory, impact_frame=10, pixel_to_meter=0.003, fps=240.0
        )

        # Expected: 54 * 0.003 * 240 * 3.6 = 139.97 km/h
        assert abs(result.speed_kmh - 140.0) < BAT_SPEED_PRECISION_KMH

    def test_slow_speed_swing(self):
        """Test with slow swing speed (~60 km/h)."""
        calculator = BatSpeedCalculator()
        # Target: 60 km/h = 16.67 m/s
        # With fps=120, pixel_to_meter=0.005:
        # displacement_per_frame = 16.67 / (0.005 * 120) = 27.78 px/frame
        detections = [
            _make_detection(8, position=(100.0, 200.0)),
            _make_detection(9, position=(127.78, 200.0)),
            _make_detection(10, position=(155.56, 200.0)),
            _make_detection(11, position=(183.34, 200.0)),
            _make_detection(12, position=(211.12, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_bat_speed(
            trajectory, impact_frame=10, pixel_to_meter=0.005, fps=120.0
        )

        assert abs(result.speed_kmh - 60.0) < BAT_SPEED_PRECISION_KMH

    def test_diagonal_movement(self):
        """Speed calculation with diagonal bat movement."""
        calculator = BatSpeedCalculator()
        # Bat moves diagonally: 30px horizontal, 40px vertical per frame
        # Euclidean displacement per frame = sqrt(30^2 + 40^2) = 50 px/frame
        detections = [
            _make_detection(8, position=(100.0, 100.0)),
            _make_detection(9, position=(130.0, 140.0)),
            _make_detection(10, position=(160.0, 180.0)),
            _make_detection(11, position=(190.0, 220.0)),
            _make_detection(12, position=(220.0, 260.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        # speed_m_s = 50 * 0.005 * 120 = 30 m/s
        # speed_kmh = 30 * 3.6 = 108 km/h
        result = calculator.calculate_bat_speed(
            trajectory, impact_frame=10, pixel_to_meter=0.005, fps=120.0
        )

        assert abs(result.speed_kmh - 108.0) < BAT_SPEED_PRECISION_KMH

    def test_partial_window_detections(self):
        """Speed calculation works with fewer detections in window."""
        calculator = BatSpeedCalculator()
        # Only 3 detections in the ±2 window (frames 9, 10, 11)
        detections = [
            _make_detection(5, position=(50.0, 200.0)),
            _make_detection(9, position=(150.0, 200.0)),
            _make_detection(10, position=(200.0, 200.0)),
            _make_detection(11, position=(250.0, 200.0)),
            _make_detection(15, position=(450.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_bat_speed(
            trajectory, impact_frame=10, pixel_to_meter=0.005, fps=120.0
        )

        # Displacement: 50+50 = 100 pixels over 2 frames = 50 px/frame
        # speed_kmh = 50 * 0.005 * 120 * 3.6 = 108 km/h
        assert abs(result.speed_kmh - 108.0) < BAT_SPEED_PRECISION_KMH

    def test_insufficient_detections_raises_error(self):
        """Raises CalibrationError when fewer than 2 detections in window."""
        calculator = BatSpeedCalculator()
        # Only 1 detection in the ±2 window around frame 10
        detections = [
            _make_detection(5, position=(50.0, 200.0)),
            _make_detection(10, position=(200.0, 200.0)),
            _make_detection(15, position=(450.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        with pytest.raises(CalibrationError, match="Insufficient"):
            calculator.calculate_bat_speed(
                trajectory, impact_frame=10, pixel_to_meter=0.005, fps=120.0
            )

    def test_zero_fps_raises_error(self):
        """Raises CalibrationError when fps is zero."""
        calculator = BatSpeedCalculator()
        detections = [
            _make_detection(8, position=(100.0, 200.0)),
            _make_detection(10, position=(200.0, 200.0)),
            _make_detection(12, position=(300.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        with pytest.raises(CalibrationError, match="FPS must be positive"):
            calculator.calculate_bat_speed(
                trajectory, impact_frame=10, pixel_to_meter=0.005, fps=0.0
            )

    def test_zero_pixel_to_meter_raises_error(self):
        """Raises CalibrationError when pixel_to_meter is zero."""
        calculator = BatSpeedCalculator()
        detections = [
            _make_detection(8, position=(100.0, 200.0)),
            _make_detection(10, position=(200.0, 200.0)),
            _make_detection(12, position=(300.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        with pytest.raises(CalibrationError, match="Pixel-to-meter ratio must be positive"):
            calculator.calculate_bat_speed(
                trajectory, impact_frame=10, pixel_to_meter=0.0, fps=120.0
            )

    def test_undetected_frames_in_window_excluded(self):
        """Undetected frames within the window are excluded from calculation."""
        calculator = BatSpeedCalculator()
        # Frame 9 is undetected, so only frames 8, 10, 11, 12 are used
        detections = [
            _make_detection(8, position=(100.0, 200.0)),
            _make_detection(9, position=(150.0, 200.0), detected=False),
            _make_detection(10, position=(200.0, 200.0)),
            _make_detection(11, position=(250.0, 200.0)),
            _make_detection(12, position=(300.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_bat_speed(
            trajectory, impact_frame=10, pixel_to_meter=0.005, fps=120.0
        )

        # Detected frames: 8, 10, 11, 12
        # Displacements: |200-100|=100, |250-200|=50, |300-250|=50 → total=200
        # Frame span: 12-8 = 4
        # displacement_per_frame = 200/4 = 50 px/frame
        # speed_kmh = 50 * 0.005 * 120 * 3.6 = 108 km/h
        assert abs(result.speed_kmh - 108.0) < BAT_SPEED_PRECISION_KMH

    def test_result_precision_field(self):
        """BatSpeedResult has correct precision value."""
        calculator = BatSpeedCalculator()
        detections = [
            _make_detection(8, position=(100.0, 200.0)),
            _make_detection(10, position=(200.0, 200.0)),
            _make_detection(12, position=(300.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_bat_speed(
            trajectory, impact_frame=10, pixel_to_meter=0.005, fps=120.0
        )

        assert result.precision == 1.0  # ±1 km/h

    def test_result_measurement_frame(self):
        """BatSpeedResult records the correct measurement frame."""
        calculator = BatSpeedCalculator()
        detections = [
            _make_detection(18, position=(100.0, 200.0)),
            _make_detection(20, position=(200.0, 200.0)),
            _make_detection(22, position=(300.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_bat_speed(
            trajectory, impact_frame=20, pixel_to_meter=0.005, fps=120.0
        )

        assert result.measurement_frame == 20
