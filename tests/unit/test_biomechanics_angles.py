"""Unit tests for launch angle and attack angle calculation (Task 9.2).

Tests LaunchAngleCalculator and AttackAngleCalculator classes covering:
- Launch angle for horizontal bat movement (0°)
- Launch angle for upward bat movement (positive angle)
- Launch angle for downward bat movement (negative angle)
- Attack angle averaging over hitting zone
- Attack angle with 150ms window calculation
- Handling of missing detections in hitting zone
- Precision values in results
"""

import math

import pytest

from app.models.bat import BatDetectionResult, BatTrajectory
from app.pipeline.biomechanics_analyzer import (
    ANGLE_PRECISION_DEGREES,
    HITTING_ZONE_DURATION_MS,
    AttackAngleCalculator,
    CalibrationError,
    LaunchAngleCalculator,
)


# --- Helper functions ---


def _make_detection(
    frame_index: int,
    position: tuple[float, float] = (100.0, 100.0),
    detected: bool = True,
    length_pixels: float = 150.0,
    confidence: float = 0.95,
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
    )


# --- LaunchAngleCalculator Tests ---


class TestLaunchAngleHorizontal:
    """Test launch angle for horizontal bat movement (0°)."""

    def test_horizontal_movement_right(self):
        """Bat moving purely to the right produces 0° launch angle."""
        calculator = LaunchAngleCalculator()
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(150.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=10)

        assert abs(result.angle_degrees - 0.0) < 0.01

    def test_horizontal_movement_left(self):
        """Bat moving purely to the left produces 180° (or -180°) launch angle."""
        calculator = LaunchAngleCalculator()
        detections = [
            _make_detection(9, position=(200.0, 200.0)),
            _make_detection(10, position=(100.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=10)

        # atan2(0, -100) = π radians = 180°
        assert abs(abs(result.angle_degrees) - 180.0) < 0.01


class TestLaunchAngleUpward:
    """Test launch angle for upward bat movement (positive angle)."""

    def test_upward_45_degrees(self):
        """Bat moving at 45° upward (right and up in world coords)."""
        calculator = LaunchAngleCalculator()
        # In image coords, up means y decreases
        # dx=50, dy=-50 (image) → angle = atan2(-(-50), 50) = atan2(50, 50) = 45°
        detections = [
            _make_detection(9, position=(100.0, 250.0)),
            _make_detection(10, position=(150.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=10)

        assert abs(result.angle_degrees - 45.0) < 0.01

    def test_upward_90_degrees(self):
        """Bat moving straight up produces 90° launch angle."""
        calculator = LaunchAngleCalculator()
        # In image coords, straight up means y decreases, dx=0
        detections = [
            _make_detection(9, position=(100.0, 250.0)),
            _make_detection(10, position=(100.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=10)

        assert abs(result.angle_degrees - 90.0) < 0.01

    def test_upward_small_angle(self):
        """Bat moving slightly upward (e.g., 10° typical launch angle)."""
        calculator = LaunchAngleCalculator()
        # For 10° angle: dx=100, dy=-tan(10°)*100 ≈ -17.63 (image coords)
        target_angle = 10.0
        dx = 100.0
        dy_image = -dx * math.tan(math.radians(target_angle))
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(100.0 + dx, 200.0 + dy_image)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=10)

        assert abs(result.angle_degrees - target_angle) < 0.5

    def test_uses_bat_head_position_when_available(self):
        """Impact attack angle uses barrel/head motion, not bat-center motion."""
        calculator = LaunchAngleCalculator()
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(150.0, 200.0)),
        ]
        detections[0].bat_head_position = (100.0, 250.0)
        detections[1].bat_head_position = (150.0, 200.0)
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=10)

        assert abs(result.angle_degrees - 45.0) < 0.01


class TestLaunchAngleDownward:
    """Test launch angle for downward bat movement (negative angle)."""

    def test_downward_45_degrees(self):
        """Bat moving at 45° downward produces -45° launch angle."""
        calculator = LaunchAngleCalculator()
        # In image coords, down means y increases
        # dx=50, dy=50 (image) → angle = atan2(-50, 50) = -45°
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(150.0, 250.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=10)

        assert abs(result.angle_degrees - (-45.0)) < 0.01

    def test_downward_90_degrees(self):
        """Bat moving straight down produces -90° launch angle."""
        calculator = LaunchAngleCalculator()
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(100.0, 250.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=10)

        assert abs(result.angle_degrees - (-90.0)) < 0.01

    def test_downward_small_angle(self):
        """Bat moving slightly downward (e.g., -5°)."""
        calculator = LaunchAngleCalculator()
        target_angle = -5.0
        dx = 100.0
        # For -5°: dy_image = -dx * tan(-5°) = dx * tan(5°) ≈ 8.75
        dy_image = -dx * math.tan(math.radians(target_angle))
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(100.0 + dx, 200.0 + dy_image)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=10)

        assert abs(result.angle_degrees - target_angle) < 0.5


class TestLaunchAngleErrors:
    """Test LaunchAngleCalculator error handling."""

    def test_missing_impact_frame_detection(self):
        """Raises CalibrationError when no detection at impact frame."""
        calculator = LaunchAngleCalculator()
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            # No detection at frame 10
        ]
        trajectory = BatTrajectory(detections=detections)

        with pytest.raises(CalibrationError, match="No bat detection at impact frame"):
            calculator.calculate_launch_angle(trajectory, impact_frame=10)

    def test_missing_pre_impact_frame_detection(self):
        """Raises CalibrationError when no detection at frame before impact."""
        calculator = LaunchAngleCalculator()
        detections = [
            # No detection at frame 9
            _make_detection(10, position=(150.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        with pytest.raises(CalibrationError, match="No bat detection at frame"):
            calculator.calculate_launch_angle(trajectory, impact_frame=10)

    def test_undetected_impact_frame(self):
        """Raises CalibrationError when impact frame detection has detected=False."""
        calculator = LaunchAngleCalculator()
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(150.0, 200.0), detected=False),
        ]
        trajectory = BatTrajectory(detections=detections)

        with pytest.raises(CalibrationError, match="No bat detection at impact frame"):
            calculator.calculate_launch_angle(trajectory, impact_frame=10)

    def test_no_movement_raises_error(self):
        """Raises CalibrationError when bat doesn't move between frames."""
        calculator = LaunchAngleCalculator()
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(100.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        with pytest.raises(CalibrationError, match="did not move"):
            calculator.calculate_launch_angle(trajectory, impact_frame=10)


class TestLaunchAnglePrecision:
    """Test precision values in LaunchAngleResult."""

    def test_precision_value(self):
        """LaunchAngleResult has correct precision value (±0.5°)."""
        calculator = LaunchAngleCalculator()
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(150.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=10)

        assert result.precision == ANGLE_PRECISION_DEGREES
        assert result.precision == 0.5

    def test_impact_frame_recorded(self):
        """LaunchAngleResult records the correct impact frame."""
        calculator = LaunchAngleCalculator()
        detections = [
            _make_detection(24, position=(100.0, 200.0)),
            _make_detection(25, position=(150.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_launch_angle(trajectory, impact_frame=25)

        assert result.impact_frame == 25


# --- AttackAngleCalculator Tests ---


class TestAttackAngleAveraging:
    """Test attack angle averaging over hitting zone."""

    def test_constant_angle_through_zone(self):
        """When bat moves at constant angle, average equals that angle."""
        calculator = AttackAngleCalculator()
        # Bat moves at 10° upward consistently through the zone
        # fps=120, 150ms = 18 frames, zone: frames 82-100
        target_angle = 10.0
        dx_per_frame = 50.0
        dy_per_frame = -dx_per_frame * math.tan(math.radians(target_angle))

        detections = []
        for i in range(82, 101):
            x = 100.0 + (i - 82) * dx_per_frame
            y = 500.0 + (i - 82) * dy_per_frame
            detections.append(_make_detection(i, position=(x, y)))

        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_attack_angle(trajectory, impact_frame=100, fps=120.0)

        assert abs(result.angle_degrees - target_angle) < 0.5

    def test_varying_angles_averaged(self):
        """Attack angle is the average of all segment angles."""
        calculator = AttackAngleCalculator()
        # fps=120, 150ms = 18 frames, zone: frames 82-100
        # Create 3 segments with different angles:
        # Segment 1 (frames 82-88): 0° (horizontal)
        # Segment 2 (frames 88-94): ~45° upward
        # Segment 3 (frames 94-100): 0° (horizontal)
        detections = []
        # Segment 1: horizontal movement
        for i in range(82, 89):
            detections.append(_make_detection(i, position=(100.0 + (i - 82) * 50.0, 300.0)))
        # Segment 2: 45° upward (dx=50, dy=-50 in image)
        base_x = 100.0 + 6 * 50.0
        base_y = 300.0
        for i in range(89, 95):
            offset = i - 88
            detections.append(
                _make_detection(i, position=(base_x + offset * 50.0, base_y - offset * 50.0))
            )
        # Segment 3: horizontal movement
        base_x2 = base_x + 6 * 50.0
        base_y2 = base_y - 6 * 50.0
        for i in range(95, 101):
            offset = i - 94
            detections.append(
                _make_detection(i, position=(base_x2 + offset * 50.0, base_y2))
            )

        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_attack_angle(trajectory, impact_frame=100, fps=120.0)

        # We have 18 segments total. 6 at 0°, 6 at 45°, 6 at 0°
        # Average = (6*0 + 6*45 + 6*0) / 18 = 15°
        assert abs(result.angle_degrees - 15.0) < 1.0


class TestAttackAngleWindow:
    """Test attack angle 150ms window calculation."""

    def test_window_at_120fps(self):
        """At 120fps, hitting zone is 18 frames before impact."""
        calculator = AttackAngleCalculator()
        # fps=120, 150ms = 18 frames
        # impact_frame=100, zone starts at frame 82
        detections = []
        for i in range(80, 101):
            detections.append(_make_detection(i, position=(100.0 + i * 10.0, 200.0)))

        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_attack_angle(trajectory, impact_frame=100, fps=120.0)

        assert result.hitting_zone_start_frame == 82
        assert result.hitting_zone_end_frame == 100

    def test_window_at_60fps(self):
        """At 60fps, hitting zone is 9 frames before impact."""
        calculator = AttackAngleCalculator()
        # fps=60, 150ms = 9 frames
        # impact_frame=50, zone starts at frame 41
        detections = []
        for i in range(40, 51):
            detections.append(_make_detection(i, position=(100.0 + i * 10.0, 200.0)))

        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_attack_angle(trajectory, impact_frame=50, fps=60.0)
        assert result.hitting_zone_start_frame == 41
        assert result.hitting_zone_end_frame == 50

    def test_window_at_240fps(self):
        """At 240fps, hitting zone is 36 frames before impact."""
        calculator = AttackAngleCalculator()
        # fps=240, 150ms = 36 frames
        # impact_frame=100, zone starts at frame 64
        detections = []
        for i in range(60, 101):
            detections.append(_make_detection(i, position=(100.0 + i * 10.0, 200.0)))

        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_attack_angle(trajectory, impact_frame=100, fps=240.0)
        assert result.hitting_zone_start_frame == 64
        assert result.hitting_zone_end_frame == 100

    def test_only_zone_detections_used(self):
        """Only detections within the hitting zone are used for calculation."""
        calculator = AttackAngleCalculator()
        # fps=120, zone: frames 82-100
        # Add detections outside zone with very different angles
        detections = [
            # Outside zone (before) - steep downward angle
            _make_detection(70, position=(100.0, 100.0)),
            _make_detection(75, position=(200.0, 500.0)),
            # Inside zone - horizontal movement
            _make_detection(85, position=(300.0, 200.0)),
            _make_detection(90, position=(400.0, 200.0)),
            _make_detection(95, position=(500.0, 200.0)),
            _make_detection(100, position=(600.0, 200.0)),
            # Outside zone (after) - should not exist but just in case
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_attack_angle(trajectory, impact_frame=100, fps=120.0)

        # Only zone detections (85, 90, 95, 100) used → all horizontal → 0°
        assert abs(result.angle_degrees - 0.0) < 0.5


class TestAttackAngleMissingDetections:
    """Test attack angle with missing detections in hitting zone."""

    def test_sparse_detections_in_zone(self):
        """Works with sparse but sufficient detections in the zone."""
        calculator = AttackAngleCalculator()
        # fps=120, zone: frames 82-100
        # Only 3 detections in zone (minimum needed is 2)
        detections = [
            _make_detection(85, position=(100.0, 200.0)),
            _make_detection(93, position=(200.0, 200.0)),
            _make_detection(100, position=(300.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_attack_angle(trajectory, impact_frame=100, fps=120.0)

        # All horizontal → 0°
        assert abs(result.angle_degrees - 0.0) < 0.5

    def test_insufficient_detections_raises_error(self):
        """Raises CalibrationError when fewer than 2 detections in zone."""
        calculator = AttackAngleCalculator()
        # fps=120, zone: frames 82-100
        # Only 1 detection in zone
        detections = [
            _make_detection(70, position=(100.0, 200.0)),
            _make_detection(100, position=(300.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        with pytest.raises(CalibrationError, match="Insufficient"):
            calculator.calculate_attack_angle(trajectory, impact_frame=100, fps=120.0)

    def test_undetected_frames_excluded(self):
        """Frames with detected=False are excluded from calculation."""
        calculator = AttackAngleCalculator()
        # fps=120, zone: frames 82-100
        detections = [
            _make_detection(85, position=(100.0, 200.0)),
            _make_detection(90, position=(200.0, 200.0), detected=False),  # excluded
            _make_detection(95, position=(300.0, 200.0)),
            _make_detection(100, position=(400.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_attack_angle(trajectory, impact_frame=100, fps=120.0)

        # Only frames 85, 95, 100 used (all horizontal) → 0°
        assert abs(result.angle_degrees - 0.0) < 0.5

    def test_zero_fps_raises_error(self):
        """Raises CalibrationError when fps is zero."""
        calculator = AttackAngleCalculator()
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(200.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        with pytest.raises(CalibrationError, match="FPS must be positive"):
            calculator.calculate_attack_angle(trajectory, impact_frame=10, fps=0.0)

    def test_negative_fps_raises_error(self):
        """Raises CalibrationError when fps is negative."""
        calculator = AttackAngleCalculator()
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(200.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        with pytest.raises(CalibrationError, match="FPS must be positive"):
            calculator.calculate_attack_angle(trajectory, impact_frame=10, fps=-30.0)


class TestAttackAnglePrecision:
    """Test precision values in LaunchAngleResult."""

    def test_precision_value(self):
        """LaunchAngleResult has correct precision value (±0.5°)."""
        calculator = AttackAngleCalculator()
        detections = [
            _make_detection(9, position=(100.0, 200.0)),
            _make_detection(10, position=(200.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_attack_angle(trajectory, impact_frame=10, fps=120.0)

        assert result.precision == ANGLE_PRECISION_DEGREES
        assert result.precision == 0.5

    def test_hitting_zone_boundaries_recorded(self):
        """LaunchAngleResult records correct hitting zone boundaries."""
        calculator = AttackAngleCalculator()
        # fps=120, 150ms = 18 frames
        detections = [
            _make_detection(85, position=(100.0, 200.0)),
            _make_detection(100, position=(200.0, 200.0)),
        ]
        trajectory = BatTrajectory(detections=detections)

        result = calculator.calculate_attack_angle(trajectory, impact_frame=100, fps=120.0)

        assert result.hitting_zone_start_frame == 82
        assert result.hitting_zone_end_frame == 100
