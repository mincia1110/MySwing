"""Trajectory prediction and tracking failure handling (Requirements 4.4, 4.5).

Provides bat position prediction when detection fails for up to 5 consecutive
frames using linear extrapolation from previous detections. Reports tracking
failure and ceases prediction when the gap exceeds 5 frames.
"""

from __future__ import annotations

from typing import List, Tuple

from app.models.bat import BatDetectionResult, BatTrajectory


# Maximum consecutive missing frames before tracking failure
MAX_PREDICTION_FRAMES = 5


class TrajectoryPredictor:
    """Predicts bat position from previous trajectory when detection fails.

    Uses linear extrapolation from up to 5 previous detected positions to
    estimate the bat's current position. Reports tracking failure and ceases
    prediction when the bat is not detected for more than 5 consecutive frames.
    """

    def __init__(self) -> None:
        self._consecutive_missing: int = 0
        self._failure_start_frame: int | None = None

    @property
    def consecutive_missing(self) -> int:
        """Number of consecutive frames where bat was not detected."""
        return self._consecutive_missing

    def reset(self) -> None:
        """Reset the predictor state (e.g., when bat is re-detected)."""
        self._consecutive_missing = 0
        self._failure_start_frame = None

    def notify_detected(self) -> None:
        """Notify that the bat was successfully detected in the current frame.

        Resets the consecutive missing counter.
        """
        self._consecutive_missing = 0
        self._failure_start_frame = None

    def notify_missing(self, frame_index: int) -> None:
        """Notify that the bat was not detected in the current frame.

        Increments the consecutive missing counter and records the failure
        start frame if this is the first missing frame in the sequence.

        Args:
            frame_index: The frame index where detection failed.
        """
        if self._consecutive_missing == 0:
            self._failure_start_frame = frame_index
        self._consecutive_missing += 1

    def predict_position(
        self,
        previous_detections: List[BatDetectionResult],
        current_frame_index: int,
    ) -> BatDetectionResult | None:
        """Predict bat position from up to 5 previous frames.

        Uses linear extrapolation from previous detected positions to estimate
        the bat's current position. Returns None if prediction should cease
        (more than 5 consecutive missing frames).

        Args:
            previous_detections: List of previous BatDetectionResult objects
                (most recent last). Only detected=True entries are used.
            current_frame_index: The frame index to predict for.

        Returns:
            A BatDetectionResult with is_predicted=True and reduced confidence,
            or None if prediction should cease (tracking failure).
        """
        if self._should_cease_prediction(self._consecutive_missing):
            return None

        # Filter to only actual detections (not predicted ones ideally, but
        # we use all detected=True entries for extrapolation)
        valid_detections = [d for d in previous_detections if d.detected]

        if not valid_detections:
            return None

        # Use up to 5 most recent detections for extrapolation
        recent = valid_detections[-MAX_PREDICTION_FRAMES:]

        # Extract positions and orientations
        positions = [d.position for d in recent]
        orientations = [d.orientation_angle for d in recent]

        # Predict position via linear extrapolation
        predicted_position = self._linear_extrapolation(positions)

        # Predict orientation
        predicted_orientation = self._extrapolate_orientation(orientations)

        # Use the last known length
        last_length = recent[-1].length_pixels

        # Reduce confidence proportionally to gap length
        base_confidence = recent[-1].confidence
        confidence = self._calculate_reduced_confidence(
            base_confidence, self._consecutive_missing
        )

        return BatDetectionResult(
            frame_index=current_frame_index,
            detected=True,
            position=predicted_position,
            orientation_angle=predicted_orientation,
            length_pixels=last_length,
            confidence=confidence,
            is_predicted=True,
        )

    def _linear_extrapolation(
        self, positions: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Linear extrapolation from recent positions.

        If only one position is available, returns that position (constant
        velocity assumption with zero velocity). If multiple positions are
        available, computes the average velocity and extrapolates one step.

        Args:
            positions: List of (x, y) positions, most recent last.

        Returns:
            Predicted (x, y) position.
        """
        if len(positions) == 1:
            return positions[0]

        # Compute average velocity from available positions
        n = len(positions)
        total_dx = positions[-1][0] - positions[0][0]
        total_dy = positions[-1][1] - positions[0][1]

        # Average velocity per frame gap
        num_gaps = n - 1
        avg_vx = total_dx / num_gaps
        avg_vy = total_dy / num_gaps

        # Extrapolate from last known position
        last_x, last_y = positions[-1]
        predicted_x = last_x + avg_vx
        predicted_y = last_y + avg_vy

        return (predicted_x, predicted_y)

    def _should_cease_prediction(self, consecutive_missing: int) -> bool:
        """Determine if prediction should cease.

        Returns True if more than 5 consecutive frames are missing,
        indicating a tracking failure.

        Args:
            consecutive_missing: Number of consecutive frames without detection.

        Returns:
            True if prediction should cease (consecutive_missing > 5).
        """
        return consecutive_missing > MAX_PREDICTION_FRAMES

    def _extrapolate_orientation(self, orientations: List[float]) -> float:
        """Predict orientation angle from previous orientations.

        Uses linear extrapolation of the orientation angle. Handles the
        wraparound at 360 degrees by using angular differences.

        Args:
            orientations: List of orientation angles in degrees (0-360),
                most recent last.

        Returns:
            Predicted orientation angle in degrees (0-360).
        """
        if len(orientations) == 1:
            return orientations[0]

        # Compute angular velocity using differences
        # Handle wraparound by using shortest angular path
        n = len(orientations)
        total_angular_change = 0.0

        for i in range(1, n):
            diff = orientations[i] - orientations[i - 1]
            # Normalize to [-180, 180] for shortest path
            while diff > 180.0:
                diff -= 360.0
            while diff < -180.0:
                diff += 360.0
            total_angular_change += diff

        avg_angular_velocity = total_angular_change / (n - 1)

        # Extrapolate from last orientation
        predicted = orientations[-1] + avg_angular_velocity

        # Normalize to [0, 360)
        predicted = predicted % 360.0
        if predicted < 0:
            predicted += 360.0

        return predicted

    def _calculate_reduced_confidence(
        self, base_confidence: float, consecutive_missing: int
    ) -> float:
        """Calculate reduced confidence proportional to gap length.

        Confidence decreases linearly with the number of consecutive missing
        frames. At 1 missing frame, confidence is reduced by 1/6 of base.
        At 5 missing frames, confidence is reduced by 5/6 of base.

        Args:
            base_confidence: The confidence of the last actual detection.
            consecutive_missing: Number of consecutive missing frames.

        Returns:
            Reduced confidence value (always >= 0).
        """
        if consecutive_missing <= 0:
            return base_confidence

        # Linear reduction: at max_frames, confidence approaches 1/6 of base
        reduction_factor = consecutive_missing / (MAX_PREDICTION_FRAMES + 1)
        reduced = base_confidence * (1.0 - reduction_factor)
        return max(0.0, reduced)

    def record_tracking_failure(
        self, trajectory: BatTrajectory, end_frame_index: int
    ) -> None:
        """Record a tracking failure interval in the BatTrajectory.

        Called when consecutive_missing exceeds the threshold and the bat
        is eventually re-detected or analysis ends.

        Args:
            trajectory: The BatTrajectory to record the failure in.
            end_frame_index: The frame index where the failure period ends
                (either re-detection frame or last frame of analysis).
        """
        if self._failure_start_frame is not None:
            trajectory.tracking_failures.append(
                (self._failure_start_frame, end_frame_index)
            )
