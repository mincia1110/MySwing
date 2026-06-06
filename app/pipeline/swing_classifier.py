"""Swing phase classification module (Requirements 5.1, 5.2, 5.3).

Classifies a baseball swing into six distinct phases using pose keypoint
sequences and bat trajectory data. Identifies transition frames between
phases and calculates phase durations in milliseconds.

Phase detection uses heuristic-based signals from keypoint positions
and bat trajectory:
- STANCE: Initial position, minimal movement, bat held still
- LOAD: Weight shift backward, hands move back, bat cocks
- STRIDE: Front foot lifts and moves forward, weight begins transfer
- ROTATION: Hip rotation begins, bat starts forward, rapid angular velocity
- IMPACT: Maximum bat speed zone, bat reaches hitting zone
- FOLLOW_THROUGH: Deceleration after impact, bat wraps around
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from app.models.bat import BatDetectionResult, BatTrajectory
from app.models.enums import SwingPhase
from app.models.pose import Keypoint, PoseResult
from app.models.swing import PhaseAnomaly, SwingPhaseResult, TransitionBoundary

# Minimum confidence threshold for a keypoint to be considered valid
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5

# Maximum ratio of low-confidence frames allowed in a phase region
# before reporting classification failure
LOW_CONFIDENCE_FRAME_RATIO = 0.5

# Minimum phase duration in milliseconds (below this is anomalous)
MIN_PHASE_DURATION_MS = 50.0

# Movement thresholds (normalized coordinates, 0-1 range)
# Threshold for detecting significant movement between frames
MOVEMENT_THRESHOLD = 0.01

# Threshold for detecting hip backward movement (load phase)
HIP_BACKWARD_THRESHOLD = 0.008

# Threshold for detecting hand backward movement (load phase)
HAND_BACKWARD_THRESHOLD = 0.01

# Threshold for front ankle lift (stride phase)
ANKLE_LIFT_THRESHOLD = 0.015

# Threshold for ankle stabilization (end of stride)
ANKLE_STABILIZE_THRESHOLD = 0.005

# Threshold for hip rotation detection (rotation phase)
HIP_ROTATION_THRESHOLD = 0.01

# Bat speed decrease ratio to detect follow-through
BAT_SPEED_DECREASE_RATIO = 0.7

# Minimum frames to consider for a valid swing sequence
MIN_SWING_FRAMES = 10

# Window size for smoothing movement signals
SMOOTHING_WINDOW = 3


class SwingPhaseClassifier:
    """Classifies a baseball swing into six distinct phases.

    Uses pose keypoint sequences and bat trajectory data to identify
    phase transitions and calculate phase durations. Implements
    heuristic-based detection using biomechanical signals.
    """

    def __init__(
        self,
        movement_threshold: float = MOVEMENT_THRESHOLD,
        hip_backward_threshold: float = HIP_BACKWARD_THRESHOLD,
        hand_backward_threshold: float = HAND_BACKWARD_THRESHOLD,
        ankle_lift_threshold: float = ANKLE_LIFT_THRESHOLD,
        ankle_stabilize_threshold: float = ANKLE_STABILIZE_THRESHOLD,
        hip_rotation_threshold: float = HIP_ROTATION_THRESHOLD,
        bat_speed_decrease_ratio: float = BAT_SPEED_DECREASE_RATIO,
        batting_direction: str = "right",
    ) -> None:
        """Initialize SwingPhaseClassifier with detection thresholds.

        Args:
            movement_threshold: General movement detection threshold.
            hip_backward_threshold: Threshold for hip backward shift.
            hand_backward_threshold: Threshold for hand backward movement.
            ankle_lift_threshold: Threshold for front ankle lift detection.
            ankle_stabilize_threshold: Threshold for ankle stabilization.
            hip_rotation_threshold: Threshold for hip rotation onset.
            bat_speed_decrease_ratio: Ratio of speed decrease for follow-through.
            batting_direction: "right" or "left" — determines which keypoints
                are used for back hip, power hand, and front ankle.
        """
        self.movement_threshold = movement_threshold
        self.hip_backward_threshold = hip_backward_threshold
        self.hand_backward_threshold = hand_backward_threshold
        self.ankle_lift_threshold = ankle_lift_threshold
        self.ankle_stabilize_threshold = ankle_stabilize_threshold
        self.hip_rotation_threshold = hip_rotation_threshold
        self.bat_speed_decrease_ratio = bat_speed_decrease_ratio

        # Keypoint names depend on batting direction (Requirement 3.7)
        if batting_direction == "left":
            self._back_hip = "left_hip"
            self._power_wrist = "left_wrist"
            self._front_ankle = "right_ankle"
        else:
            self._back_hip = "right_hip"
            self._power_wrist = "right_wrist"
            self._front_ankle = "left_ankle"

    def classify_phases(
        self,
        pose_sequence: List[PoseResult],
        bat_trajectory: BatTrajectory,
        fps: float,
    ) -> SwingPhaseResult:
        """Classify a swing into six phases using pose and bat data.

        Identifies transition frames between phases, assigns frame ranges
        to each phase, and calculates durations in milliseconds.
        Detects anomalies (missing or abnormally short phases) and
        classification failures due to insufficient pose data.

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.
            bat_trajectory: BatTrajectory with detections and speeds.
            fps: Video frame rate (frames per second).

        Returns:
            SwingPhaseResult with phases, transitions, durations, and anomalies.
        """
        if not pose_sequence or fps <= 0:
            return SwingPhaseResult()

        # Find transition frames
        transitions = self.find_transition_frames(pose_sequence, bat_trajectory)

        # Build phase ranges from transitions
        phases = self._build_phase_ranges(pose_sequence, transitions)

        # Calculate durations
        phase_durations_ms = self.calculate_phase_durations(transitions, fps)

        # Also calculate duration for first and last phases from frame ranges
        for phase, (start, end) in phases.items():
            if phase not in phase_durations_ms:
                phase_durations_ms[phase] = (end - start) / fps * 1000.0

        # Build intermediate result for anomaly detection
        phase_result = SwingPhaseResult(
            phases=phases,
            transitions=transitions,
            phase_durations_ms=phase_durations_ms,
            anomalies=[],
        )

        # Detect anomalies (missing or abnormally short phases)
        anomalies = self.detect_anomalies(phase_result, fps)

        # Detect classification failures and add as anomalies
        classification_failures = self.detect_classification_failures(
            pose_sequence, phase_result
        )

        phase_result.anomalies = anomalies
        phase_result.classification_failures = classification_failures

        return phase_result

    def find_transition_frames(
        self,
        pose_sequence: List[PoseResult],
        bat_trajectory: BatTrajectory,
    ) -> List[TransitionBoundary]:
        """Identify transition frames between consecutive swing phases.

        Uses biomechanical signals to detect phase boundaries:
        - Stance→Load: Back hip begins moving backward OR hands move back
        - Load→Stride: Front foot lifts (front ankle y changes significantly)
        - Stride→Rotation: Front foot plants AND hip rotation begins
        - Rotation→Impact: Bat speed reaches peak OR bat enters hitting zone
        - Impact→Follow-through: Bat speed begins decreasing after peak

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.
            bat_trajectory: BatTrajectory with detections and speeds.

        Returns:
            List of TransitionBoundary in chronological order.
        """
        if len(pose_sequence) < MIN_SWING_FRAMES:
            return []

        transitions: List[TransitionBoundary] = []
        start_frame = pose_sequence[0].frame_index
        end_frame = pose_sequence[-1].frame_index

        # Extract signals for transition detection
        hip_movement = self._compute_hip_backward_signal(pose_sequence)
        hand_movement = self._compute_hand_backward_signal(pose_sequence)
        ankle_lift = self._compute_ankle_lift_signal(pose_sequence)
        ankle_stable = self._compute_ankle_stabilize_signal(pose_sequence)
        hip_rotation = self._compute_hip_rotation_signal(pose_sequence)
        bat_speeds = self._get_bat_speeds(bat_trajectory, start_frame, end_frame)

        # 1. Stance → Load: back hip or hands begin moving backward
        stance_to_load = self._find_stance_to_load(
            pose_sequence, hip_movement, hand_movement
        )
        if stance_to_load is not None:
            transitions.append(
                TransitionBoundary(
                    from_phase=SwingPhase.STANCE,
                    to_phase=SwingPhase.LOAD,
                    frame_index=stance_to_load,
                    confidence=0.8,
                )
            )

        # 2. Load → Stride: front foot lifts
        load_to_stride = self._find_load_to_stride(
            pose_sequence, ankle_lift, stance_to_load
        )
        if load_to_stride is not None:
            transitions.append(
                TransitionBoundary(
                    from_phase=SwingPhase.LOAD,
                    to_phase=SwingPhase.STRIDE,
                    frame_index=load_to_stride,
                    confidence=0.8,
                )
            )

        # 3. Stride → Rotation: front foot plants AND hip rotation begins
        stride_to_rotation = self._find_stride_to_rotation(
            pose_sequence, ankle_stable, hip_rotation, load_to_stride
        )
        if stride_to_rotation is not None:
            transitions.append(
                TransitionBoundary(
                    from_phase=SwingPhase.STRIDE,
                    to_phase=SwingPhase.ROTATION,
                    frame_index=stride_to_rotation,
                    confidence=0.8,
                )
            )

        # 4. Rotation → Impact: bat speed reaches peak
        rotation_to_impact = self._find_rotation_to_impact(
            bat_speeds, start_frame, stride_to_rotation
        )
        if rotation_to_impact is not None:
            transitions.append(
                TransitionBoundary(
                    from_phase=SwingPhase.ROTATION,
                    to_phase=SwingPhase.IMPACT,
                    frame_index=rotation_to_impact,
                    confidence=0.85,
                )
            )

        # 5. Impact → Follow-through: bat speed begins decreasing
        impact_to_follow = self._find_impact_to_follow_through(
            bat_speeds, start_frame, rotation_to_impact
        )
        if impact_to_follow is not None:
            transitions.append(
                TransitionBoundary(
                    from_phase=SwingPhase.IMPACT,
                    to_phase=SwingPhase.FOLLOW_THROUGH,
                    frame_index=impact_to_follow,
                    confidence=0.85,
                )
            )

        return transitions

    def calculate_phase_durations(
        self,
        transitions: List[TransitionBoundary],
        fps: float,
    ) -> Dict[SwingPhase, float]:
        """Calculate the duration of each phase in milliseconds.

        Duration formula: phase_duration_ms = (end_frame - start_frame) / fps * 1000

        Args:
            transitions: List of TransitionBoundary in chronological order.
            fps: Video frame rate (frames per second).

        Returns:
            Dictionary mapping SwingPhase to duration in milliseconds.
        """
        if not transitions or fps <= 0:
            return {}

        durations: Dict[SwingPhase, float] = {}

        # Calculate duration between consecutive transitions
        for i in range(len(transitions) - 1):
            current = transitions[i]
            next_trans = transitions[i + 1]
            phase = current.to_phase
            duration_ms = (next_trans.frame_index - current.frame_index) / fps * 1000.0
            durations[phase] = duration_ms

        return durations

    def detect_anomalies(
        self,
        phase_result: SwingPhaseResult,
        fps: float,
    ) -> List[PhaseAnomaly]:
        """Detect anomalies in swing phase classification (Requirement 5.4).

        Checks each of the 6 expected swing phases for:
        - Missing phases: phase not present in the classification result
        - Abnormally short phases: phase duration < 50ms

        Args:
            phase_result: The swing phase classification result.
            fps: Video frame rate (frames per second).

        Returns:
            List of PhaseAnomaly for all detected anomalies.
        """
        anomalies: List[PhaseAnomaly] = []
        all_phases = [
            SwingPhase.STANCE,
            SwingPhase.LOAD,
            SwingPhase.STRIDE,
            SwingPhase.ROTATION,
            SwingPhase.IMPACT,
            SwingPhase.FOLLOW_THROUGH,
        ]

        for phase in all_phases:
            if phase not in phase_result.phases:
                # Phase is missing entirely
                anomalies.append(
                    PhaseAnomaly(
                        phase=phase,
                        anomaly_type="missing",
                        duration_ms=None,
                    )
                )
            elif phase in phase_result.phase_durations_ms:
                duration_ms = phase_result.phase_durations_ms[phase]
                if duration_ms < MIN_PHASE_DURATION_MS:
                    # Phase is abnormally short
                    anomalies.append(
                        PhaseAnomaly(
                            phase=phase,
                            anomaly_type="abnormally_short",
                            duration_ms=duration_ms,
                        )
                    )

        return anomalies

    def detect_classification_failures(
        self,
        pose_sequence: List[PoseResult],
        phase_result: SwingPhaseResult,
    ) -> List[str]:
        """Detect classification failures due to insufficient pose data (Requirement 5.5).

        Checks if any phases could not be reliably classified because too many
        frames in the phase region have low-confidence pose data. A phase is
        considered to have a classification failure if more than 50% of its
        frames have low confidence (is_low_confidence=True) or insufficient
        keypoints (overall_confidence < 0.5).

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.
            phase_result: The swing phase classification result.

        Returns:
            List of phase names (str) that could not be reliably classified.
        """
        if not pose_sequence or not phase_result.phases:
            return []

        # Build a frame_index → PoseResult lookup
        pose_by_frame: Dict[int, PoseResult] = {
            pose.frame_index: pose for pose in pose_sequence
        }

        failed_phases: List[str] = []

        for phase, (start_frame, end_frame) in phase_result.phases.items():
            # Count frames in this phase region
            total_frames = 0
            low_confidence_frames = 0

            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx in pose_by_frame:
                    total_frames += 1
                    pose = pose_by_frame[frame_idx]
                    if (
                        pose.is_low_confidence
                        or pose.overall_confidence < KEYPOINT_CONFIDENCE_THRESHOLD
                    ):
                        low_confidence_frames += 1

            # If more than 50% of frames are low-confidence, report failure
            if total_frames > 0 and (
                low_confidence_frames / total_frames > LOW_CONFIDENCE_FRAME_RATIO
            ):
                failed_phases.append(phase.value)

        return failed_phases

    # -------------------------------------------------------------------------
    # Private helper methods for signal computation
    # -------------------------------------------------------------------------

    def _get_keypoint_by_name(
        self, pose: PoseResult, name: str
    ) -> Optional[Keypoint]:
        """Get a keypoint by name from a PoseResult.

        Args:
            pose: PoseResult containing keypoints.
            name: Name of the keypoint to find.

        Returns:
            Keypoint if found, None otherwise.
        """
        for kp in pose.keypoints:
            if kp.name == name:
                return kp
        return None

    def _compute_hip_backward_signal(
        self, pose_sequence: List[PoseResult]
    ) -> List[float]:
        """Compute frame-by-frame hip backward movement signal.

        Measures the absolute x-displacement of the back hip between
        consecutive frames. Positive values indicate backward movement
        (weight shift away from pitcher in either direction).

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.

        Returns:
            List of displacement values (length = len(pose_sequence) - 1).
        """
        signals: List[float] = []
        for i in range(1, len(pose_sequence)):
            prev_hip = self._get_keypoint_by_name(pose_sequence[i - 1], self._back_hip)
            curr_hip = self._get_keypoint_by_name(pose_sequence[i], self._back_hip)

            if prev_hip and curr_hip and prev_hip.confidence >= 0.5 and curr_hip.confidence >= 0.5:
                # Backward movement = absolute x displacement of back hip
                displacement = abs(curr_hip.x - prev_hip.x)
                signals.append(displacement)
            else:
                signals.append(0.0)

        return signals

    def _compute_hand_backward_signal(
        self, pose_sequence: List[PoseResult]
    ) -> List[float]:
        """Compute frame-by-frame hand backward movement signal.

        Measures absolute x-displacement of wrists between consecutive frames.

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.

        Returns:
            List of displacement values (length = len(pose_sequence) - 1).
        """
        signals: List[float] = []
        for i in range(1, len(pose_sequence)):
            prev_wrist = self._get_keypoint_by_name(pose_sequence[i - 1], self._power_wrist)
            curr_wrist = self._get_keypoint_by_name(pose_sequence[i], self._power_wrist)

            if prev_wrist and curr_wrist and prev_wrist.confidence >= 0.5 and curr_wrist.confidence >= 0.5:
                displacement = abs(curr_wrist.x - prev_wrist.x)
                signals.append(displacement)
            else:
                signals.append(0.0)

        return signals

    def _compute_ankle_lift_signal(
        self, pose_sequence: List[PoseResult]
    ) -> List[float]:
        """Compute frame-by-frame front ankle vertical movement signal.

        Measures upward (negative y in image coords) displacement of the
        front ankle. Positive values indicate the foot is lifting.

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.

        Returns:
            List of displacement values (length = len(pose_sequence) - 1).
        """
        signals: List[float] = []
        for i in range(1, len(pose_sequence)):
            prev_ankle = self._get_keypoint_by_name(pose_sequence[i - 1], self._front_ankle)
            curr_ankle = self._get_keypoint_by_name(pose_sequence[i], self._front_ankle)

            if prev_ankle and curr_ankle and prev_ankle.confidence >= 0.5 and curr_ankle.confidence >= 0.5:
                # In image coordinates, y decreases upward, so lift = prev_y - curr_y
                displacement = prev_ankle.y - curr_ankle.y
                signals.append(displacement)
            else:
                signals.append(0.0)

        return signals

    def _compute_ankle_stabilize_signal(
        self, pose_sequence: List[PoseResult]
    ) -> List[float]:
        """Compute frame-by-frame front ankle stability signal.

        Measures absolute movement of the front ankle. Low values indicate
        the foot has planted (stabilized).

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.

        Returns:
            List of absolute movement values (length = len(pose_sequence) - 1).
        """
        signals: List[float] = []
        for i in range(1, len(pose_sequence)):
            prev_ankle = self._get_keypoint_by_name(pose_sequence[i - 1], self._front_ankle)
            curr_ankle = self._get_keypoint_by_name(pose_sequence[i], self._front_ankle)

            if prev_ankle and curr_ankle and prev_ankle.confidence >= 0.5 and curr_ankle.confidence >= 0.5:
                dx = abs(curr_ankle.x - prev_ankle.x)
                dy = abs(curr_ankle.y - prev_ankle.y)
                movement = math.sqrt(dx * dx + dy * dy)
                signals.append(movement)
            else:
                signals.append(0.0)

        return signals

    def _compute_hip_rotation_signal(
        self, pose_sequence: List[PoseResult]
    ) -> List[float]:
        """Compute frame-by-frame hip rotation signal.

        Measures the change in distance between left and right hips,
        which indicates rotation in the frontal plane.

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.

        Returns:
            List of rotation signal values (length = len(pose_sequence) - 1).
        """
        signals: List[float] = []
        for i in range(1, len(pose_sequence)):
            prev_left_hip = self._get_keypoint_by_name(pose_sequence[i - 1], "left_hip")
            prev_right_hip = self._get_keypoint_by_name(pose_sequence[i - 1], "right_hip")
            curr_left_hip = self._get_keypoint_by_name(pose_sequence[i], "left_hip")
            curr_right_hip = self._get_keypoint_by_name(pose_sequence[i], "right_hip")

            if (
                prev_left_hip
                and prev_right_hip
                and curr_left_hip
                and curr_right_hip
                and all(
                    kp.confidence >= 0.5
                    for kp in [prev_left_hip, prev_right_hip, curr_left_hip, curr_right_hip]
                )
            ):
                prev_dist = abs(prev_left_hip.x - prev_right_hip.x)
                curr_dist = abs(curr_left_hip.x - curr_right_hip.x)
                # Rotation causes the hip distance to change
                rotation_signal = abs(curr_dist - prev_dist)
                signals.append(rotation_signal)
            else:
                signals.append(0.0)

        return signals

    def _get_bat_speeds(
        self,
        bat_trajectory: BatTrajectory,
        start_frame: int,
        end_frame: int,
    ) -> List[float]:
        """Get bat speeds aligned to the frame range.

        Args:
            bat_trajectory: BatTrajectory with speed data.
            start_frame: First frame index of the pose sequence.
            end_frame: Last frame index of the pose sequence.

        Returns:
            List of bat speeds indexed relative to start_frame.
        """
        total_frames = end_frame - start_frame + 1
        speeds = [0.0] * total_frames

        if not bat_trajectory.bat_speed_pixels_per_frame:
            return speeds

        # Map bat speeds to frame indices
        for i, speed in enumerate(bat_trajectory.bat_speed_pixels_per_frame):
            if i < len(bat_trajectory.detections) - 1:
                det = bat_trajectory.detections[i]
                frame_offset = det.frame_index - start_frame
                if 0 <= frame_offset < total_frames:
                    speeds[frame_offset] = speed

        return speeds

    def _find_stance_to_load(
        self,
        pose_sequence: List[PoseResult],
        hip_movement: List[float],
        hand_movement: List[float],
    ) -> Optional[int]:
        """Find the transition frame from Stance to Load phase.

        Detects when the back hip begins moving backward OR hands begin
        moving back, indicating the start of the loading phase.

        Args:
            pose_sequence: List of PoseResult.
            hip_movement: Hip backward movement signal.
            hand_movement: Hand backward movement signal.

        Returns:
            Frame index of the transition, or None if not detected.
        """
        for i in range(len(hip_movement)):
            if (
                hip_movement[i] > self.hip_backward_threshold
                or hand_movement[i] > self.hand_backward_threshold
            ):
                return pose_sequence[i + 1].frame_index

        return None

    def _find_load_to_stride(
        self,
        pose_sequence: List[PoseResult],
        ankle_lift: List[float],
        stance_to_load: Optional[int],
    ) -> Optional[int]:
        """Find the transition frame from Load to Stride phase.

        Detects when the front foot lifts (front ankle y-coordinate
        changes significantly), indicating stride initiation.

        Args:
            pose_sequence: List of PoseResult.
            ankle_lift: Ankle lift signal.
            stance_to_load: Frame index of Stance→Load transition.

        Returns:
            Frame index of the transition, or None if not detected.
        """
        # Start searching after the Stance→Load transition
        search_start = 0
        if stance_to_load is not None:
            for idx, pose in enumerate(pose_sequence):
                if pose.frame_index >= stance_to_load:
                    search_start = max(0, idx - 1)
                    break

        for i in range(search_start, len(ankle_lift)):
            if ankle_lift[i] > self.ankle_lift_threshold:
                return pose_sequence[i + 1].frame_index

        return None

    def _find_stride_to_rotation(
        self,
        pose_sequence: List[PoseResult],
        ankle_stable: List[float],
        hip_rotation: List[float],
        load_to_stride: Optional[int],
    ) -> Optional[int]:
        """Find the transition frame from Stride to Rotation phase.

        Detects when the front foot plants (ankle stabilizes) AND hip
        rotation begins.

        Args:
            pose_sequence: List of PoseResult.
            ankle_stable: Ankle stability signal.
            hip_rotation: Hip rotation signal.
            load_to_stride: Frame index of Load→Stride transition.

        Returns:
            Frame index of the transition, or None if not detected.
        """
        # Cannot detect stride→rotation without a prior load→stride transition
        if load_to_stride is None:
            return None

        # Start searching after the Load→Stride transition
        search_start = 0
        for idx, pose in enumerate(pose_sequence):
            if pose.frame_index >= load_to_stride:
                search_start = max(0, idx - 1)
                break

        # Need at least a few frames of stride before rotation
        search_start = search_start + 2

        for i in range(search_start, len(ankle_stable)):
            ankle_planted = ankle_stable[i] < self.ankle_stabilize_threshold
            hip_rotating = hip_rotation[i] > self.hip_rotation_threshold

            if ankle_planted and hip_rotating:
                return pose_sequence[i + 1].frame_index

        # Fallback: if ankle plants but no clear hip rotation detected,
        # look for just ankle stabilization after sufficient stride frames
        for i in range(search_start, len(ankle_stable)):
            if ankle_stable[i] < self.ankle_stabilize_threshold:
                return pose_sequence[i + 1].frame_index

        return None

    def _find_rotation_to_impact(
        self,
        bat_speeds: List[float],
        start_frame: int,
        stride_to_rotation: Optional[int],
    ) -> Optional[int]:
        """Find the transition frame from Rotation to Impact phase.

        Detects when bat speed reaches its peak, indicating the bat
        has entered the hitting zone.

        Args:
            bat_speeds: List of bat speeds indexed from start_frame.
            start_frame: First frame index.
            stride_to_rotation: Frame index of Stride→Rotation transition.

        Returns:
            Frame index of the transition, or None if not detected.
        """
        if not bat_speeds or max(bat_speeds) == 0:
            return None

        # Start searching after the Stride→Rotation transition
        search_start = 0
        if stride_to_rotation is not None:
            search_start = stride_to_rotation - start_frame

        search_start = max(0, min(search_start, len(bat_speeds) - 1))

        # Find the peak bat speed after rotation starts
        max_speed = 0.0
        max_speed_idx = search_start

        for i in range(search_start, len(bat_speeds)):
            if bat_speeds[i] > max_speed:
                max_speed = bat_speeds[i]
                max_speed_idx = i

        if max_speed > 0:
            return start_frame + max_speed_idx

        return None

    def _find_impact_to_follow_through(
        self,
        bat_speeds: List[float],
        start_frame: int,
        rotation_to_impact: Optional[int],
    ) -> Optional[int]:
        """Find the transition frame from Impact to Follow-through phase.

        Detects when bat speed begins decreasing after reaching its peak,
        indicating the bat has passed through the hitting zone.

        Args:
            bat_speeds: List of bat speeds indexed from start_frame.
            start_frame: First frame index.
            rotation_to_impact: Frame index of Rotation→Impact transition.

        Returns:
            Frame index of the transition, or None if not detected.
        """
        if not bat_speeds:
            return None

        # Start searching after the impact frame
        search_start = 0
        if rotation_to_impact is not None:
            search_start = rotation_to_impact - start_frame + 1

        search_start = max(0, min(search_start, len(bat_speeds) - 1))

        # Find where speed drops below the threshold ratio of peak
        peak_speed = max(bat_speeds) if bat_speeds else 0.0
        if peak_speed == 0:
            return None

        threshold = peak_speed * self.bat_speed_decrease_ratio

        for i in range(search_start, len(bat_speeds)):
            if bat_speeds[i] < threshold:
                return start_frame + i

        # If no clear decrease found, use the frame after peak + a few frames
        if rotation_to_impact is not None:
            follow_frame = rotation_to_impact + 2
            if follow_frame <= start_frame + len(bat_speeds) - 1:
                return follow_frame

        return None

    def _build_phase_ranges(
        self,
        pose_sequence: List[PoseResult],
        transitions: List[TransitionBoundary],
    ) -> Dict[SwingPhase, Tuple[int, int]]:
        """Build phase frame ranges from transition boundaries.

        Args:
            pose_sequence: List of PoseResult ordered by frame_index.
            transitions: List of TransitionBoundary in chronological order.

        Returns:
            Dictionary mapping SwingPhase to (start_frame, end_frame) tuples.
        """
        if not pose_sequence:
            return {}

        phases: Dict[SwingPhase, Tuple[int, int]] = {}
        start_frame = pose_sequence[0].frame_index
        end_frame = pose_sequence[-1].frame_index

        if not transitions:
            # No transitions detected - entire sequence is STANCE
            phases[SwingPhase.STANCE] = (start_frame, end_frame)
            return phases

        # First phase: STANCE from start to first transition
        phases[SwingPhase.STANCE] = (start_frame, transitions[0].frame_index)

        # Middle phases: between consecutive transitions
        for i, trans in enumerate(transitions):
            phase = trans.to_phase
            phase_start = trans.frame_index

            if i + 1 < len(transitions):
                phase_end = transitions[i + 1].frame_index
            else:
                phase_end = end_frame

            phases[phase] = (phase_start, phase_end)

        return phases
