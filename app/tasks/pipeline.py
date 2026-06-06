"""Pipeline task stubs for the swing analysis workflow.

Defines Celery tasks for each stage of the analysis pipeline:
- Video preprocessing
- Pose estimation
- Bat detection
- Swing phase classification
- Biomechanics analysis
- Swing evaluation
- Report generation
- Orchestrator task that chains the pipeline

Requirements: 6.9, 6.10, 8.1
"""

import dataclasses
import logging
import math
import os
import shutil
import statistics
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from celery.exceptions import SoftTimeLimitExceeded

from app.core.celery_app import DEFAULT_RETRY_POLICY, celery_app
from app.db.session import sync_session_factory

logger = logging.getLogger(__name__)

# Analysis status constants
STATUS_PENDING = "pending"
STATUS_PREPROCESSING = "preprocessing"
STATUS_ANALYZING = "analyzing"
STATUS_EVALUATING = "evaluating"
STATUS_GENERATING_REPORT = "generating_report"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def _update_analysis_status(
    analysis_id: str,
    status: str,
    error_message: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> None:
    """Update the analysis record status in the database.

    Args:
        analysis_id: UUID of the analysis record.
        status: New status value.
        error_message: Optional error message (for failed status).
        started_at: Optional timestamp when analysis started.
        completed_at: Optional timestamp when analysis completed.
    """
    from app.db.models import AnalysisTable

    session = sync_session_factory()
    try:
        analysis = session.query(AnalysisTable).filter(
            AnalysisTable.id == uuid.UUID(analysis_id)
        ).first()
        if analysis:
            analysis.status = status
            if error_message is not None:
                analysis.error_message = error_message
            if started_at is not None:
                analysis.started_at = started_at
            if completed_at is not None:
                analysis.completed_at = completed_at
            session.commit()
            logger.info(
                "Updated analysis %s status to '%s'", analysis_id, status
            )
        else:
            logger.warning("Analysis %s not found for status update", analysis_id)
    except Exception as e:
        session.rollback()
        logger.error(
            "Failed to update analysis %s status: %s", analysis_id, str(e)
        )
    finally:
        session.close()


def _get_analysis_data(analysis_id: str) -> dict[str, Any] | None:
    """Retrieve analysis record with related video and user profile data.

    Args:
        analysis_id: UUID of the analysis record.

    Returns:
        Dictionary with analysis data or None if not found.
    """
    from app.db.models import AnalysisTable, UserProfileTable, VideoTable

    session = sync_session_factory()
    try:
        analysis = session.query(AnalysisTable).filter(
            AnalysisTable.id == uuid.UUID(analysis_id)
        ).first()
        if not analysis:
            return None

        video = session.query(VideoTable).filter(
            VideoTable.id == analysis.video_id
        ).first()

        user_profile = session.query(UserProfileTable).filter(
            UserProfileTable.user_id == analysis.user_id
        ).first()

        result = {
            "analysis_id": analysis_id,
            "user_id": str(analysis.user_id),
            "video_id": str(analysis.video_id),
            "video_file_key": video.file_key if video else None,
            "video_fps": video.frame_rate if video else 30.0,
            "video_width": video.resolution_width if video else 1920,
            "video_height": video.resolution_height if video else 1080,
            "user_profile": None,
        }

        if user_profile:
            result["user_profile"] = {
                "height": user_profile.height,
                "bat_length": user_profile.bat_length,
                "batting_direction": user_profile.batting_direction,
                "weight": user_profile.weight,
                "level": user_profile.level,
                "age_group": user_profile.age_group,
            }

        return result
    except Exception as e:
        logger.error("Failed to get analysis data for %s: %s", analysis_id, str(e))
        return None
    finally:
        session.close()


def _build_analysis_metadata(preprocessing_result: dict | None) -> dict[str, Any]:
    """Build API-facing metadata that documents analysis-time preprocessing."""
    if not preprocessing_result:
        return {}

    normalization_metadata = {
        "normalization_applied": bool(
            preprocessing_result.get("normalization_applied", False)
        ),
        "normalization_target_fps": preprocessing_result.get(
            "normalization_target_fps"
        ),
        "normalization_crop_box": preprocessing_result.get(
            "normalization_crop_box"
        ),
        "normalization_sampled_frame_count": preprocessing_result.get(
            "normalization_sampled_frame_count"
        ),
        "original_fps": preprocessing_result.get("original_fps"),
        "original_video_width": preprocessing_result.get("original_video_width"),
        "original_video_height": preprocessing_result.get("original_video_height"),
        "original_frame_count": preprocessing_result.get("original_frame_count"),
        "analysis_fps": preprocessing_result.get("fps"),
        "analysis_video_width": preprocessing_result.get("video_width"),
        "analysis_video_height": preprocessing_result.get("video_height"),
        "analysis_frame_count": preprocessing_result.get("frame_count"),
    }

    return {
        "video_normalization": normalization_metadata,
        "analysis_coordinate_system": preprocessing_result.get(
            "analysis_coordinate_system"
        ),
        "canonical_batting_direction": preprocessing_result.get(
            "canonical_batting_direction"
        ),
    }



def _save_analysis_result(
    analysis_id: str,
    biomechanics_data: dict,
    swing_phases_data: dict,
    evaluations_data: dict,
    improvements_data: dict,
    drill_recommendations: dict,
    overlay_video_key: str | None = None,
    processing_time_seconds: float | None = None,
) -> None:
    """Save analysis results to the database.

    Args:
        analysis_id: UUID of the analysis record.
        biomechanics_data: Biomechanics analysis results as dict.
        swing_phases_data: Swing phase classification results as dict.
        evaluations_data: Evaluation results as dict.
        improvements_data: Improvement areas as dict.
        drill_recommendations: Drill recommendations as dict.
        overlay_video_key: S3 key for overlay video.
        processing_time_seconds: Total processing time.
    """
    from app.db.models import AnalysisResultTable

    session = sync_session_factory()
    try:
        existing = session.query(AnalysisResultTable).filter(
            AnalysisResultTable.analysis_id == uuid.UUID(analysis_id)
        ).first()

        if existing:
            existing.biomechanics_data = biomechanics_data
            existing.swing_phases_data = swing_phases_data
            existing.evaluations_data = evaluations_data
            existing.improvements_data = improvements_data
            existing.drill_recommendations = drill_recommendations
            existing.overlay_video_key = overlay_video_key
            existing.processing_time_seconds = processing_time_seconds
        else:
            result = AnalysisResultTable(
                analysis_id=uuid.UUID(analysis_id),
                biomechanics_data=biomechanics_data,
                swing_phases_data=swing_phases_data,
                evaluations_data=evaluations_data,
                improvements_data=improvements_data,
                drill_recommendations=drill_recommendations,
                overlay_video_key=overlay_video_key,
                processing_time_seconds=processing_time_seconds,
            )
            session.add(result)

        session.commit()
        logger.info("Saved analysis result for %s", analysis_id)
    except Exception as e:
        session.rollback()
        logger.error(
            "Failed to save analysis result for %s: %s", analysis_id, str(e)
        )
        raise
    finally:
        session.close()


# ============================================================================
# Serialization helpers
# ============================================================================


def _serialize_dataclass(obj: Any) -> Any:
    """Recursively serialize a dataclass (or list/dict of dataclasses) to dicts.

    Handles nested dataclasses, enums, tuples, and numpy types.
    """
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = _serialize_dataclass(value)
        return result
    if isinstance(obj, list):
        return [_serialize_dataclass(item) for item in obj]
    if isinstance(obj, dict):
        return {
            _serialize_dataclass(k): _serialize_dataclass(v)
            for k, v in obj.items()
        }
    if isinstance(obj, tuple):
        return [_serialize_dataclass(item) for item in obj]
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _deserialize_pose_sequence(data: list[dict]) -> list:
    """Deserialize pose_sequence from dicts back to PoseResult objects."""
    from app.models.pose import Keypoint, PoseResult

    results = []
    for item in data:
        keypoints = [
            Keypoint(
                x=kp["x"], y=kp["y"], z=kp["z"],
                confidence=kp["confidence"], name=kp["name"],
            )
            for kp in item.get("keypoints", [])
        ]
        results.append(PoseResult(
            frame_index=item["frame_index"],
            keypoints=keypoints,
            person_id=item.get("person_id", 0),
            is_primary_batter=item.get("is_primary_batter", True),
            overall_confidence=item.get("overall_confidence", 0.0),
            is_low_confidence=item.get("is_low_confidence", False),
        ))
    return results


def _deserialize_bat_trajectory(data: dict) -> Any:
    """Deserialize bat_trajectory from dict back to BatTrajectory object."""
    from app.models.bat import BatDetectionResult, BatTrajectory

    if not data or not isinstance(data, dict):
        return BatTrajectory()

    detections = []
    for det in data.get("detections", []):
        pos = det.get("position", [0.0, 0.0])
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            position = (float(pos[0]), float(pos[1]))
        else:
            position = (0.0, 0.0)
        detections.append(BatDetectionResult(
            frame_index=det.get("frame_index", 0),
            detected=det.get("detected", False),
            position=position,
            orientation_angle=det.get("orientation_angle", 0.0),
            length_pixels=det.get("length_pixels", 0.0),
            confidence=det.get("confidence", 0.0),
            is_predicted=det.get("is_predicted", False),
            coordinate_space=det.get("coordinate_space", "pixel"),
            bat_head_position=(
                (float(det["bat_head_position"][0]), float(det["bat_head_position"][1]))
                if isinstance(det.get("bat_head_position"), (list, tuple))
                and len(det.get("bat_head_position")) >= 2
                else None
            ),
        ))

    failures = []
    for f in data.get("tracking_failures", []):
        if isinstance(f, (list, tuple)) and len(f) >= 2:
            failures.append((int(f[0]), int(f[1])))

    return BatTrajectory(
        detections=detections,
        bat_speed_pixels_per_frame=data.get("bat_speed_pixels_per_frame", []),
        tracking_accuracy=data.get("tracking_accuracy", 0.0),
        tracking_failures=failures,
    )


# ============================================================================
# Frame I/O helpers
# ============================================================================


def _save_frames_to_temp_dir(frames: list[np.ndarray]) -> str:
    """Save frames as individual .npy files in a temp directory.

    Returns the temp directory path.
    """
    temp_dir = tempfile.mkdtemp(prefix="myswing_frames_")
    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_dir, f"frame_{i:06d}.npy")
        np.save(frame_path, frame)
    return temp_dir


def _load_frames_from_temp_dir(temp_dir: str) -> list[np.ndarray]:
    """Load frames from a temp directory of .npy files."""
    if not temp_dir or not os.path.isdir(temp_dir):
        return []

    frame_files = sorted(
        f for f in os.listdir(temp_dir) if f.endswith(".npy")
    )
    frames = []
    for fname in frame_files:
        frame_path = os.path.join(temp_dir, fname)
        frames.append(np.load(frame_path))
    return frames


# ============================================================================
# Orchestrator task
# ============================================================================


@celery_app.task(
    bind=True,
    name="app.tasks.pipeline.analyze_swing_task",
    max_retries=DEFAULT_RETRY_POLICY["max_retries"],
    retry_backoff=DEFAULT_RETRY_POLICY["retry_backoff"],
    retry_backoff_max=DEFAULT_RETRY_POLICY["retry_backoff_max"],
    retry_jitter=DEFAULT_RETRY_POLICY["retry_jitter"],
    soft_time_limit=600,
    time_limit=720,
)
def analyze_swing_task(self, analysis_id: str) -> dict[str, Any]:
    """Orchestrator task that coordinates the full analysis pipeline.

    Chains: preprocess -> pose estimation -> bat detection ->
    swing classification -> biomechanics -> evaluation -> report generation.

    Each step updates the DB status:
    pending -> preprocessing -> analyzing -> evaluating -> generating_report -> completed/failed

    Implements graceful degradation: partial failures produce partial results.
    Error handling: catches exceptions, updates status to 'failed' with error_message.
    Retry logic: max 2 retries with exponential backoff.
    Timeout: 600s soft limit, 720s hard limit.

    Args:
        analysis_id: UUID of the analysis record.

    Returns:
        Dictionary with analysis results and status.
    """
    start_time = time.time()
    logger.info("Starting swing analysis pipeline for analysis_id=%s", analysis_id)
    temp_dir = None

    try:
        # Mark as started
        _update_analysis_status(
            analysis_id,
            STATUS_PREPROCESSING,
            started_at=datetime.now(timezone.utc),
        )

        # Get analysis data from DB
        analysis_data = _get_analysis_data(analysis_id)
        if not analysis_data:
            raise ValueError(f"Analysis record not found: {analysis_id}")

        video_file_key = analysis_data.get("video_file_key")
        user_profile = analysis_data.get("user_profile")
        fps = analysis_data.get("video_fps", 30.0)
        video_width = analysis_data.get("video_width", 1920)
        video_height = analysis_data.get("video_height", 1080)

        # Determine if frames should be flipped for left-handed batters
        flip_horizontal = False
        if user_profile:
            flip_horizontal = user_profile.get("batting_direction", "right") == "left"

        # Explicitly track coordinate system contract used by downstream analyzers.
        # If flipped, analysis runs in canonical RHB space; otherwise original space.
        analysis_coordinate_system = "canonical_rhb" if flip_horizontal else "original"

        # ---- Step 1: Preprocess Video ----
        preprocessing_result = _run_preprocessing(
            analysis_id, video_file_key, flip_horizontal=flip_horizontal
        )
        preprocessing_result["analysis_coordinate_system"] = analysis_coordinate_system
        preprocessing_result["canonical_batting_direction"] = "right"
        temp_dir = preprocessing_result.get("frames_dir")
        fps = preprocessing_result.get("fps", fps)
        video_width = preprocessing_result.get("video_width", video_width)
        video_height = preprocessing_result.get("video_height", video_height)

        # ---- Step 2: Pose Estimation + Wrist-based Bat Estimation ----
        _update_analysis_status(analysis_id, STATUS_ANALYZING)

        pose_result = _run_pose_estimation(analysis_id, preprocessing_result)
        bat_result = _run_wrist_bat_estimation(
            analysis_id,
            pose_result,
            preprocessing_result,
            dominant_hand="right",  # canonical RHB analysis space after optional LHB frame flip
        )

        # ---- Step 3: Swing Classification ----
        swing_phases_result = _run_swing_classification(
            analysis_id,
            pose_result,
            bat_result,
            fps,
            batting_direction="right",  # canonical RHB analysis space after optional LHB frame flip
        )

        # ---- Step 4: Biomechanics Analysis ----
        _update_analysis_status(analysis_id, STATUS_EVALUATING)

        biomechanics_result = _run_biomechanics_analysis(
            analysis_id, pose_result, bat_result, swing_phases_result,
            user_profile, fps, preprocessing_result
        )

        # ---- Step 5: Swing Evaluation ----
        evaluation_result = _run_swing_evaluation(
            analysis_id, biomechanics_result, bat_result, pose_result,
            swing_phases_result, user_profile, fps
        )

        # ---- Step 6: Report Generation ----
        _update_analysis_status(analysis_id, STATUS_GENERATING_REPORT)

        report_result = _run_report_generation(
            analysis_id, evaluation_result, biomechanics_result,
            swing_phases_result, preprocessing_result, pose_result, bat_result
        )

        # ---- Complete ----
        processing_time = time.time() - start_time

        # Save results to DB
        _save_analysis_result(
            analysis_id=analysis_id,
            biomechanics_data=biomechanics_result,
            swing_phases_data=swing_phases_result,
            evaluations_data=evaluation_result.get("evaluations", {}),
            improvements_data=evaluation_result.get("improvements", {}),
            drill_recommendations=evaluation_result.get(
                "drill_recommendations", {}
            ),
            overlay_video_key=report_result.get("overlay_video_key"),
            processing_time_seconds=processing_time,
        )

        _update_analysis_status(
            analysis_id,
            STATUS_COMPLETED,
            completed_at=datetime.now(timezone.utc),
        )

        logger.info(
            "Pipeline completed for analysis_id=%s in %.2fs",
            analysis_id,
            processing_time,
        )

        return {
            "analysis_id": analysis_id,
            "status": STATUS_COMPLETED,
            "processing_time_seconds": processing_time,
        }

    except SoftTimeLimitExceeded:
        logger.error(
            "Pipeline timed out for analysis_id=%s (soft limit 600s)", analysis_id
        )
        _update_analysis_status(
            analysis_id,
            STATUS_FAILED,
            error_message="Analysis timed out (exceeded 600 second soft limit)",
            completed_at=datetime.now(timezone.utc),
        )
        return {
            "analysis_id": analysis_id,
            "status": STATUS_FAILED,
            "error_message": "Analysis timed out (exceeded 600 second soft limit)",
        }

    except Exception as exc:
        logger.error(
            "Pipeline failed for analysis_id=%s: %s",
            analysis_id,
            str(exc),
            exc_info=True,
        )
        _update_analysis_status(
            analysis_id,
            STATUS_FAILED,
            error_message=str(exc),
            completed_at=datetime.now(timezone.utc),
        )

        # Retry with exponential backoff if retries remain
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            "analysis_id": analysis_id,
            "status": STATUS_FAILED,
            "error_message": str(exc),
        }
    finally:
        # Clean up temp files
        if temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                logger.warning("Failed to clean up temp dir: %s", temp_dir)


# ============================================================================
# Pipeline step implementations
# ============================================================================


def _run_preprocessing(
    analysis_id: str, video_file_key: str | None,
    flip_horizontal: bool = False,
) -> dict[str, Any]:
    """Run video preprocessing step.

    Downloads video from S3, extracts frames, validates metadata,
    and runs quality checks. If flip_horizontal is True, frames are
    mirrored so a left-handed batter appears right-handed.

    Args:
        analysis_id: UUID of the analysis record.
        video_file_key: S3 key of the uploaded video file.
        flip_horizontal: If True, horizontally flip all frames (for LHB).

    Returns:
        Dictionary with preprocessing results.
    """
    logger.info("Preprocessing video for analysis_id=%s", analysis_id)

    if not video_file_key:
        raise ValueError("No video file key provided for preprocessing")

    try:
        import cv2

        from app.services.quality_checker import VideoQualityChecker
        from app.services.s3_client import S3Client
        from app.services.video_normalizer import normalize_frames_for_analysis
        from app.services.video_validator import extract_metadata

        # Download video from S3 to a temp file
        s3_client = S3Client()
        temp_video = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, prefix="myswing_video_"
        )
        temp_video_path = temp_video.name
        temp_video.close()

        s3_client._client.download_file(
            s3_client._bucket, video_file_key, temp_video_path
        )

        # Extract metadata
        metadata = extract_metadata(temp_video_path)
        original_fps = metadata.frame_rate if metadata.frame_rate > 0 else 30.0

        # Extract frames using OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        frames: list[np.ndarray] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()

        original_frame_count = len(frames)
        normalization = normalize_frames_for_analysis(
            frames,
            source_fps=original_fps,
        )
        frames = normalization.frames
        fps = normalization.fps

        # Horizontally flip frames for left-handed batters (Requirement 3.7)
        if flip_horizontal:
            logger.info("Flipping frames horizontally for LHB analysis_id=%s", analysis_id)
            frames = [cv2.flip(f, 1) for f in frames]

        frame_count = len(frames)

        # Save frames to temp directory for memory efficiency
        frames_dir = _save_frames_to_temp_dir(frames)
        del frames  # Free memory

        # Run quality check
        quality_checker = VideoQualityChecker()
        quality_result = quality_checker.check_quality(temp_video_path)

        # Clean up temp video file
        os.unlink(temp_video_path)

        return {
            "analysis_id": analysis_id,
            "video_file_key": video_file_key,
            "frames_extracted": True,
            "frames_dir": frames_dir,
            "video_path": temp_video_path,
            "fps": fps,
            "frame_count": frame_count,
            "video_width": normalization.video_width,
            "video_height": normalization.video_height,
            "original_fps": normalization.original_fps,
            "original_video_width": normalization.original_width,
            "original_video_height": normalization.original_height,
            "original_frame_count": original_frame_count,
            "normalization_applied": (
                normalization.crop_box is not None
                or len(normalization.sampled_frame_indices) != original_frame_count
            ),
            "normalization_target_fps": normalization.target_fps,
            "normalization_crop_box": normalization.crop_box,
            "normalization_sampled_frame_count": len(
                normalization.sampled_frame_indices
            ),
            "flip_horizontal": flip_horizontal,
            "quality_result": _serialize_dataclass(quality_result),
            "status": "completed",
        }

    except Exception as e:
        logger.warning(
            "Preprocessing with ML modules failed for analysis_id=%s: %s, "
            "falling back to stub result",
            analysis_id,
            str(e),
        )
        # Graceful fallback: return minimal result so pipeline can continue
        return {
            "analysis_id": analysis_id,
            "video_file_key": video_file_key,
            "frames_extracted": True,
            "frames_dir": None,
            "video_path": None,
            "fps": 30.0,
            "frame_count": 0,
            "video_width": 1920,
            "video_height": 1080,
            "original_fps": 30.0,
            "original_video_width": 1920,
            "original_video_height": 1080,
            "original_frame_count": 0,
            "normalization_applied": False,
            "normalization_target_fps": 30.0,
            "normalization_crop_box": None,
            "normalization_sampled_frame_count": 0,
            "quality_result": None,
            "status": "completed",
        }


def _run_pose_estimation(
    analysis_id: str, preprocessing_result: dict
) -> dict[str, Any]:
    """Run pose estimation step.

    Loads frames, runs PoseEstimator on each frame, applies PoseTracker
    for multi-frame tracking, and identifies the primary batter.

    Args:
        analysis_id: UUID of the analysis record.
        preprocessing_result: Output from preprocessing step.

    Returns:
        Dictionary with pose estimation results.
    """
    logger.info("Estimating pose for analysis_id=%s", analysis_id)

    try:
        from app.pipeline.batter_identifier import BatterIdentifier
        from app.pipeline.pose_estimator import PoseEstimator
        from app.pipeline.pose_tracker import PoseTracker

        frames_dir = preprocessing_result.get("frames_dir")
        if not frames_dir:
            return {
                "analysis_id": analysis_id,
                "pose_sequence": [],
                "status": "completed",
            }

        frames = _load_frames_from_temp_dir(frames_dir)
        if not frames:
            return {
                "analysis_id": analysis_id,
                "pose_sequence": [],
                "status": "completed",
            }

        # Create PoseEstimator with static_image_mode for batch processing
        estimator = PoseEstimator(
            min_confidence=0.5, static_image_mode=True
        )

        try:
            # Process each frame
            pose_results = []
            for i, frame in enumerate(frames):
                if estimator.is_available:
                    pose = estimator.process_frame(frame, frame_index=i)
                    pose_results.append(pose)
        finally:
            estimator.close()

        if not pose_results:
            return {
                "analysis_id": analysis_id,
                "pose_sequence": [],
                "status": "completed",
            }

        # Apply PoseTracker for multi-frame tracking + interpolation
        tracker = PoseTracker()
        tracked_results = tracker.track_across_frames(pose_results)

        # Filter out frames with no keypoints (can't analyze empty frames)
        tracked_results = [r for r in tracked_results if r.keypoints]

        if not tracked_results:
            return {
                "analysis_id": analysis_id,
                "pose_sequence": [],
                "status": "completed",
            }

        # Apply BatterIdentifier if multiple persons detected
        # (PoseEstimator returns single person, but future-proof)
        if len(tracked_results) > 1:
            # Only pass results with keypoints to BatterIdentifier
            results_with_keypoints = [r for r in tracked_results if r.keypoints]
            if results_with_keypoints:
                identifier = BatterIdentifier()
                # Use frame center as batting zone center
                primary = identifier.identify_primary_batter(
                    results_with_keypoints, batting_zone_center=(0.5, 0.5)
                )
                tracked_results = [
                    r for r in tracked_results
                    if r.person_id == primary.person_id
                ]

        # Serialize for inter-step transport
        serialized = _serialize_dataclass(tracked_results)

        return {
            "analysis_id": analysis_id,
            "pose_sequence": serialized,
            "status": "completed",
        }

    except Exception as e:
        logger.warning(
            "Pose estimation partially failed for analysis_id=%s: %s",
            analysis_id,
            str(e),
        )
        return {
            "analysis_id": analysis_id,
            "pose_sequence": [],
            "status": "partial_failure",
            "error": str(e),
        }


def _run_wrist_bat_estimation(
    analysis_id: str,
    pose_result: dict[str, Any],
    preprocessing_result: dict[str, Any],
    dominant_hand: str = "right",
) -> dict[str, Any]:
    """Estimate bat trajectory from wrist/elbow pose keypoints.

    Wrist-based estimation is the production bat trajectory path. It avoids the
    legacy object-detector path, which was previously run and then overwritten
    by this estimation step.
    """
    logger.info("Estimating bat trajectory from wrist keypoints for analysis_id=%s", analysis_id)

    try:
        pose_sequence_data = pose_result.get("pose_sequence", [])
        if not pose_sequence_data:
            logger.info(
                "No pose data available for wrist bat estimation, analysis_id=%s",
                analysis_id,
            )
            return {
                "analysis_id": analysis_id,
                "bat_trajectory": {},
                "status": "completed",
                "method": "wrist_estimation",
            }

        from app.pipeline.wrist_bat_estimator import WristBatEstimator

        pose_sequence = _deserialize_pose_sequence(pose_sequence_data)
        estimator = WristBatEstimator(dominant_hand=dominant_hand)

        video_width = preprocessing_result.get("video_width", 1920)
        video_height = preprocessing_result.get("video_height", 1080)

        estimated_trajectory = estimator.estimate_trajectory(
            pose_sequence,
            video_width=video_width,
            video_height=video_height,
        )

        return {
            "analysis_id": analysis_id,
            "bat_trajectory": _serialize_dataclass(estimated_trajectory),
            "status": "completed",
            "method": "wrist_estimation",
        }

    except Exception as e:
        logger.warning(
            "Wrist bat estimation failed for analysis_id=%s: %s",
            analysis_id,
            str(e),
        )
        return {
            "analysis_id": analysis_id,
            "bat_trajectory": {},
            "status": "partial_failure",
            "method": "wrist_estimation",
            "error": str(e),
        }


def _run_bat_detection(
    analysis_id: str, preprocessing_result: dict[str, Any]
) -> dict[str, Any]:
    """Legacy detector step retained as a no-op compatibility shim.

    The production pipeline uses ``_run_wrist_bat_estimation`` after pose
    estimation. This no-op preserves older unit tests or direct callers without
    importing detector-specific dependencies.
    """
    logger.info("Skipping legacy object-detector bat detection for analysis_id=%s", analysis_id)
    return {
        "analysis_id": analysis_id,
        "bat_trajectory": {},
        "status": "completed",
        "method": "skipped_legacy_detector",
    }


def _apply_wrist_bat_fallback(
    analysis_id: str,
    bat_result: dict[str, Any],
    pose_result: dict[str, Any],
    preprocessing_result: dict[str, Any],
    dominant_hand: str = "right",
) -> dict[str, Any]:
    """Backward-compatible wrapper for the wrist-primary bat estimation path."""
    if not pose_result.get("pose_sequence"):
        return bat_result
    return _run_wrist_bat_estimation(
        analysis_id,
        pose_result,
        preprocessing_result,
        dominant_hand=dominant_hand,
    )


def _get_bat_detection_point(det: Any) -> tuple[float, float]:
    """Get the most representative bat point for motion metrics.

    Prefer bat_head_position when available; otherwise fall back to center.
    Supports both dataclass objects and plain dict payloads.
    """
    if isinstance(det, dict):
        bat_head = det.get("bat_head_position")
        if isinstance(bat_head, (list, tuple)) and len(bat_head) >= 2:
            return float(bat_head[0]), float(bat_head[1])
        pos = det.get("position", (0.0, 0.0))
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return float(pos[0]), float(pos[1])
        return 0.0, 0.0

    bat_head = getattr(det, "bat_head_position", None)
    if isinstance(bat_head, (list, tuple)) and len(bat_head) >= 2:
        return float(bat_head[0]), float(bat_head[1])
    pos = getattr(det, "position", (0.0, 0.0))
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return float(pos[0]), float(pos[1])
    return 0.0, 0.0


def _detection_detected(det: Any) -> bool:
    if isinstance(det, dict):
        return bool(det.get("detected", False))
    return bool(getattr(det, "detected", False))


def _detection_frame_index(det: Any) -> int:
    if isinstance(det, dict):
        return int(det.get("frame_index", 0))
    return int(getattr(det, "frame_index", 0))


def _estimate_impact_frame_from_bat_speed(bat_trajectory: Any) -> tuple[int, float, str]:
    """Estimate impact frame from smoothed bat-speed profile.

    Returns:
        (impact_frame, confidence, method)
    """
    detections = sorted(
        [d for d in getattr(bat_trajectory, "detections", []) if _detection_detected(d)],
        key=_detection_frame_index,
    )
    if len(detections) < 2:
        return 0, 0.0, "insufficient_detections"

    frames: list[int] = []
    raw_speeds: list[float] = []
    for i in range(1, len(detections)):
        d1 = detections[i - 1]
        d2 = detections[i]
        p1 = _get_bat_detection_point(d1)
        p2 = _get_bat_detection_point(d2)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        raw_speeds.append(math.sqrt(dx * dx + dy * dy))
        frames.append(_detection_frame_index(d2))

    if not raw_speeds:
        return 0, 0.0, "no_speed_samples"

    # Median smoothing (window=3) to suppress 1-frame spikes.
    smoothed = []
    for i in range(len(raw_speeds)):
        win = raw_speeds[max(0, i - 1): min(len(raw_speeds), i + 2)]
        smoothed.append(statistics.median(win))

    peak_indices = []
    for i, v in enumerate(smoothed):
        left = smoothed[i - 1] if i > 0 else float("-inf")
        right = smoothed[i + 1] if i + 1 < len(smoothed) else float("-inf")
        if v >= left and v >= right and v > 0:
            peak_indices.append(i)

    if not peak_indices:
        peak_indices = [max(range(len(smoothed)), key=lambda idx: smoothed[idx])]

    # In full swing sequences, early/mid-swing acceleration bursts can be faster
    # than the actual hitting-zone segment. Prefer local peaks in the latter
    # portion of the detected trajectory; short synthetic/unit sequences keep
    # the unrestricted behavior.
    candidate_peak_indices = peak_indices
    if len(smoothed) >= 40 and frames:
        start_frame = frames[0]
        end_frame = frames[-1]
        duration = end_frame - start_frame
        min_impact_frame = start_frame + int(duration * 0.55)
        max_impact_frame = start_frame + int(duration * 0.70)
        hitting_zone_peak_indices = [
            idx for idx in peak_indices
            if min_impact_frame <= frames[idx] <= max_impact_frame
        ]
        if hitting_zone_peak_indices:
            candidate_peak_indices = hitting_zone_peak_indices

    candidate_peak_indices.sort(key=lambda idx: smoothed[idx], reverse=True)
    best_idx = candidate_peak_indices[0]
    impact_frame = frames[best_idx]
    best_speed = smoothed[best_idx]
    second_speed = smoothed[candidate_peak_indices[1]] if len(candidate_peak_indices) > 1 else 0.0

    dominance = (best_speed - second_speed) / best_speed if best_speed > 1e-9 else 0.0
    sample_factor = min(1.0, len(smoothed) / 8.0)
    confidence = max(0.05, min(0.99, (0.4 + 0.6 * max(0.0, dominance)) * sample_factor))

    return impact_frame, float(confidence), "smoothed_peak_speed"


def _estimate_phases_from_speed(
    bat_trajectory: Any,
    pose_sequence: list,
    fps: float,
) -> tuple[dict, dict]:
    """Estimate swing phases from bat speed pattern when classifier fails.

    Divides the swing into 6 phases based on bat speed profile:
    - Stance: first 15% (minimal movement)
    - Load: 15-30% (building energy)
    - Stride: 30-50% (weight transfer)
    - Rotation: 50-75% (acceleration)
    - Impact: around peak speed frame (±2 frames)
    - Follow-through: after impact to end

    Args:
        bat_trajectory: BatTrajectory with detections.
        pose_sequence: List of PoseResult.
        fps: Video frame rate.

    Returns:
        Tuple of (phases_dict, durations_dict) with string keys.
    """
    detections = bat_trajectory.detections
    if not detections:
        return {}, {}

    total_frames = len(detections)
    if total_frames < 10:
        return {}, {}

    peak_frame, _, _ = _estimate_impact_frame_from_bat_speed(bat_trajectory)
    if peak_frame <= 0:
        return {}, {}
    start_frame = detections[0].frame_index
    end_frame = detections[-1].frame_index
    total_duration = end_frame - start_frame

    if total_duration <= 0:
        return {}, {}

    # Estimate phase boundaries relative to the speed-derived peak while keeping
    # ranges chronological. The peak can appear earlier than the generic 50%
    # stride boundary on noisy wrist-estimated trajectories, so cap earlier
    # phases to leave room for rotation and impact instead of producing
    # inverted ranges such as impact [59, 56].
    stance_end = start_frame + int(total_duration * 0.15)
    load_end = start_frame + int(total_duration * 0.30)
    stride_end = start_frame + int(total_duration * 0.50)

    impact_start = max(start_frame, min(peak_frame, end_frame))
    impact_end = min(impact_start + 2, end_frame)

    rotation_end = max(start_frame, impact_start - 1)
    stride_end = min(stride_end, max(start_frame, rotation_end - 1))
    load_end = min(load_end, max(start_frame, stride_end - 1))
    stance_end = min(stance_end, max(start_frame, load_end - 1))

    load_start = min(stance_end + 1, load_end)
    stride_start = min(load_end + 1, stride_end)
    rotation_start = min(stride_end + 1, rotation_end)
    follow_start = min(impact_end + 1, end_frame)

    phases = {
        "stance": [start_frame, stance_end],
        "load": [load_start, load_end],
        "stride": [stride_start, stride_end],
        "rotation": [rotation_start, rotation_end],
        "impact": [impact_start, impact_end],
        "follow_through": [follow_start, end_frame],
    }

    # Calculate durations in ms
    durations = {}
    for phase_name, (s, e) in phases.items():
        duration_ms = (e - s) / fps * 1000.0
        durations[phase_name] = max(0.0, duration_ms)

    return phases, durations


def _run_swing_classification(
    analysis_id: str,
    pose_result: dict,
    bat_result: dict,
    fps: float,
    batting_direction: str = "right",
) -> dict[str, Any]:
    """Run swing phase classification step.

    Deserializes pose_sequence and bat_trajectory, then classifies
    swing into 6 phases using SwingPhaseClassifier.

    Args:
        analysis_id: UUID of the analysis record.
        pose_result: Output from pose estimation.
        bat_result: Output from bat detection.
        fps: Video frame rate.

    Returns:
        Dictionary with swing phase classification results.
    """
    logger.info("Classifying swing phases for analysis_id=%s", analysis_id)

    try:
        from app.pipeline.swing_classifier import SwingPhaseClassifier

        # Deserialize inputs
        pose_sequence_data = pose_result.get("pose_sequence", [])
        bat_trajectory_data = bat_result.get("bat_trajectory", {})

        if not pose_sequence_data:
            return {
                "analysis_id": analysis_id,
                "phases": {},
                "transitions": [],
                "phase_durations_ms": {},
                "anomalies": [],
                "status": "completed",
            }

        pose_sequence = _deserialize_pose_sequence(pose_sequence_data)
        bat_trajectory = _deserialize_bat_trajectory(bat_trajectory_data)

        # Create classifier (batting_direction from user profile if available)
        classifier = SwingPhaseClassifier(batting_direction=batting_direction)

        # Classify phases
        phase_result = classifier.classify_phases(
            pose_sequence, bat_trajectory, fps
        )

        # Serialize result
        serialized = _serialize_dataclass(phase_result)

        # Fallback: if no phases or core middle phases are missing, estimate from
        # bat speed pattern. Real sample analyses can detect stance/load/impact
        # while missing stride/rotation entirely; treating that partial result as
        # complete makes downstream metrics unstable and leaves phase durations n/a.
        phases_dict = serialized.get("phases", {})
        expected_phases = {
            "stance",
            "load",
            "stride",
            "rotation",
            "impact",
            "follow_through",
        }
        missing_core_phases = expected_phases.difference(phases_dict)
        if (not phases_dict or missing_core_phases) and bat_trajectory.detections:
            phases_dict, durations_dict = _estimate_phases_from_speed(
                bat_trajectory, pose_sequence, fps
            )
            if phases_dict:
                serialized["phases"] = phases_dict
                serialized["phase_durations_ms"] = durations_dict
                serialized["anomalies"] = []
                logger.info(
                    "Used speed-based phase estimation fallback for analysis_id=%s",
                    analysis_id,
                )

        return {
            "analysis_id": analysis_id,
            "phases": serialized.get("phases", {}),
            "transitions": serialized.get("transitions", []),
            "phase_durations_ms": serialized.get("phase_durations_ms", {}),
            "anomalies": serialized.get("anomalies", []),
            "status": "completed",
        }

    except Exception as e:
        logger.warning(
            "Swing classification partially failed for analysis_id=%s: %s",
            analysis_id,
            str(e),
        )
        return {
            "analysis_id": analysis_id,
            "phases": {},
            "transitions": [],
            "phase_durations_ms": {},
            "anomalies": [],
            "status": "partial_failure",
            "error": str(e),
        }


def _run_biomechanics_analysis(
    analysis_id: str,
    pose_result: dict,
    bat_result: dict,
    swing_phases_result: dict,
    user_profile: dict | None,
    fps: float,
    preprocessing_result: dict | None = None,
) -> dict[str, Any]:
    """Run biomechanics analysis step.

    Creates BiomechanicsOrchestrator and runs full analysis with
    pose_sequence, bat_trajectory, user calibration data, and swing phases.

    Args:
        analysis_id: UUID of the analysis record.
        pose_result: Output from pose estimation.
        bat_result: Output from bat detection.
        swing_phases_result: Output from swing classification.
        user_profile: User profile data for calibration.
        fps: Video frame rate.
        preprocessing_result: Output from preprocessing step (contains video dimensions).

    Returns:
        Dictionary with biomechanics analysis results.
    """
    logger.info("Analyzing biomechanics for analysis_id=%s", analysis_id)

    unmeasurable_metrics: list[dict] = []

    try:
        from app.pipeline.biomechanics_analyzer import BiomechanicsOrchestrator

        # Deserialize inputs
        pose_sequence_data = pose_result.get("pose_sequence", [])
        bat_trajectory_data = bat_result.get("bat_trajectory", {})

        if not pose_sequence_data:
            unmeasurable_metrics.append({
                "metric_name": "all",
                "reason": "No pose data available",
            })
            return {
                "analysis_id": analysis_id,
                "bat_speed": None,
                "attack_angle": None,
                "kinematic_chain": None,
                "rotation": None,
                "hand_path_efficiency": None,
                "analysis_metadata": _build_analysis_metadata(preprocessing_result),
                "unmeasurable_metrics": unmeasurable_metrics,
                "status": "completed",
            }

        pose_sequence = _deserialize_pose_sequence(pose_sequence_data)
        bat_trajectory = _deserialize_bat_trajectory(bat_trajectory_data)

        # Extract user calibration data
        user_height_cm = 175.0  # default
        bat_length_meters = 0.86  # default ~34 inches
        if user_profile:
            user_height_cm = user_profile.get("height", 175.0) or 175.0
            bat_length_raw = user_profile.get("bat_length", 34.0) or 34.0
            # Convert bat_length from inches to meters if needed
            bat_length_meters = bat_length_raw * 0.0254 if bat_length_raw > 2 else bat_length_raw

        # Determine impact frame from swing phases
        phases = swing_phases_result.get("phases", {})
        impact_frame = 0
        # Look for impact phase — use the midpoint of the phase range
        # since impact is an interval, not a single frame.
        impact_phase = phases.get("impact", phases.get("IMPACT"))
        if isinstance(impact_phase, (list, tuple)) and len(impact_phase) >= 2:
            impact_frame = int((impact_phase[0] + impact_phase[1]) // 2)
        elif isinstance(impact_phase, (list, tuple)) and len(impact_phase) >= 1:
            impact_frame = int(impact_phase[0])
        elif isinstance(impact_phase, dict):
            start = impact_phase.get("start", 0)
            end = impact_phase.get("end", start)
            impact_frame = int((start + end) // 2) if end > start else int(start)

        logger.info(
            "Impact frame determination: impact_phase=%s, impact_frame=%d for analysis_id=%s",
            impact_phase, impact_frame, analysis_id,
        )

        impact_frame_method = "phase_midpoint"
        impact_frame_confidence = 0.9

        # If impact_frame is 0 (no explicit impact phase found),
        # estimate from smoothed bat-speed profile and expose confidence.
        if impact_frame == 0:
            impact_frame_method = "smoothed_peak_speed"
            impact_frame_confidence = 0.0
            if bat_trajectory.detections and len(bat_trajectory.detections) > 1:
                est_frame, est_confidence, est_method = _estimate_impact_frame_from_bat_speed(
                    bat_trajectory
                )
                if est_frame > 0:
                    impact_frame = est_frame
                    impact_frame_confidence = est_confidence
                    impact_frame_method = est_method
                    logger.info(
                        "Estimated impact_frame=%d (confidence=%.3f, method=%s) for analysis_id=%s",
                        impact_frame,
                        impact_frame_confidence,
                        impact_frame_method,
                        analysis_id,
                    )

        # Build swing_phases dict for orchestrator
        swing_phases_dict = {}
        rotation_phase = phases.get("rotation", phases.get("ROTATION"))
        if isinstance(rotation_phase, (list, tuple)) and len(rotation_phase) >= 2:
            swing_phases_dict["rotation_start_frame"] = int(rotation_phase[0])
            swing_phases_dict["rotation_end_frame"] = int(rotation_phase[1])

        # Extract stride phase for stride/cog metrics
        stride_phase = phases.get("stride", phases.get("STRIDE"))
        if isinstance(stride_phase, (list, tuple)) and len(stride_phase) >= 2:
            swing_phases_dict["stride_start_frame"] = int(stride_phase[0])
            swing_phases_dict["stride_end_frame"] = int(stride_phase[1])

        # Extract load phase for cog_drop / spine_angle
        load_phase = phases.get("load", phases.get("LOAD"))
        if isinstance(load_phase, (list, tuple)) and len(load_phase) >= 2:
            swing_phases_dict["load_frame"] = int(load_phase[0])

        # Fallback: estimate stride/load from frame proportions when classifier fails
        if pose_sequence_data:
            total = len(pose_sequence_data)
            if "stride_start_frame" not in swing_phases_dict and total > 6:
                swing_phases_dict["stride_start_frame"] = int(total * 0.15)
                swing_phases_dict["stride_end_frame"] = int(total * 0.35)
            if "load_frame" not in swing_phases_dict and total > 3:
                swing_phases_dict["load_frame"] = int(total * 0.10)

        # If no rotation phase found, use middle portion of swing as fallback
        if not swing_phases_dict.get("rotation_start_frame") and pose_sequence_data:
            total_frames = len(pose_sequence_data)
            if total_frames > 4:
                # Use middle 60% of frames as rotation window
                start_pct = 0.2
                end_pct = 0.8
                swing_phases_dict["rotation_start_frame"] = int(total_frames * start_pct)
                swing_phases_dict["rotation_end_frame"] = int(total_frames * end_pct)
                logger.info(
                    "No rotation phase found, using frames %d-%d "
                    "as rotation window for analysis_id=%s",
                    swing_phases_dict["rotation_start_frame"],
                    swing_phases_dict["rotation_end_frame"],
                    analysis_id,
                )

        # Ensure start_frame and end_frame are set for kinematic chain analysis
        if "start_frame" not in swing_phases_dict and pose_sequence_data:
            swing_phases_dict["start_frame"] = 0
            swing_phases_dict["end_frame"] = len(pose_sequence_data) - 1

        # Create orchestrator and run analysis
        orchestrator = BiomechanicsOrchestrator()

        # Extract video dimensions for aspect ratio correction. The production
        # wrist-based estimator emits normalized coordinates (0-1), so the
        # biomechanics layer needs frame dimensions to convert direction and
        # distance consistently.
        video_width = 1
        video_height = 1
        if bat_result.get("method") == "wrist_estimation":
            if preprocessing_result:
                video_width = preprocessing_result.get("video_width", 1920)
                video_height = preprocessing_result.get("video_height", 1080)
            else:
                video_width = 1920
                video_height = 1080

        # Analysis currently runs in canonical RHB space after optional flip.
        # Keep explicit for call-site clarity.
        canonical_batting_direction = "right"

        biomechanics_result = orchestrator.analyze(
            pose_sequence=pose_sequence,
            bat_trajectory=bat_trajectory,
            user_height_cm=user_height_cm,
            bat_length_meters=bat_length_meters,
            impact_frame=impact_frame,
            swing_phases=swing_phases_dict,
            fps=fps,
            video_width=video_width,
            video_height=video_height,
            batting_direction=canonical_batting_direction,
        )

        # Serialize result
        serialized = _serialize_dataclass(biomechanics_result)
        serialized["analysis_id"] = analysis_id
        serialized["impact_frame"] = int(impact_frame)
        serialized["impact_frame_method"] = impact_frame_method
        serialized["impact_frame_confidence"] = float(impact_frame_confidence)
        serialized["analysis_metadata"] = _build_analysis_metadata(preprocessing_result)
        serialized["status"] = "completed"
        return serialized

    except Exception as e:
        logger.warning(
            "Biomechanics analysis partially failed for analysis_id=%s: %s",
            analysis_id,
            str(e),
        )
        unmeasurable_metrics.append({
            "metric_name": "all",
            "reason": str(e),
        })
        return {
            "analysis_id": analysis_id,
            "bat_speed": None,
            "attack_angle": None,
            "kinematic_chain": None,
            "rotation": None,
            "hand_path_efficiency": None,
            "analysis_metadata": _build_analysis_metadata(preprocessing_result),
            "unmeasurable_metrics": unmeasurable_metrics,
            "status": "partial_failure",
            "error": str(e),
        }


def _run_swing_evaluation(
    analysis_id: str,
    biomechanics_result: dict,
    bat_result: dict,
    pose_result: dict,
    swing_phases_result: dict,
    user_profile: dict | None,
    fps: float,
) -> dict[str, Any]:
    """Run swing evaluation step.

    Creates ReferenceComparator, ModernPrinciplesEvaluator, ImprovementRanker,
    and DrillRecommender to evaluate the swing against references and principles.

    Args:
        analysis_id: UUID of the analysis record.
        biomechanics_result: Output from biomechanics analysis.
        bat_result: Output from bat detection.
        pose_result: Output from pose estimation.
        swing_phases_result: Output from swing classification.
        user_profile: User profile data for reference comparison.
        fps: Video frame rate.

    Returns:
        Dictionary with evaluation results.
    """
    logger.info("Evaluating swing for analysis_id=%s", analysis_id)

    try:
        # Reconstruct BiomechanicsResult from dict for evaluators
        from app.models.biomechanics import (
            BatSpeedResult,
            BiomechanicsResult,
            KinematicChainResult,
            LaunchAngleResult,
            RotationResult,
        )
        from app.pipeline.report_generator import DrillRecommender
        from app.pipeline.swing_evaluator import (
            ImprovementRanker,
            ModernPrinciplesEvaluator,
            ReferenceComparator,
            WeightTransferAnalyzer,
        )

        # Rebuild BiomechanicsResult object from serialized dict
        bio = BiomechanicsResult()
        if biomechanics_result.get("bat_speed"):
            bs = biomechanics_result["bat_speed"]
            bio.bat_speed = BatSpeedResult(
                speed_kmh=bs["speed_kmh"],
                precision=bs["precision"],
                measurement_frame=bs["measurement_frame"],
            )
        if biomechanics_result.get("attack_angle"):
            aa = biomechanics_result["attack_angle"]
            bio.attack_angle = LaunchAngleResult(
                angle_degrees=aa["angle_degrees"],
                precision=aa["precision"],
                impact_frame=aa.get("impact_frame", 0),
            )
        if biomechanics_result.get("kinematic_chain"):
            kc = biomechanics_result["kinematic_chain"]
            bio.kinematic_chain = KinematicChainResult(
                sequence_correct=kc.get("sequence_correct", False),
                timing_gaps_ms=kc.get("timing_gaps_ms", {}),
            )
        if biomechanics_result.get("rotation"):
            rot = biomechanics_result["rotation"]
            bio.rotation = RotationResult(
                hip_rotation_speed_dps=rot.get("hip_rotation_speed_dps", 0),
                shoulder_rotation_speed_dps=rot.get("shoulder_rotation_speed_dps", 0),
                hip_shoulder_separation_degrees=rot.get("hip_shoulder_separation_degrees", 0),
                rotation_phase_start_frame=rot.get("rotation_phase_start_frame", 0),
                rotation_phase_end_frame=rot.get("rotation_phase_end_frame", 0),
            )
        if biomechanics_result.get("hand_path_efficiency") is not None:
            bio.hand_path_efficiency = biomechanics_result["hand_path_efficiency"]

        # Swing quality metrics (may be None)
        for field in ["stride_length_cm", "cog_sway_cm", "cog_drop_cm",
                      "head_stability_cm", "front_knee_flexion_degrees",
                      "spine_angle_degrees"]:
            if biomechanics_result.get(field) is not None:
                setattr(bio, field, biomechanics_result[field])

        # Reference comparison
        level = user_profile.get("level") if user_profile else None
        age_group = user_profile.get("age_group") if user_profile else None

        comparator = ReferenceComparator()
        evaluations = comparator.compare_with_reference(
            bio, level=level, age_group=age_group
        )

        # Modern principles evaluation
        bat_trajectory = _deserialize_bat_trajectory(
            bat_result.get("bat_trajectory", {})
        )
        principles_evaluator = ModernPrinciplesEvaluator()
        principles_evaluation = principles_evaluator.evaluate_principles(
            bio, bat_trajectory, fps
        )

        # Weight transfer analysis
        pose_sequence_data = pose_result.get("pose_sequence", [])
        weight_transfer_result = {}
        if pose_sequence_data:
            from app.models.swing import SwingPhaseResult as SwingPhaseResultModel

            pose_sequence = _deserialize_pose_sequence(pose_sequence_data)
            # Rebuild SwingPhaseResult for weight transfer analyzer
            wt_analyzer = WeightTransferAnalyzer()
            # Create a minimal SwingPhaseResult from the dict
            spr = SwingPhaseResultModel()
            # We pass the raw swing_phases_result phases
            # The analyzer needs SwingPhase enum keys
            from app.models.enums import SwingPhase

            phases_raw = swing_phases_result.get("phases", {})
            for phase_name, frame_range in phases_raw.items():
                try:
                    phase_enum = SwingPhase(phase_name)
                    if isinstance(frame_range, (list, tuple)) and len(frame_range) >= 2:
                        spr.phases[phase_enum] = (int(frame_range[0]), int(frame_range[1]))
                except (ValueError, TypeError):
                    pass

            wt_result = wt_analyzer.analyze_weight_transfer(
                pose_sequence, spr, fps
            )
            weight_transfer_result = _serialize_dataclass(wt_result)

        # Rank improvements
        ranker = ImprovementRanker()
        improvements = ranker.rank_improvements(evaluations)

        # Recommend drills
        recommender = DrillRecommender()
        drill_recommendations = recommender.recommend_drills(improvements)

        return {
            "analysis_id": analysis_id,
            "evaluations": _serialize_dataclass(evaluations),
            "improvements": _serialize_dataclass(improvements),
            "drill_recommendations": _serialize_dataclass(drill_recommendations),
            "principles_evaluation": principles_evaluation,
            "weight_transfer": weight_transfer_result,
            "status": "completed",
        }

    except Exception as e:
        logger.warning(
            "Swing evaluation partially failed for analysis_id=%s: %s",
            analysis_id,
            str(e),
        )
        return {
            "analysis_id": analysis_id,
            "evaluations": [],
            "improvements": [],
            "drill_recommendations": [],
            "principles_evaluation": {},
            "weight_transfer": {},
            "status": "partial_failure",
            "error": str(e),
        }


def _run_report_generation(
    analysis_id: str,
    evaluation_result: dict,
    biomechanics_result: dict,
    swing_phases_result: dict,
    preprocessing_result: dict,
    pose_result: dict,
    bat_result: dict,
) -> dict[str, Any]:
    """Run report generation step.

    Creates OverlayRenderer, renders overlay video from frames + pose + bat
    trajectory, uploads to S3, and builds metrics table.

    Args:
        analysis_id: UUID of the analysis record.
        evaluation_result: Output from swing evaluation.
        biomechanics_result: Output from biomechanics analysis.
        swing_phases_result: Output from swing classification.
        preprocessing_result: Output from preprocessing.
        pose_result: Output from pose estimation.
        bat_result: Output from bat detection.

    Returns:
        Dictionary with report generation results.
    """
    logger.info("Generating report for analysis_id=%s", analysis_id)

    try:
        from app.pipeline.report_generator import (
            MetricsTableBuilder,
            OverlayRenderer,
        )
        from app.services.s3_client import S3Client

        overlay_video_key = None
        metrics_table: list = []
        comparison_view = None

        # Render overlay video if frames are available
        frames_dir = preprocessing_result.get("frames_dir")
        flip_horizontal = preprocessing_result.get("flip_horizontal", False)
        if frames_dir and os.path.isdir(frames_dir):
            frames = _load_frames_from_temp_dir(frames_dir)
            if frames:
                pose_sequence_data = pose_result.get("pose_sequence", [])
                bat_trajectory_data = bat_result.get("bat_trajectory", {})

                pose_sequence = _deserialize_pose_sequence(pose_sequence_data)
                bat_trajectory = _deserialize_bat_trajectory(bat_trajectory_data)

                # If frames were flipped for LHB analysis, flip everything back
                # for overlay rendering so the video appears in original orientation.
                if flip_horizontal:
                    import cv2
                    frames = [cv2.flip(f, 1) for f in frames]
                    frame_w = frames[0].shape[1] if frames else 1
                    # Mirror pose keypoints: x' = 1 - x (normalized coords)
                    for pose in pose_sequence:
                        for kp in pose.keypoints:
                            kp.x = 1.0 - kp.x
                    # Mirror bat positions: x' = width - x (pixel coords)
                    for det in bat_trajectory.detections:
                        if det.detected:
                            x, y = det.position
                            det.position = (frame_w - x, y)

                fps = preprocessing_result.get("fps", 30.0)

                # Render overlay video to temp file
                renderer = OverlayRenderer()
                temp_overlay = tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False, prefix="myswing_overlay_"
                )
                temp_overlay_path = temp_overlay.name
                temp_overlay.close()

                renderer.render_overlay_video(
                    frames=frames,
                    pose_sequence=pose_sequence,
                    bat_trajectory=bat_trajectory,
                    output_path=temp_overlay_path,
                    fps=fps,
                )

                # Upload overlay to S3
                try:
                    s3_client = S3Client()
                    overlay_video_key = renderer._upload_to_s3(
                        temp_overlay_path, s3_client
                    )
                except Exception as upload_err:
                    logger.warning(
                        "Failed to upload overlay video: %s", str(upload_err)
                    )
                finally:
                    if os.path.exists(temp_overlay_path):
                        os.unlink(temp_overlay_path)

        # Build metrics table from evaluations
        evaluations_data = evaluation_result.get("evaluations", [])
        if evaluations_data:
            from app.models.enums import MetricRating
            from app.models.evaluation import MetricEvaluation

            # Reconstruct MetricEvaluation objects for table builder
            metric_evals = []
            for ev in evaluations_data:
                if isinstance(ev, dict):
                    try:
                        rating_val = ev.get("rating", "within_range")
                        if isinstance(rating_val, str):
                            rating = MetricRating(rating_val)
                        else:
                            rating = rating_val
                        metric_evals.append(MetricEvaluation(
                            metric_name=ev["metric_name"],
                            measured_value=ev["measured_value"],
                            unit=ev.get("unit", ""),
                            reference_min=ev.get("reference_min", 0),
                            reference_max=ev.get("reference_max", 0),
                            deviation_percent=ev.get("deviation_percent", 0),
                            rating=rating,
                            color_code=ev.get("color_code", "green"),
                        ))
                    except (KeyError, ValueError):
                        continue

            if metric_evals:
                table_builder = MetricsTableBuilder()
                metrics_table = table_builder.build_metrics_table(metric_evals)

        return {
            "analysis_id": analysis_id,
            "overlay_video_key": overlay_video_key,
            "metrics_table": metrics_table,
            "comparison_view": comparison_view,
            "trend_data": None,
            "status": "completed",
        }

    except Exception as e:
        logger.warning(
            "Report generation partially failed for analysis_id=%s: %s",
            analysis_id,
            str(e),
        )
        return {
            "analysis_id": analysis_id,
            "overlay_video_key": None,
            "metrics_table": [],
            "comparison_view": None,
            "trend_data": None,
            "status": "partial_failure",
            "error": str(e),
        }


# ============================================================================
# Individual Celery task wrappers
# ============================================================================


@celery_app.task(
    bind=True,
    name="app.tasks.pipeline.preprocess_video_task",
    max_retries=DEFAULT_RETRY_POLICY["max_retries"],
    retry_backoff=DEFAULT_RETRY_POLICY["retry_backoff"],
    retry_backoff_max=DEFAULT_RETRY_POLICY["retry_backoff_max"],
    retry_jitter=DEFAULT_RETRY_POLICY["retry_jitter"],
)
def preprocess_video_task(
    self, analysis_id: str, video_file_key: str
) -> dict[str, Any]:
    """Preprocess uploaded video: extract frames, validate metadata, check quality.

    Args:
        analysis_id: UUID of the analysis record.
        video_file_key: S3 key of the uploaded video file.

    Returns:
        Dictionary with extracted frame paths and video metadata.
    """
    logger.info(
        "Preprocessing video for analysis_id=%s, file_key=%s",
        analysis_id,
        video_file_key,
    )
    return _run_preprocessing(analysis_id, video_file_key)


@celery_app.task(
    bind=True,
    name="app.tasks.pipeline.estimate_pose_task",
    max_retries=DEFAULT_RETRY_POLICY["max_retries"],
    retry_backoff=DEFAULT_RETRY_POLICY["retry_backoff"],
    retry_backoff_max=DEFAULT_RETRY_POLICY["retry_backoff_max"],
    retry_jitter=DEFAULT_RETRY_POLICY["retry_jitter"],
)
def estimate_pose_task(
    self, analysis_id: str, preprocessing_result: dict
) -> dict[str, Any]:
    """Estimate body pose keypoints from video frames using MediaPipe.

    Args:
        analysis_id: UUID of the analysis record.
        preprocessing_result: Output from preprocess_video_task.

    Returns:
        Dictionary with pose estimation results (keypoints per frame).
    """
    logger.info("Estimating pose for analysis_id=%s", analysis_id)
    return _run_pose_estimation(analysis_id, preprocessing_result)


@celery_app.task(
    bind=True,
    name="app.tasks.pipeline.detect_bat_task",
    max_retries=DEFAULT_RETRY_POLICY["max_retries"],
    retry_backoff=DEFAULT_RETRY_POLICY["retry_backoff"],
    retry_backoff_max=DEFAULT_RETRY_POLICY["retry_backoff_max"],
    retry_jitter=DEFAULT_RETRY_POLICY["retry_jitter"],
)
def detect_bat_task(
    self, analysis_id: str, preprocessing_result: dict
) -> dict[str, Any]:
    """Legacy object-detector task retained for compatibility.

    The full production pipeline estimates bat trajectory from pose keypoints
    after ``estimate_pose_task``. This standalone task has no pose input, so it
    intentionally delegates to the no-op legacy shim.
    """
    logger.info("Skipping legacy detector task for analysis_id=%s", analysis_id)
    return _run_bat_detection(analysis_id, preprocessing_result)


@celery_app.task(
    bind=True,
    name="app.tasks.pipeline.classify_swing_task",
    max_retries=DEFAULT_RETRY_POLICY["max_retries"],
    retry_backoff=DEFAULT_RETRY_POLICY["retry_backoff"],
    retry_backoff_max=DEFAULT_RETRY_POLICY["retry_backoff_max"],
    retry_jitter=DEFAULT_RETRY_POLICY["retry_jitter"],
)
def classify_swing_task(
    self, analysis_id: str, pose_result: dict, bat_result: dict
) -> dict[str, Any]:
    """Classify swing into 6 phases based on pose and bat trajectory.

    Args:
        analysis_id: UUID of the analysis record.
        pose_result: Output from estimate_pose_task.
        bat_result: Output from detect_bat_task.

    Returns:
        Dictionary with swing phase classification results.
    """
    logger.info("Classifying swing phases for analysis_id=%s", analysis_id)
    analysis_data = _get_analysis_data(analysis_id) or {}
    fps = analysis_data.get("video_fps", 30.0)
    return _run_swing_classification(
        analysis_id,
        pose_result,
        bat_result,
        fps,
        # Wrapper path is canonical RHB unless future metadata says otherwise.
        batting_direction="right",
    )


@celery_app.task(
    bind=True,
    name="app.tasks.pipeline.analyze_biomechanics_task",
    max_retries=DEFAULT_RETRY_POLICY["max_retries"],
    retry_backoff=DEFAULT_RETRY_POLICY["retry_backoff"],
    retry_backoff_max=DEFAULT_RETRY_POLICY["retry_backoff_max"],
    retry_jitter=DEFAULT_RETRY_POLICY["retry_jitter"],
)
def analyze_biomechanics_task(
    self,
    analysis_id: str,
    pose_result: dict,
    bat_result: dict,
    swing_phases: dict,
    preprocessing_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform biomechanical analysis: bat speed, launch angle, kinematic chain.

    Args:
        analysis_id: UUID of the analysis record.
        pose_result: Output from estimate_pose_task.
        bat_result: Output from detect_bat_task.
        swing_phases: Output from classify_swing_task.
        preprocessing_result: Optional preprocessing metadata with frame/video dimensions.

    Returns:
        Dictionary with biomechanics analysis results.
    """
    logger.info("Analyzing biomechanics for analysis_id=%s", analysis_id)
    analysis_data = _get_analysis_data(analysis_id) or {}
    user_profile = analysis_data.get("user_profile")
    fps = analysis_data.get("video_fps", 30.0)

    if preprocessing_result is None:
        preprocessing_result = {
            "video_width": analysis_data.get("video_width", 1920),
            "video_height": analysis_data.get("video_height", 1080),
        }

    return _run_biomechanics_analysis(
        analysis_id,
        pose_result,
        bat_result,
        swing_phases,
        user_profile,
        fps,
        preprocessing_result,
    )


@celery_app.task(
    bind=True,
    name="app.tasks.pipeline.evaluate_swing_task",
    max_retries=DEFAULT_RETRY_POLICY["max_retries"],
    retry_backoff=DEFAULT_RETRY_POLICY["retry_backoff"],
    retry_backoff_max=DEFAULT_RETRY_POLICY["retry_backoff_max"],
    retry_jitter=DEFAULT_RETRY_POLICY["retry_jitter"],
)
def evaluate_swing_task(
    self, analysis_id: str, biomechanics_result: dict, user_profile: dict
) -> dict[str, Any]:
    """Evaluate swing against modern hitting principles and reference data.

    Args:
        analysis_id: UUID of the analysis record.
        biomechanics_result: Output from analyze_biomechanics_task.
        user_profile: User profile data for reference comparison.

    Returns:
        Dictionary with swing evaluation results and improvement areas.
    """
    logger.info("Evaluating swing for analysis_id=%s", analysis_id)
    analysis_data = _get_analysis_data(analysis_id) or {}
    fps = analysis_data.get("video_fps", 30.0)
    return _run_swing_evaluation(
        analysis_id, biomechanics_result, {}, {}, {}, user_profile, fps
    )


@celery_app.task(
    bind=True,
    name="app.tasks.pipeline.generate_report_task",
    max_retries=DEFAULT_RETRY_POLICY["max_retries"],
    retry_backoff=DEFAULT_RETRY_POLICY["retry_backoff"],
    retry_backoff_max=DEFAULT_RETRY_POLICY["retry_backoff_max"],
    retry_jitter=DEFAULT_RETRY_POLICY["retry_jitter"],
)
def generate_report_task(
    self, analysis_id: str, evaluation_result: dict,
    biomechanics_result: dict
) -> dict[str, Any]:
    """Generate the final analysis report with overlay video and metrics.

    Args:
        analysis_id: UUID of the analysis record.
        evaluation_result: Output from evaluate_swing_task.
        biomechanics_result: Output from analyze_biomechanics_task.

    Returns:
        Dictionary with report generation results.
    """
    logger.info("Generating report for analysis_id=%s", analysis_id)
    return _run_report_generation(
        analysis_id, evaluation_result, biomechanics_result, {}, {}, {}, {}
    )
