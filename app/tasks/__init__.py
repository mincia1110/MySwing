"""Celery task definitions for the swing analysis pipeline."""

from app.tasks.pipeline import (
    analyze_biomechanics_task,
    analyze_swing_task,
    classify_swing_task,
    detect_bat_task,
    estimate_pose_task,
    evaluate_swing_task,
    generate_report_task,
    preprocess_video_task,
)

__all__ = [
    "analyze_swing_task",
    "preprocess_video_task",
    "estimate_pose_task",
    "detect_bat_task",
    "classify_swing_task",
    "analyze_biomechanics_task",
    "evaluate_swing_task",
    "generate_report_task",
]
