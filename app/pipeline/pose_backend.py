"""Factory for configured pose-estimation backends."""

from __future__ import annotations

from app.core.config import settings
from app.pipeline.pose_estimator import PoseEstimatorProtocol


def create_pose_estimator(
    *,
    min_confidence: float = 0.5,
    backend: str | None = None,
) -> PoseEstimatorProtocol:
    """Create the requested backend without importing optional runtimes eagerly."""
    selected = backend or settings.pose_backend
    if selected == "mediapipe":
        from app.pipeline.pose_estimator import PoseEstimator

        return PoseEstimator(
            min_confidence=min_confidence,
            # Production passes consecutive normalized video frames. Enabling
            # MediaPipe's temporal tracker materially reduces one-frame misses
            # when this backend is used as the RTMPose fallback.
            static_image_mode=False,
            model_complexity=1,
        )
    if selected == "rtmpose":
        from app.pipeline.rtmpose_estimator import RTMPoseEstimator

        return RTMPoseEstimator(
            min_confidence=min_confidence,
            mode=settings.rtmpose_mode,
            inference_backend=settings.rtmpose_inference_backend,
            device=settings.rtmpose_device,
        )
    raise ValueError(f"unsupported pose backend: {selected!r}")
