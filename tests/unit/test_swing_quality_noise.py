"""Unit tests for swing-quality noise robustness."""

from app.models.pose import Keypoint, PoseResult
from app.pipeline.biomechanics_analyzer import SwingQualityAnalyzer


def _pose(frame_index: int, nose_x: float, nose_y: float) -> PoseResult:
    return PoseResult(
        frame_index=frame_index,
        keypoints=[Keypoint(x=nose_x, y=nose_y, z=0.0, confidence=0.99, name="nose")],
        person_id=0,
        is_primary_batter=True,
        overall_confidence=0.99,
        is_low_confidence=False,
    )


def test_head_stability_uses_p95_not_single_frame_max_spike():
    sq = SwingQualityAnalyzer()

    # Mostly stable around x~0.5, with one extreme outlier frame.
    poses = [
        _pose(0, 0.50, 0.50),
        _pose(1, 0.51, 0.50),
        _pose(2, 0.49, 0.50),
        _pose(3, 0.50, 0.51),
        _pose(4, 0.90, 0.90),  # outlier
    ]

    stability_cm = sq.head_stability_cm(poses, pixel_to_meter=1.0)

    assert stability_cm is not None
    # With max-distance this would be very large; p95 should damp one-frame spike.
    assert stability_cm < 40.0
    assert stability_cm > 0.0
