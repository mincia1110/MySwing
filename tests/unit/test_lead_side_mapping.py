"""Unit tests for batting-direction lead-side mapping in biomechanics analyzers."""

from app.models.pose import Keypoint, PoseResult
from app.pipeline.biomechanics_analyzer import HandPathAnalyzer, SwingQualityAnalyzer


def _kp(name: str, x: float, y: float, conf: float = 0.99) -> Keypoint:
    return Keypoint(x=x, y=y, z=0.0, confidence=conf, name=name)


def _pose(frame_index: int, keypoints: list[Keypoint]) -> PoseResult:
    return PoseResult(
        frame_index=frame_index,
        keypoints=keypoints,
        person_id=0,
        is_primary_batter=True,
        overall_confidence=0.99,
        is_low_confidence=False,
    )


def test_hand_path_efficiency_uses_lead_wrist_by_batting_direction():
    analyzer = HandPathAnalyzer()

    # Left wrist path: straight line (efficiency ~1.0)
    # Right wrist path: zig-zag (efficiency < 1.0)
    poses = [
        _pose(0, [_kp("left_wrist", 0.0, 0.0), _kp("right_wrist", 0.0, 0.0)]),
        _pose(1, [_kp("left_wrist", 1.0, 0.0), _kp("right_wrist", 1.0, 1.0)]),
        _pose(2, [_kp("left_wrist", 2.0, 0.0), _kp("right_wrist", 2.0, 0.0)]),
    ]

    rhb_eff = analyzer.calculate_hand_path_efficiency(poses, 0, 2, batting_direction="right")
    lhb_eff = analyzer.calculate_hand_path_efficiency(poses, 0, 2, batting_direction="left")

    assert rhb_eff == 1.0
    assert lhb_eff < rhb_eff


def test_front_knee_flexion_uses_lead_side_mapping():
    sq = SwingQualityAnalyzer()

    # left leg angle at knee: 90 deg
    # right leg angle at knee: 180 deg
    pose = _pose(
        10,
        [
            _kp("left_hip", 0.0, 1.0),
            _kp("left_knee", 0.0, 0.0),
            _kp("left_ankle", 1.0, 0.0),
            _kp("right_hip", 10.0, 1.0),
            _kp("right_knee", 10.0, 0.0),
            _kp("right_ankle", 10.0, -1.0),
        ],
    )

    phases = {"stride_end_frame": 10}

    rhb_angle = sq.front_knee_flexion_degrees([pose], phases, batting_direction="right")
    lhb_angle = sq.front_knee_flexion_degrees([pose], phases, batting_direction="left")

    assert rhb_angle is not None and abs(rhb_angle - 90.0) < 1e-6
    assert lhb_angle is not None and abs(lhb_angle - 180.0) < 1e-6
