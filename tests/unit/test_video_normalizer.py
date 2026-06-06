import numpy as np

from app.services.video_normalizer import (
    detect_active_content_box,
    normalize_frames_for_analysis,
)


def _solid_frame(width: int, height: int, value: int = 128) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_downsamples_60fps_frames_to_30fps_timeline():
    frames = [_solid_frame(64, 48, value=i) for i in range(60)]

    result = normalize_frames_for_analysis(frames, source_fps=60.0, target_fps=30.0)

    assert result.fps == 30.0
    assert len(result.frames) == 30
    assert result.sampled_frame_indices[:5] == [0, 2, 4, 6, 8]
    assert result.original_width == 64
    assert result.original_height == 48


def test_keeps_30fps_frame_count():
    frames = [_solid_frame(64, 48) for _ in range(30)]

    result = normalize_frames_for_analysis(frames, source_fps=30.0, target_fps=30.0)

    assert result.fps == 30.0
    assert len(result.frames) == 30
    assert result.sampled_frame_indices == list(range(30))


def test_detects_and_crops_pillarbox_content():
    frame = np.zeros((108, 192, 3), dtype=np.uint8)
    frame[:, 64:128] = 180

    crop_box = detect_active_content_box([frame])
    result = normalize_frames_for_analysis([frame], source_fps=30.0)

    assert crop_box == (64, 0, 64, 108)
    assert result.crop_box == (64, 0, 64, 108)
    assert result.video_width == 64
    assert result.video_height == 108
    assert result.frames[0].shape[:2] == (108, 64)


def test_ignores_negligible_crop():
    frame = _solid_frame(100, 80)

    result = normalize_frames_for_analysis([frame], source_fps=30.0)

    assert result.crop_box is None
    assert result.video_width == 100
    assert result.video_height == 80
