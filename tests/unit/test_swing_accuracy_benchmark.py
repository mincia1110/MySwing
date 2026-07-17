"""Unit tests for the service-free swing accuracy benchmark."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.swing_accuracy_benchmark import (
    CANDIDATE_CONFIGS,
    PipelineComponents,
    apply_conservative_phase_fallback,
    compute_pose_coverage,
    compute_segment_length_cv,
    find_frame_files,
    impact_absolute_error,
    load_frames_input,
    prepare_analysis_frames,
    run_candidate,
    select_candidate_configs,
    select_conditioning_modes,
    sha256_frame_files,
    validate_output_path,
)


def _keypoint(name: str, x: float = 0.0, y: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(name=name, x=x, y=y, z=0.0, confidence=0.9)


def _pose(
    frame_index: int,
    keypoints: list[SimpleNamespace],
    *,
    low_confidence: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        frame_index=frame_index,
        keypoints=keypoints,
        is_low_confidence=low_confidence,
        overall_confidence=0.9,
    )


def _incomplete_phase_metrics() -> dict:
    phase_names = (
        "stance",
        "load",
        "stride",
        "rotation",
        "impact",
        "follow_through",
    )
    return {
        "phase_ranges": {phase: None for phase in phase_names},
        "phase_durations_ms": {phase: None for phase in phase_names},
        "transition_frames": {},
        "transitions": [],
        "estimated_impact_frame": None,
        "estimated_impact_source": None,
        "anomalies": [],
        "classification_failures": [],
    }


def _bat_trajectory(observed_count: int) -> SimpleNamespace:
    positions = (0.0, 0.01, 0.02, 0.03, 0.04, 0.10, 0.50, 0.60, 0.65, 0.67)
    detections = [
        SimpleNamespace(
            frame_index=frame_index,
            detected=True,
            position=(position, 0.0),
            bat_head_position=None,
            is_predicted=frame_index >= observed_count,
        )
        for frame_index, position in enumerate(positions)
    ]
    return SimpleNamespace(detections=detections)


def test_frame_file_order_and_hash_are_deterministic_and_content_sensitive(
    tmp_path,
) -> None:
    later = tmp_path / "frame_000010.npy"
    earlier = tmp_path / "frame_000002.npy"
    ignored = tmp_path / "other.npy"
    np.save(later, np.full((2, 3, 3), 10, dtype=np.uint8))
    np.save(earlier, np.full((2, 3, 3), 2, dtype=np.uint8))
    np.save(ignored, np.zeros((2, 3, 3), dtype=np.uint8))

    frame_files = find_frame_files(tmp_path)
    assert [path.name for path in frame_files] == [
        "frame_000002.npy",
        "frame_000010.npy",
    ]
    original_hash = sha256_frame_files(frame_files)
    assert sha256_frame_files(list(reversed(frame_files))) == original_hash

    np.save(later, np.full((2, 3, 3), 11, dtype=np.uint8))
    assert sha256_frame_files(frame_files) != original_hash


def test_load_frames_input_infers_dimensions_and_uses_read_only_arrays(
    tmp_path,
) -> None:
    np.save(
        tmp_path / "frame_000000.npy",
        np.zeros((4, 6, 3), dtype=np.uint8),
    )

    loaded = load_frames_input(tmp_path)

    assert (loaded.width, loaded.height, loaded.fps) == (6, 4, 30.0)
    assert loaded.frames[0].flags.writeable is False
    with pytest.raises(ValueError, match="does not match frame width"):
        load_frames_input(tmp_path, width=7)


def test_left_handed_video_frames_are_mirrored_but_canonical_arrays_are_not(
    tmp_path,
) -> None:
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    common = {
        "path": tmp_path,
        "frames": [frame],
        "width": 3,
        "height": 2,
        "fps": 30.0,
        "sha256": "synthetic",
        "details": {},
    }
    from scripts.swing_accuracy_benchmark import BenchmarkInput

    video_input = BenchmarkInput(mode="video", **common)
    video_frames, video_metadata = prepare_analysis_frames(
        video_input,
        "left",
    )
    assert np.array_equal(video_frames[0], frame[:, ::-1])
    assert video_frames[0].flags.c_contiguous
    assert video_metadata["horizontal_flip_applied"] is True

    frames_input = BenchmarkInput(mode="frames_dir", **common)
    canonical_frames, frames_metadata = prepare_analysis_frames(
        frames_input,
        "left",
    )
    assert canonical_frames[0] is frame
    assert frames_metadata["input_contract"] == (
        "frames_dir_must_already_be_canonical_rhb"
    )


def test_pose_coverage_counts_missing_and_empty_frames_as_low_confidence() -> None:
    poses = [
        _pose(
            0,
            [
                _keypoint("left_shoulder"),
                _keypoint("right_shoulder"),
                _keypoint("left_hip"),
            ],
        ),
        _pose(
            1,
            [_keypoint("left_shoulder")],
            low_confidence=True,
        ),
        _pose(2, []),
    ]

    metrics = compute_pose_coverage(poses, frame_count=4)

    assert metrics["pose_frame_coverage"] == pytest.approx(0.5)
    assert metrics["low_confidence_frame_fraction"] == pytest.approx(0.75)
    assert metrics["mean_keypoint_count"] == pytest.approx(1.0)
    assert metrics["critical_joint_coverage"]["shoulders"] == pytest.approx(
        3 / 8
    )
    assert metrics["critical_joint_coverage"]["hips"] == pytest.approx(1 / 8)


def test_segment_length_cv_uses_pixel_aspect_correction_and_nulls_missing() -> None:
    poses = [
        _pose(
            0,
            [
                _keypoint("left_shoulder", 0.1, 0.1),
                _keypoint("left_elbow", 0.2, 0.1),
            ],
        ),
        _pose(
            1,
            [
                _keypoint("left_shoulder", 0.3, 0.2),
                _keypoint("left_elbow", 0.3, 0.4),
            ],
        ),
        _pose(
            2,
            [
                _keypoint("left_shoulder", 0.6, 0.5),
                _keypoint("left_elbow", 0.7, 0.5),
            ],
        ),
    ]

    metrics = compute_segment_length_cv(poses, width=100, height=50)

    assert metrics["segment_coefficient_of_variation"][
        "left_upper_arm"
    ] == pytest.approx(0.0, abs=1e-12)
    assert metrics["median_segment_length_coefficient_of_variation"] == (
        pytest.approx(0.0, abs=1e-12)
    )
    assert metrics["segment_coefficient_of_variation"][
        "right_upper_arm"
    ] is None


def test_impact_absolute_error_requires_both_frames() -> None:
    assert impact_absolute_error(17, 12) == 5
    assert impact_absolute_error(None, 12) is None
    assert impact_absolute_error(17, None) is None


def test_fully_predicted_wrist_proxy_never_uses_speed_phase_fallback() -> None:
    raw_metrics = _incomplete_phase_metrics()

    final_metrics = apply_conservative_phase_fallback(
        raw_metrics,
        _bat_trajectory(observed_count=0),
        pose_sequence=[],
        fps=30.0,
    )

    assert final_metrics["fallback_applied"] is False
    assert final_metrics["result_source"] == "swing_phase_classifier"
    assert final_metrics["fallback_reason"] == (
        "insufficient_observed_non_predicted_bat_lines"
    )
    assert final_metrics["estimated_impact_frame"] is None
    assert final_metrics["fallback_evidence"][
        "observed_non_predicted_bat_lines"
    ] == 0


def test_three_observed_bat_lines_enable_conservative_speed_phase_fallback() -> None:
    raw_metrics = _incomplete_phase_metrics()

    final_metrics = apply_conservative_phase_fallback(
        raw_metrics,
        _bat_trajectory(observed_count=3),
        pose_sequence=[],
        fps=30.0,
    )

    assert final_metrics["fallback_applied"] is True
    assert final_metrics["result_source"] == "observed_bat_contact_only"
    # Only the three observed points participate; the later, faster predicted
    # wrist segment must not move the estimate to frame 6.
    assert final_metrics["estimated_impact_frame"] == 1
    assert final_metrics["estimated_impact_source"] == (
        "speed_fallback.smoothed_peak_speed"
    )
    assert final_metrics["phase_ranges"]["impact"] == [1, 1]
    assert all(
        final_metrics["phase_ranges"][phase] is None
        for phase in ("stance", "load", "stride", "rotation", "follow_through")
    )
    assert final_metrics["transitions"] == []
    assert final_metrics["fallback_evidence"][
        "observed_non_predicted_bat_lines"
    ] == 3
    assert raw_metrics["estimated_impact_frame"] is None


def test_protected_output_paths_are_rejected(tmp_path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    with pytest.raises(ValueError, match="outputs"):
        validate_output_path(
            repository_root / "outputs" / "benchmark.json",
            repository_root=repository_root,
        )
    with pytest.raises(ValueError, match="sample_videos"):
        validate_output_path(
            repository_root / "sample_videos" / "benchmark.json",
            repository_root=repository_root,
        )

    allowed = repository_root / "benchmarks" / "result.json"
    assert validate_output_path(
        allowed, repository_root=repository_root
    ) == allowed


def test_candidate_and_conditioning_selection_defaults_and_deduplicates() -> None:
    assert [config.name for config in select_candidate_configs(None)] == [
        "static_full",
        "static_heavy",
        "video_full",
        "video_heavy",
        "rtmpose_lightweight",
        "rtmpose_balanced",
        "rtmpose_performance",
    ]
    assert [
        config.name
        for config in select_candidate_configs(
            ["video_heavy", "static_full", "video_heavy"]
        )
    ] == ["video_heavy", "static_full"]
    assert select_conditioning_modes(None) == ["on", "off"]
    assert select_conditioning_modes(["off", "off"]) == ["off"]
    with pytest.raises(ValueError, match="unknown candidate"):
        select_candidate_configs(["unknown"])


def test_rtmpose_candidate_uses_onnx_cpu_estimator() -> None:
    captured: dict = {}

    class EmptyRTMPoseEstimator:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.is_available = True

        def process_frame(self, _frame, frame_index):
            return _pose(frame_index, [])

        def close(self):
            return None

    components = PipelineComponents(
        pose_estimator=object,
        pose_tracker=object,
        wrist_bat_estimator=object,
        pose_constrained_bat_tracker=object,
        swing_phase_classifier=object,
        rtmpose_estimator=EmptyRTMPoseEstimator,
    )

    result = run_candidate(
        CANDIDATE_CONFIGS["rtmpose_balanced"],
        frames=[np.zeros((2, 3, 3), dtype=np.uint8)],
        width=3,
        height=2,
        fps=30.0,
        batting_direction="right",
        conditioning_modes=["off"],
        components=components,
    )

    assert captured == {
        "min_confidence": 0.5,
        "mode": "balanced",
        "inference_backend": "onnxruntime",
        "device": "cpu",
    }
    # Estimation succeeds; downstream fails because the injected pipeline is
    # deliberately incomplete. The backend selection itself remains proven.
    assert result["status"] == "failed"
    assert result["runs"][0]["failure"]["stage"] == "pose_tracking"


def test_candidate_estimator_failure_is_structured_for_every_pair() -> None:
    class FailingPoseEstimator:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("synthetic estimator failure")

    components = PipelineComponents(
        pose_estimator=FailingPoseEstimator,
        pose_tracker=object,
        wrist_bat_estimator=object,
        pose_constrained_bat_tracker=object,
        swing_phase_classifier=object,
    )

    result = run_candidate(
        CANDIDATE_CONFIGS["video_full"],
        frames=[np.zeros((2, 3, 3), dtype=np.uint8)],
        width=3,
        height=2,
        fps=30.0,
        batting_direction="right",
        conditioning_modes=["on", "off"],
        components=components,
    )

    assert result["status"] == "failed"
    assert result["failure"] == {
        "stage": "pose_estimation",
        "type": "RuntimeError",
        "message": "synthetic estimator failure",
    }
    assert [run["status"] for run in result["runs"]] == ["failed", "failed"]
    assert all(
        run["failure"]["stage"] == "pose_estimation"
        for run in result["runs"]
    )
