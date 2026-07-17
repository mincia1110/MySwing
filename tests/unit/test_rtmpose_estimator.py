"""Unit tests for the optional RTMPose ONNX backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.pipeline.rtmpose_estimator import COCO_TO_ESSENTIAL, RTMPoseEstimator


class _FakeBody:
    def __init__(self, coordinates: np.ndarray, scores: np.ndarray) -> None:
        self.coordinates = coordinates
        self.scores = scores
        self.frames: list[np.ndarray] = []

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.frames.append(frame)
        return self.coordinates, self.scores


def _coco_person(x_offset: float = 0.0) -> np.ndarray:
    points = np.zeros((17, 2), dtype=np.float64)
    for index in range(17):
        points[index] = (20.0 + index + x_offset, 10.0 + 2 * index)
    return points


def test_rtmpose_maps_coco_output_to_normalized_pipeline_contract() -> None:
    captured: dict[str, Any] = {}
    fake_body = _FakeBody(
        _coco_person()[np.newaxis, ...],
        np.full((1, 17), 0.9, dtype=np.float32),
    )

    def factory(**kwargs: Any) -> _FakeBody:
        captured.update(kwargs)
        return fake_body

    estimator = RTMPoseEstimator(
        mode="balanced",
        solution_factory=factory,
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = estimator.process_frame(frame, frame_index=7)

    assert captured == {
        "mode": "balanced",
        "backend": "onnxruntime",
        "device": "cpu",
        "to_openpose": False,
    }
    assert fake_body.frames == [frame]
    assert result.frame_index == 7
    assert result.is_primary_batter is True
    by_name = {keypoint.name: keypoint for keypoint in result.keypoints}
    assert set(COCO_TO_ESSENTIAL.values()).issubset(by_name)
    assert {"head", "spine", "neck"}.issubset(by_name)
    assert "left_index" not in by_name
    assert by_name["nose"].x == pytest.approx(0.1)
    assert by_name["nose"].y == pytest.approx(0.1)
    assert by_name["nose"].z == 0.0
    assert by_name["head"].x == by_name["nose"].x
    assert by_name["spine"].x == pytest.approx(
        (by_name["left_shoulder"].x + by_name["right_shoulder"].x) / 2
    )
    assert result.overall_confidence == pytest.approx(0.9)
    assert result.is_low_confidence is False


def test_rtmpose_filters_low_confidence_and_handles_empty_detection() -> None:
    scores = np.full((1, 17), 0.9, dtype=np.float32)
    scores[0, 9] = 0.2
    estimator = RTMPoseEstimator(
        min_confidence=0.5,
        solution_factory=lambda **_kwargs: _FakeBody(
            _coco_person()[np.newaxis, ...], scores
        ),
    )

    result = estimator.process_frame(np.zeros((100, 200, 3), dtype=np.uint8))
    assert "left_wrist" not in {keypoint.name for keypoint in result.keypoints}

    empty = RTMPoseEstimator(
        solution_factory=lambda **_kwargs: _FakeBody(
            np.empty((0, 17, 2)), np.empty((0, 17))
        ),
    )
    empty_result = empty.process_frame(
        np.zeros((100, 200, 3), dtype=np.uint8)
    )
    assert empty_result.keypoints == []
    assert empty_result.overall_confidence == 0.0
    assert empty_result.is_low_confidence is True


def test_rtmpose_selects_more_confident_person() -> None:
    people = np.stack([_coco_person(60.0), _coco_person(0.0)])
    scores = np.stack(
        [
            np.full(17, 0.95, dtype=np.float32),
            np.full(17, 0.55, dtype=np.float32),
        ]
    )
    estimator = RTMPoseEstimator(
        solution_factory=lambda **_kwargs: _FakeBody(people, scores)
    )

    result = estimator.process_frame(np.zeros((100, 200, 3), dtype=np.uint8))
    nose = next(keypoint for keypoint in result.keypoints if keypoint.name == "nose")
    assert nose.x == pytest.approx(0.4)
    assert nose.confidence == pytest.approx(0.95)


def test_rtmpose_limits_top_down_pose_inference_to_primary_bbox() -> None:
    class Detector:
        def __call__(self, _frame):
            return np.asarray(
                [
                    [90.0, 40.0, 120.0, 80.0],
                    [20.0, 5.0, 180.0, 98.0],
                ]
            )

    class PoseModel:
        def __init__(self) -> None:
            self.bboxes = []

        def __call__(self, _frame, *, bboxes):
            self.bboxes.append(bboxes)
            return (
                _coco_person()[np.newaxis, ...],
                np.full((1, 17), 0.9, dtype=np.float32),
            )

    class Solution:
        def __init__(self) -> None:
            self.det_model = Detector()
            self.pose_model = PoseModel()

    solution = Solution()
    estimator = RTMPoseEstimator(
        solution_factory=lambda **_kwargs: solution
    )
    estimator.process_frame(np.zeros((100, 200, 3), dtype=np.uint8))

    assert solution.pose_model.bboxes == [[[20.0, 5.0, 180.0, 98.0]]]


def test_rtmpose_validates_configuration_and_output_shape() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        RTMPoseEstimator(
            min_confidence=1.1,
            solution_factory=lambda **_kwargs: None,
        )
    with pytest.raises(ValueError, match="unknown RTMPose mode"):
        RTMPoseEstimator(
            mode="unknown",
            solution_factory=lambda **_kwargs: None,
        )

    estimator = RTMPoseEstimator(
        solution_factory=lambda **_kwargs: _FakeBody(
            np.zeros((1, 5, 2)), np.zeros((1, 5))
        )
    )
    with pytest.raises(ValueError, match="RTMPose output"):
        estimator.process_frame(np.zeros((100, 200, 3), dtype=np.uint8))


def test_rtmpose_close_marks_estimator_unavailable() -> None:
    estimator = RTMPoseEstimator(
        solution_factory=lambda **_kwargs: _FakeBody(
            _coco_person()[np.newaxis, ...],
            np.full((1, 17), 0.9, dtype=np.float32),
        )
    )
    assert estimator.is_available
    estimator.close()
    assert not estimator.is_available
    with pytest.raises(RuntimeError, match="RTMPose is not available"):
        estimator.process_frame(np.zeros((100, 200, 3), dtype=np.uint8))
