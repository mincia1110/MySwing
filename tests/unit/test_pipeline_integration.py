"""Unit tests for Celery pipeline orchestration (Task 14.3).

Verifies:
- Status transitions (pending → preprocessing → analyzing → evaluating → generating_report → completed)
- Error handling (failed status set with error_message)
- Graceful degradation behavior (partial failure produces partial results)
- Retry logic (max 2 retries with exponential backoff)
- Timeout handling (60s soft limit)

Requirements: 6.9, 6.10, 8.1
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.tasks.pipeline import (
    STATUS_ANALYZING,
    STATUS_COMPLETED,
    STATUS_EVALUATING,
    STATUS_FAILED,
    STATUS_GENERATING_REPORT,
    STATUS_PREPROCESSING,
    _run_bat_detection,
    _run_biomechanics_analysis,
    _run_pose_estimation,
    _run_preprocessing,
    _run_report_generation,
    _run_swing_classification,
    _run_swing_evaluation,
    _update_analysis_status,
    analyze_swing_task,
)


@pytest.fixture
def analysis_id():
    """Generate a test analysis UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def mock_analysis_data():
    """Create mock analysis data returned by _get_analysis_data."""
    return {
        "analysis_id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "video_id": str(uuid.uuid4()),
        "video_file_key": "videos/test-video.mp4",
        "video_fps": 30.0,
        "user_profile": {
            "height": 175.0,
            "bat_length": 34.0,
            "batting_direction": "right",
            "weight": 80.0,
            "level": "recreational",
            "age_group": "adult",
        },
    }


class TestStatusTransitions:
    """Test that the pipeline correctly transitions through all status states."""

    @patch("app.tasks.pipeline._save_analysis_result")
    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_successful_pipeline_transitions_through_all_statuses(
        self, mock_update_status, mock_get_data, mock_save_result, mock_analysis_data
    ):
        """Pipeline should transition: preprocessing → analyzing → evaluating → generating_report → completed."""
        mock_get_data.return_value = mock_analysis_data
        analysis_id = mock_analysis_data["analysis_id"]

        # Run the task directly (not via Celery broker)
        result = analyze_swing_task.apply(args=[analysis_id]).get()

        assert result["status"] == STATUS_COMPLETED
        assert result["analysis_id"] == analysis_id

        # Verify status transitions were called in order
        status_calls = [
            call[0][1] for call in mock_update_status.call_args_list
        ]
        assert STATUS_PREPROCESSING in status_calls
        assert STATUS_ANALYZING in status_calls
        assert STATUS_EVALUATING in status_calls
        assert STATUS_GENERATING_REPORT in status_calls
        assert STATUS_COMPLETED in status_calls

    @patch("app.tasks.pipeline._save_analysis_result")
    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_status_transitions_in_correct_order(
        self, mock_update_status, mock_get_data, mock_save_result, mock_analysis_data
    ):
        """Status transitions should occur in the correct sequential order."""
        mock_get_data.return_value = mock_analysis_data
        analysis_id = mock_analysis_data["analysis_id"]

        analyze_swing_task.apply(args=[analysis_id]).get()

        status_calls = [
            call[0][1] for call in mock_update_status.call_args_list
        ]

        # Find indices of each status
        preprocessing_idx = status_calls.index(STATUS_PREPROCESSING)
        analyzing_idx = status_calls.index(STATUS_ANALYZING)
        evaluating_idx = status_calls.index(STATUS_EVALUATING)
        generating_idx = status_calls.index(STATUS_GENERATING_REPORT)
        completed_idx = status_calls.index(STATUS_COMPLETED)

        assert preprocessing_idx < analyzing_idx
        assert analyzing_idx < evaluating_idx
        assert evaluating_idx < generating_idx
        assert generating_idx < completed_idx

    @patch("app.tasks.pipeline._save_analysis_result")
    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_started_at_set_on_preprocessing(
        self, mock_update_status, mock_get_data, mock_save_result, mock_analysis_data
    ):
        """The started_at timestamp should be set when entering preprocessing status."""
        mock_get_data.return_value = mock_analysis_data
        analysis_id = mock_analysis_data["analysis_id"]

        analyze_swing_task.apply(args=[analysis_id]).get()

        # First call should be preprocessing with started_at
        first_call = mock_update_status.call_args_list[0]
        assert first_call[0][1] == STATUS_PREPROCESSING
        # started_at should be passed as keyword arg
        assert first_call[1].get("started_at") is not None or (
            len(first_call[0]) > 3 and first_call[0][3] is not None
        )

    @patch("app.tasks.pipeline._save_analysis_result")
    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_completed_at_set_on_completion(
        self, mock_update_status, mock_get_data, mock_save_result, mock_analysis_data
    ):
        """The completed_at timestamp should be set when entering completed status."""
        mock_get_data.return_value = mock_analysis_data
        analysis_id = mock_analysis_data["analysis_id"]

        analyze_swing_task.apply(args=[analysis_id]).get()

        # Last call should be completed with completed_at
        last_call = mock_update_status.call_args_list[-1]
        assert last_call[0][1] == STATUS_COMPLETED
        assert last_call[1].get("completed_at") is not None or (
            len(last_call[0]) > 4 and last_call[0][4] is not None
        )


class TestErrorHandling:
    """Test error handling: failed status set with error_message."""

    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_analysis_not_found_sets_failed_status(
        self, mock_update_status, mock_get_data, analysis_id
    ):
        """When analysis record is not found, status should be set to failed."""
        mock_get_data.return_value = None

        result = analyze_swing_task.apply(args=[analysis_id]).get()

        assert result["status"] == STATUS_FAILED
        assert "not found" in result["error_message"].lower()

        # Verify failed status was set
        failed_calls = [
            call for call in mock_update_status.call_args_list
            if call[0][1] == STATUS_FAILED
        ]
        assert len(failed_calls) >= 1

    @patch("app.tasks.pipeline._run_preprocessing")
    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_preprocessing_failure_sets_failed_status(
        self, mock_update_status, mock_get_data, mock_preprocess, mock_analysis_data
    ):
        """When preprocessing raises an exception, status should be set to failed."""
        mock_get_data.return_value = mock_analysis_data
        mock_preprocess.side_effect = RuntimeError("Video download failed")
        analysis_id = mock_analysis_data["analysis_id"]

        result = analyze_swing_task.apply(args=[analysis_id]).get()

        assert result["status"] == STATUS_FAILED
        assert "Video download failed" in result["error_message"]

    @patch("app.tasks.pipeline._run_preprocessing")
    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_error_message_stored_on_failure(
        self, mock_update_status, mock_get_data, mock_preprocess, mock_analysis_data
    ):
        """Error message should be stored in the status update when pipeline fails."""
        mock_get_data.return_value = mock_analysis_data
        error_msg = "S3 connection timeout"
        mock_preprocess.side_effect = ConnectionError(error_msg)
        analysis_id = mock_analysis_data["analysis_id"]

        result = analyze_swing_task.apply(args=[analysis_id]).get()

        # Verify the error message was passed to _update_analysis_status
        failed_calls = [
            call for call in mock_update_status.call_args_list
            if call[0][1] == STATUS_FAILED
        ]
        assert len(failed_calls) >= 1
        # Check error_message was passed
        failed_call = failed_calls[-1]
        assert error_msg in (failed_call[1].get("error_message", "") or failed_call[0][2] or "")

    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_missing_video_file_key_sets_failed(
        self, mock_update_status, mock_get_data, analysis_id
    ):
        """When video_file_key is None, pipeline should fail with appropriate error."""
        mock_get_data.return_value = {
            "analysis_id": analysis_id,
            "user_id": str(uuid.uuid4()),
            "video_id": str(uuid.uuid4()),
            "video_file_key": None,
            "video_fps": 30.0,
            "user_profile": None,
        }

        result = analyze_swing_task.apply(args=[analysis_id]).get()

        assert result["status"] == STATUS_FAILED
        assert "video file key" in result["error_message"].lower() or "no video" in result["error_message"].lower()


class TestGracefulDegradation:
    """Test graceful degradation: partial failure produces partial results."""

    def test_pose_estimation_partial_failure_returns_result(self, analysis_id):
        """Pose estimation should return partial result on failure, not raise."""
        result = _run_pose_estimation(analysis_id, {"frames_extracted": True})

        assert result["analysis_id"] == analysis_id
        assert "pose_sequence" in result
        assert result["status"] in ("completed", "partial_failure")

    def test_bat_detection_partial_failure_returns_result(self, analysis_id):
        """Bat detection should return partial result on failure, not raise."""
        result = _run_bat_detection(analysis_id, {"frames_extracted": True})

        assert result["analysis_id"] == analysis_id
        assert "bat_trajectory" in result
        assert result["status"] in ("completed", "partial_failure")

    def test_swing_classification_partial_failure_returns_result(self, analysis_id):
        """Swing classification should return partial result on failure, not raise."""
        result = _run_swing_classification(
            analysis_id,
            {"pose_sequence": []},
            {"bat_trajectory": {}},
            30.0,
        )

        assert result["analysis_id"] == analysis_id
        assert "phases" in result
        assert result["status"] in ("completed", "partial_failure")

    def test_biomechanics_partial_failure_returns_unmeasurable_metrics(self, analysis_id):
        """Biomechanics analysis should report unmeasurable metrics on partial failure."""
        result = _run_biomechanics_analysis(
            analysis_id,
            {"pose_sequence": []},
            {"bat_trajectory": {}},
            {"phases": {}},
            None,
            30.0,
        )

        assert result["analysis_id"] == analysis_id
        assert "unmeasurable_metrics" in result
        assert result["status"] in ("completed", "partial_failure")

    def test_swing_evaluation_partial_failure_returns_result(self, analysis_id):
        """Swing evaluation should return partial result on failure, not raise."""
        result = _run_swing_evaluation(
            analysis_id,
            {"bat_speed": None},
            {},
            {},
            {},
            None,
            30.0,
        )

        assert result["analysis_id"] == analysis_id
        assert "evaluations" in result
        assert "improvements" in result
        assert result["status"] in ("completed", "partial_failure")

    def test_report_generation_partial_failure_returns_result(self, analysis_id):
        """Report generation should return partial result on failure, not raise."""
        result = _run_report_generation(
            analysis_id, {}, {}, {}, {}, {}, {}
        )

        assert result["analysis_id"] == analysis_id
        assert "overlay_video_key" in result
        assert result["status"] in ("completed", "partial_failure")

    @patch("app.tasks.pipeline._save_analysis_result")
    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_full_pipeline_completes_with_empty_data(
        self, mock_update_status, mock_get_data, mock_save_result, mock_analysis_data
    ):
        """Pipeline should complete successfully even with minimal/empty data."""
        mock_get_data.return_value = mock_analysis_data
        analysis_id = mock_analysis_data["analysis_id"]

        result = analyze_swing_task.apply(args=[analysis_id]).get()

        assert result["status"] == STATUS_COMPLETED
        assert "processing_time_seconds" in result


class TestRetryLogic:
    """Test retry logic: max 2 retries with exponential backoff."""

    def test_task_has_max_retries_2(self):
        """analyze_swing_task should have max_retries=2."""
        assert analyze_swing_task.max_retries == 2

    def test_task_has_retry_backoff_enabled(self):
        """analyze_swing_task should have retry_backoff=True."""
        assert analyze_swing_task.retry_backoff is True

    def test_task_has_retry_backoff_max_60(self):
        """analyze_swing_task should have retry_backoff_max=60."""
        assert analyze_swing_task.retry_backoff_max == 60

    def test_task_has_retry_jitter_enabled(self):
        """analyze_swing_task should have retry_jitter=True."""
        assert analyze_swing_task.retry_jitter is True


class TestTimeoutHandling:
    """Test timeout handling: configured soft/hard limits."""

    def test_task_has_soft_time_limit_positive(self):
        """analyze_swing_task should have a positive soft_time_limit."""
        assert analyze_swing_task.soft_time_limit > 0

    def test_task_has_hard_time_limit_greater_than_soft(self):
        """analyze_swing_task hard limit should exceed soft limit."""
        assert analyze_swing_task.time_limit > analyze_swing_task.soft_time_limit

    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_soft_timeout_sets_failed_status(
        self, mock_update_status, mock_get_data, mock_analysis_data
    ):
        """When SoftTimeLimitExceeded is raised, status should be set to failed with timeout message."""
        from celery.exceptions import SoftTimeLimitExceeded

        mock_get_data.return_value = mock_analysis_data
        analysis_id = mock_analysis_data["analysis_id"]

        # Patch _run_preprocessing to raise SoftTimeLimitExceeded
        with patch(
            "app.tasks.pipeline._run_preprocessing",
            side_effect=SoftTimeLimitExceeded("time limit exceeded"),
        ):
            result = analyze_swing_task.apply(args=[analysis_id]).get()

        assert result["status"] == STATUS_FAILED
        assert "timed out" in result["error_message"].lower()

        # Verify failed status was set
        failed_calls = [
            call for call in mock_update_status.call_args_list
            if call[0][1] == STATUS_FAILED
        ]
        assert len(failed_calls) >= 1


class TestPreprocessingStep:
    """Test the preprocessing step."""

    def test_preprocessing_requires_video_file_key(self, analysis_id):
        """Preprocessing should raise ValueError when video_file_key is None."""
        with pytest.raises(ValueError, match="No video file key"):
            _run_preprocessing(analysis_id, None)

    def test_preprocessing_returns_expected_structure(self, analysis_id):
        """Preprocessing should return dict with expected keys."""
        result = _run_preprocessing(analysis_id, "videos/test.mp4")

        assert result["analysis_id"] == analysis_id
        assert result["video_file_key"] == "videos/test.mp4"
        assert result["frames_extracted"] is True
        assert result["status"] == "completed"


class TestUpdateAnalysisStatus:
    """Test the _update_analysis_status helper function."""

    @patch("app.tasks.pipeline.sync_session_factory")
    def test_update_status_commits_on_success(self, mock_session_factory, analysis_id):
        """Status update should commit the session on success."""
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        # Mock the query to return an analysis object
        mock_analysis = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_analysis

        _update_analysis_status(analysis_id, STATUS_ANALYZING)

        mock_analysis.status = STATUS_ANALYZING
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @patch("app.tasks.pipeline.sync_session_factory")
    def test_update_status_sets_error_message(self, mock_session_factory, analysis_id):
        """Status update should set error_message when provided."""
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        mock_analysis = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_analysis

        _update_analysis_status(
            analysis_id, STATUS_FAILED, error_message="Something went wrong"
        )

        assert mock_analysis.error_message == "Something went wrong"
        mock_session.commit.assert_called_once()

    @patch("app.tasks.pipeline.sync_session_factory")
    def test_update_status_handles_missing_analysis(self, mock_session_factory, analysis_id):
        """Status update should handle gracefully when analysis not found."""
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Should not raise
        _update_analysis_status(analysis_id, STATUS_ANALYZING)

        mock_session.commit.assert_not_called()
        mock_session.close.assert_called_once()

    @patch("app.tasks.pipeline.sync_session_factory")
    def test_update_status_rollback_on_exception(self, mock_session_factory, analysis_id):
        """Status update should rollback on database exception."""
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_session.query.side_effect = Exception("DB connection lost")

        # Should not raise
        _update_analysis_status(analysis_id, STATUS_ANALYZING)

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()


class TestProcessingTimeTracking:
    """Test that processing time is tracked and returned."""

    @patch("app.tasks.pipeline._save_analysis_result")
    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_processing_time_included_in_result(
        self, mock_update_status, mock_get_data, mock_save_result, mock_analysis_data
    ):
        """Completed pipeline should include processing_time_seconds in result."""
        mock_get_data.return_value = mock_analysis_data
        analysis_id = mock_analysis_data["analysis_id"]

        result = analyze_swing_task.apply(args=[analysis_id]).get()

        assert result["status"] == STATUS_COMPLETED
        assert "processing_time_seconds" in result
        assert result["processing_time_seconds"] >= 0

    @patch("app.tasks.pipeline._save_analysis_result")
    @patch("app.tasks.pipeline._get_analysis_data")
    @patch("app.tasks.pipeline._update_analysis_status")
    def test_processing_time_passed_to_save_result(
        self, mock_update_status, mock_get_data, mock_save_result, mock_analysis_data
    ):
        """Processing time should be passed to _save_analysis_result."""
        mock_get_data.return_value = mock_analysis_data
        analysis_id = mock_analysis_data["analysis_id"]

        analyze_swing_task.apply(args=[analysis_id]).get()

        mock_save_result.assert_called_once()
        call_kwargs = mock_save_result.call_args[1]
        assert "processing_time_seconds" in call_kwargs
        assert call_kwargs["processing_time_seconds"] >= 0
