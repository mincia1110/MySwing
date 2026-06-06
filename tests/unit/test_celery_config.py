"""Unit tests for Celery + Redis task queue configuration.

Verifies:
- Celery app configuration (broker, backend, serialization, timezone)
- Task routing to correct queues
- Retry policy settings
- Worker settings (concurrency, prefetch, time limits)
"""

import pytest

from app.core.celery_app import (
    DEFAULT_RETRY_POLICY,
    celery_app,
    task_queues,
    task_routes,
)


class TestCeleryAppConfiguration:
    """Test Celery app instance configuration."""

    def test_broker_url_uses_redis(self):
        """Celery broker should be configured to use Redis."""
        assert "redis://" in celery_app.conf.broker_url

    def test_result_backend_uses_redis(self):
        """Celery result backend should be configured to use Redis."""
        assert "redis://" in celery_app.conf.result_backend

    def test_broker_and_backend_use_different_redis_dbs(self):
        """Broker and backend should use different Redis databases to avoid conflicts."""
        broker_url = celery_app.conf.broker_url
        backend_url = celery_app.conf.result_backend
        assert broker_url != backend_url

    def test_task_serializer_is_json(self):
        """Task serializer should be JSON for interoperability."""
        assert celery_app.conf.task_serializer == "json"

    def test_result_serializer_is_json(self):
        """Result serializer should be JSON."""
        assert celery_app.conf.result_serializer == "json"

    def test_accept_content_is_json(self):
        """Accepted content types should include JSON."""
        assert "json" in celery_app.conf.accept_content

    def test_timezone_is_utc(self):
        """Timezone should be UTC."""
        assert celery_app.conf.timezone == "UTC"

    def test_enable_utc(self):
        """UTC should be enabled."""
        assert celery_app.conf.enable_utc is True

    def test_task_acks_late_enabled(self):
        """Task acknowledgement should be late for reliability."""
        assert celery_app.conf.task_acks_late is True

    def test_task_track_started_enabled(self):
        """Task started tracking should be enabled."""
        assert celery_app.conf.task_track_started is True


class TestTaskRouting:
    """Test task routing configuration."""

    def test_three_queues_defined(self):
        """Three queues should be defined: default, video_processing, analysis."""
        queue_names = {q.name for q in task_queues}
        assert queue_names == {"default", "video_processing", "analysis"}

    def test_preprocess_video_routes_to_video_processing_queue(self):
        """Video preprocessing task should route to video_processing queue."""
        route = task_routes["app.tasks.pipeline.preprocess_video_task"]
        assert route["queue"] == "video_processing"

    def test_analyze_swing_routes_to_analysis_queue(self):
        """Swing analysis orchestrator should route to analysis queue."""
        route = task_routes["app.tasks.pipeline.analyze_swing_task"]
        assert route["queue"] == "analysis"

    def test_estimate_pose_routes_to_analysis_queue(self):
        """Pose estimation task should route to analysis queue."""
        route = task_routes["app.tasks.pipeline.estimate_pose_task"]
        assert route["queue"] == "analysis"

    def test_detect_bat_routes_to_analysis_queue(self):
        """Bat detection task should route to analysis queue."""
        route = task_routes["app.tasks.pipeline.detect_bat_task"]
        assert route["queue"] == "analysis"

    def test_classify_swing_routes_to_analysis_queue(self):
        """Swing classification task should route to analysis queue."""
        route = task_routes["app.tasks.pipeline.classify_swing_task"]
        assert route["queue"] == "analysis"

    def test_analyze_biomechanics_routes_to_analysis_queue(self):
        """Biomechanics analysis task should route to analysis queue."""
        route = task_routes["app.tasks.pipeline.analyze_biomechanics_task"]
        assert route["queue"] == "analysis"

    def test_evaluate_swing_routes_to_analysis_queue(self):
        """Swing evaluation task should route to analysis queue."""
        route = task_routes["app.tasks.pipeline.evaluate_swing_task"]
        assert route["queue"] == "analysis"

    def test_generate_report_routes_to_analysis_queue(self):
        """Report generation task should route to analysis queue."""
        route = task_routes["app.tasks.pipeline.generate_report_task"]
        assert route["queue"] == "analysis"

    def test_default_queue_is_default(self):
        """Default queue should be 'default'."""
        assert celery_app.conf.task_default_queue == "default"


class TestRetryPolicy:
    """Test retry policy configuration."""

    def test_max_retries_is_2(self):
        """Max retries should be 2."""
        assert DEFAULT_RETRY_POLICY["max_retries"] == 2

    def test_retry_backoff_enabled(self):
        """Exponential backoff should be enabled."""
        assert DEFAULT_RETRY_POLICY["retry_backoff"] is True

    def test_retry_backoff_max_is_60(self):
        """Maximum backoff delay should be 60 seconds."""
        assert DEFAULT_RETRY_POLICY["retry_backoff_max"] == 60

    def test_retry_jitter_enabled(self):
        """Retry jitter should be enabled to avoid thundering herd."""
        assert DEFAULT_RETRY_POLICY["retry_jitter"] is True

    def test_celery_app_max_retries(self):
        """Celery app-level max retries should be 2."""
        assert celery_app.conf.task_max_retries == 2


class TestWorkerSettings:
    """Test worker configuration."""

    def test_concurrency_is_2(self):
        """Worker concurrency should be 2 for CPU-bound CV tasks."""
        assert celery_app.conf.worker_concurrency == 2

    def test_prefetch_multiplier_is_1(self):
        """Prefetch multiplier should be 1 for heavy tasks (one at a time per worker)."""
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_task_time_limit_is_120(self):
        """Hard time limit should be 120 seconds."""
        assert celery_app.conf.task_time_limit == 120

    def test_task_soft_time_limit_is_60(self):
        """Soft time limit should be 60 seconds (matches Req 6.10 with buffer)."""
        assert celery_app.conf.task_soft_time_limit == 60


class TestPipelineTasksRegistered:
    """Test that pipeline tasks are properly registered."""

    def test_analyze_swing_task_exists(self):
        """analyze_swing_task should be importable."""
        from app.tasks.pipeline import analyze_swing_task

        assert analyze_swing_task.name == "app.tasks.pipeline.analyze_swing_task"

    def test_preprocess_video_task_exists(self):
        """preprocess_video_task should be importable."""
        from app.tasks.pipeline import preprocess_video_task

        assert preprocess_video_task.name == "app.tasks.pipeline.preprocess_video_task"

    def test_estimate_pose_task_exists(self):
        """estimate_pose_task should be importable."""
        from app.tasks.pipeline import estimate_pose_task

        assert estimate_pose_task.name == "app.tasks.pipeline.estimate_pose_task"

    def test_detect_bat_task_exists(self):
        """detect_bat_task should be importable."""
        from app.tasks.pipeline import detect_bat_task

        assert detect_bat_task.name == "app.tasks.pipeline.detect_bat_task"

    def test_classify_swing_task_exists(self):
        """classify_swing_task should be importable."""
        from app.tasks.pipeline import classify_swing_task

        assert classify_swing_task.name == "app.tasks.pipeline.classify_swing_task"

    def test_analyze_biomechanics_task_exists(self):
        """analyze_biomechanics_task should be importable."""
        from app.tasks.pipeline import analyze_biomechanics_task

        assert analyze_biomechanics_task.name == "app.tasks.pipeline.analyze_biomechanics_task"

    def test_evaluate_swing_task_exists(self):
        """evaluate_swing_task should be importable."""
        from app.tasks.pipeline import evaluate_swing_task

        assert evaluate_swing_task.name == "app.tasks.pipeline.evaluate_swing_task"

    def test_generate_report_task_exists(self):
        """generate_report_task should be importable."""
        from app.tasks.pipeline import generate_report_task

        assert generate_report_task.name == "app.tasks.pipeline.generate_report_task"

    def test_all_pipeline_tasks_have_retry_policy(self):
        """All pipeline tasks should have retry policy configured."""
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

        tasks = [
            analyze_swing_task,
            preprocess_video_task,
            estimate_pose_task,
            detect_bat_task,
            classify_swing_task,
            analyze_biomechanics_task,
            evaluate_swing_task,
            generate_report_task,
        ]

        for task in tasks:
            assert task.max_retries == 2, f"{task.name} should have max_retries=2"
            assert task.retry_backoff is True, f"{task.name} should have retry_backoff=True"
            assert task.retry_backoff_max == 60, f"{task.name} should have retry_backoff_max=60"
            assert task.retry_jitter is True, f"{task.name} should have retry_jitter=True"
