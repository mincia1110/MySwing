"""Celery application configuration.

Configures the Celery task queue with Redis as broker and backend,
task routing, retry policies, and worker settings for the swing analysis pipeline.
"""

from kombu import Exchange, Queue

from celery import Celery

from app.core.config import settings

# Define exchanges
default_exchange = Exchange("default", type="direct")
video_exchange = Exchange("video", type="direct")
analysis_exchange = Exchange("analysis", type="direct")

# Define task queues
task_queues = (
    Queue("default", default_exchange, routing_key="default"),
    Queue("video_processing", video_exchange, routing_key="video_processing"),
    Queue("analysis", analysis_exchange, routing_key="analysis"),
)

# Task routing configuration
task_routes = {
    "app.tasks.pipeline.preprocess_video_task": {"queue": "video_processing"},
    "app.tasks.pipeline.analyze_swing_task": {"queue": "analysis"},
    "app.tasks.pipeline.estimate_pose_task": {"queue": "analysis"},
    "app.tasks.pipeline.detect_bat_task": {"queue": "analysis"},
    "app.tasks.pipeline.classify_swing_task": {"queue": "analysis"},
    "app.tasks.pipeline.analyze_biomechanics_task": {"queue": "analysis"},
    "app.tasks.pipeline.evaluate_swing_task": {"queue": "analysis"},
    "app.tasks.pipeline.generate_report_task": {"queue": "analysis"},
}

# Create Celery app
celery_app = Celery(
    "myswing",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Apply configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task tracking
    task_track_started=True,
    # Reliability: acknowledge after task completes (not before)
    task_acks_late=True,
    # Worker settings
    worker_concurrency=2,  # CPU-bound CV tasks
    worker_prefetch_multiplier=1,  # One task at a time per worker for heavy tasks
    # Time limits
    task_time_limit=120,  # Hard limit: 120 seconds
    task_soft_time_limit=60,  # Soft limit: 60 seconds (matches Req 6.10 with buffer)
    # Retry policy defaults
    task_default_retry_delay=5,
    task_max_retries=2,
    # Queues and routing
    task_queues=task_queues,
    task_routes=task_routes,
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    # Result settings
    result_expires=3600,  # Results expire after 1 hour
)

# Default retry policy for tasks
DEFAULT_RETRY_POLICY = {
    "max_retries": 2,
    "retry_backoff": True,
    "retry_backoff_max": 60,
    "retry_jitter": True,
}

# Auto-discover tasks in the app.tasks package
celery_app.autodiscover_tasks(["app.tasks"])
