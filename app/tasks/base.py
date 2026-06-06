"""Base task class with retry logic and error handling.

Provides a custom Celery base task with built-in retry policy,
on_failure logging/status updates, and on_retry logging.
"""

import logging

from celery import Task

from app.core.celery_app import DEFAULT_RETRY_POLICY, celery_app

logger = logging.getLogger(__name__)


class BaseAnalysisTask(Task):
    """Custom base task for swing analysis pipeline tasks.

    Provides:
    - Automatic retry with exponential backoff (max_retries=2)
    - on_failure handler for logging and status updates
    - on_retry handler for logging retry attempts
    """

    # Retry policy
    autoretry_for = (Exception,)
    max_retries = DEFAULT_RETRY_POLICY["max_retries"]
    retry_backoff = DEFAULT_RETRY_POLICY["retry_backoff"]
    retry_backoff_max = DEFAULT_RETRY_POLICY["retry_backoff_max"]
    retry_jitter = DEFAULT_RETRY_POLICY["retry_jitter"]

    def on_failure(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo) -> None:
        """Called when the task fails after all retries are exhausted.

        Logs the failure and can be extended to update analysis status in DB.
        """
        logger.error(
            "Task %s[%s] failed after %d retries: %s",
            self.name,
            task_id,
            self.max_retries,
            str(exc),
            exc_info=einfo,
        )

    def on_retry(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo) -> None:
        """Called when the task is retried.

        Logs the retry attempt with the exception that caused it.
        """
        logger.warning(
            "Task %s[%s] retrying (attempt %d/%d): %s",
            self.name,
            task_id,
            self.request.retries + 1,
            self.max_retries,
            str(exc),
        )

    def on_success(self, retval, task_id: str, args: tuple, kwargs: dict) -> None:
        """Called when the task succeeds.

        Logs successful completion.
        """
        logger.info(
            "Task %s[%s] completed successfully.",
            self.name,
            task_id,
        )


# Register the base task with the celery app
BaseAnalysisTask = celery_app.register_task(BaseAnalysisTask())  # type: ignore[assignment]
