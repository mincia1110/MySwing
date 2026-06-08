"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "MySwing"
    debug: bool = False

    # Database
    database_url: str = "postgresql://myswing:myswing@localhost:5432/myswing"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # S3/MinIO
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_name: str = "myswing-videos"
    s3_region: str = "us-east-1"

    # CORS - frontend dev server origins by default
    cors_allow_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Video Upload Constraints
    max_file_size_bytes: int = 500 * 1024 * 1024  # 500MB
    recommended_min_video_duration_seconds: float = 3.0
    recommended_max_video_duration_seconds: float = 7.0
    max_video_duration_seconds: float = 10.0
    ideal_video_duration_seconds: float = 5.0
    min_resolution_width: int = 1280
    min_resolution_height: int = 720
    min_frame_rate: float = 30.0
    supported_formats: list[str] = ["mp4", "mov", "avi"]

    model_config = {"env_prefix": "MYSWING_", "env_file": ".env"}


settings = Settings()
