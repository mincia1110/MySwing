"""Application-wide constants for the Baseball Swing Analysis service."""

from app.core.config import settings

# Supported video formats
SUPPORTED_FORMATS: set[str] = set(settings.supported_formats)
SUPPORTED_MIME_TYPES: set[str] = {"video/mp4", "video/quicktime", "video/x-msvideo"}

# File size limits
MAX_FILE_SIZE_BYTES: int = settings.max_file_size_bytes

# Single-swing video input policy
RECOMMENDED_MIN_DURATION_SEC: float = settings.recommended_min_video_duration_seconds
RECOMMENDED_MAX_DURATION_SEC: float = settings.recommended_max_video_duration_seconds
MAX_DURATION_SEC: float = settings.max_video_duration_seconds
IDEAL_DURATION_SEC: float = settings.ideal_video_duration_seconds

# Backward-compatible alias used by existing validator code/tests.
MAX_DURATION_SECONDS: float = MAX_DURATION_SEC

# Minimum resolution requirements
MIN_RESOLUTION_WIDTH: int = settings.min_resolution_width
MIN_RESOLUTION_HEIGHT: int = settings.min_resolution_height

# Minimum frame rate requirement
MIN_FRAME_RATE: float = settings.min_frame_rate
