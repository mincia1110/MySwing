"""Application-wide constants for the Baseball Swing Analysis service."""

# Supported video formats
SUPPORTED_FORMATS: set[str] = {"mp4", "mov", "avi"}
SUPPORTED_MIME_TYPES: set[str] = {"video/mp4", "video/quicktime", "video/x-msvideo"}

# File size limits
MAX_FILE_SIZE_BYTES: int = 500 * 1024 * 1024  # 500MB

# Video duration limits
MAX_DURATION_SECONDS: float = 300.0  # 5 minutes

# Minimum resolution requirements
MIN_RESOLUTION_WIDTH: int = 1280
MIN_RESOLUTION_HEIGHT: int = 720

# Minimum frame rate requirement
MIN_FRAME_RATE: float = 30.0
