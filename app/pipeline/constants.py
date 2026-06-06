"""Constants for the analysis pipeline modules."""

# MediaPipe Pose landmark indices → essential keypoint names
# MediaPipe provides 33 landmarks; we map to 17+ essential keypoints for swing analysis.
MEDIAPIPE_TO_ESSENTIAL: dict[int, str] = {
    0: "head",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

# "spine" is a calculated midpoint of left_shoulder (11) and right_shoulder (12)
# It is not directly mapped from a single MediaPipe landmark.
SPINE_SOURCE_INDICES: tuple[int, int] = (11, 12)
SPINE_KEYPOINT_NAME: str = "spine"

# Complete list of essential keypoint names (17 total: 13 direct + 1 calculated + 3 additional)
ESSENTIAL_KEYPOINT_NAMES: list[str] = [
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "spine",  # calculated midpoint of left_shoulder and right_shoulder
    "left_ear",
    "right_ear",
    "neck",  # calculated midpoint of shoulders at neck level
]

# Additional MediaPipe landmarks that can be useful (ears for head orientation)
ADDITIONAL_MEDIAPIPE_LANDMARKS: dict[int, str] = {
    7: "left_ear",
    8: "right_ear",
}

# Default minimum confidence threshold for keypoint filtering
DEFAULT_MIN_CONFIDENCE: float = 0.5

# Maximum processing time per frame in milliseconds
MAX_FRAME_PROCESSING_TIME_MS: float = 100.0
