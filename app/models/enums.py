"""Enum definitions for the Baseball Swing Analysis service."""

from enum import Enum


class SwingPhase(Enum):
    """Six distinct phases of a baseball swing (Requirement 5.1)."""

    STANCE = "stance"
    LOAD = "load"
    STRIDE = "stride"
    ROTATION = "rotation"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"


class MetricRating(Enum):
    """Rating classification for measured metrics against reference ranges (Requirement 7.7)."""

    BELOW_RANGE = "below_range"
    WITHIN_RANGE = "within_range"
    ABOVE_RANGE = "above_range"


class QualityStatus(Enum):
    """Quality check status for video validation (Requirement 9.8)."""

    PASS = "pass"
    WARNING = "warning"


class BattingDirection(Enum):
    """Batting direction indicating left-handed or right-handed batter (Requirement 2.1)."""

    LEFT = "left"
    RIGHT = "right"
