"""User profile data model (Requirement 2.1, 2.5)."""

from dataclasses import dataclass
from typing import Literal, Optional

from app.models.enums import BattingDirection


@dataclass
class UserProfile:
    """User profile containing physical and batting characteristics.

    Required fields (Requirement 2.1):
        - height: User height in cm (100-220)
        - bat_length: Bat length in inches (24-36) or cm (61-91)
        - batting_direction: Left-handed or right-handed

    Optional fields (Requirements 2.3, 2.4):
        - weight: Body weight in kg
        - camera_direction: Camera filming direction
        - age_group: User's age group
        - level: Playing level
        - bat_weight: Bat weight in oz (16-36)
    """

    # Required fields (Requirement 2.1)
    height: float  # cm, valid range: 100-220
    bat_length: float  # inches (24-36) or cm (61-91)
    batting_direction: BattingDirection

    # Optional recommended fields (Requirement 2.3)
    weight: Optional[float] = None  # kg
    camera_direction: Optional[Literal["front", "side", "rear"]] = None

    # Optional fields (Requirement 2.4)
    age_group: Optional[str] = None
    level: Optional[Literal["professional", "college", "high_school", "recreational"]] = None
    bat_weight: Optional[float] = None  # oz, valid range: 16-36
