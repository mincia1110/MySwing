"""Pydantic schemas for user profile API endpoints (Requirements 2.1, 2.5)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.models.enums import BattingDirection


class UserProfileCreate(BaseModel):
    """Schema for creating a user profile.

    Validates required fields and range constraints per Requirement 2.5:
    - height: 100-220 cm
    - bat_length: 24-36 inches (or 61-91 cm)
    - bat_weight: 16-36 oz (if provided)
    """

    # Required fields (Requirement 2.1)
    height: float = Field(
        ...,
        ge=100.0,
        le=220.0,
        description="User height in cm (100-220)",
    )
    bat_length: float = Field(
        ...,
        description="Bat length in inches (24-36) or cm (61-91)",
    )
    batting_direction: Literal["left", "right"] = Field(
        ...,
        description="Batting direction: left-handed or right-handed",
    )

    # Optional recommended fields (Requirement 2.3)
    weight: Optional[float] = Field(
        default=None,
        gt=0,
        description="Body weight in kg",
    )
    camera_direction: Optional[Literal["front", "side", "rear"]] = Field(
        default=None,
        description="Camera filming direction",
    )

    # Optional fields (Requirement 2.4)
    age_group: Optional[str] = Field(
        default=None,
        description="User's age group",
    )
    level: Optional[Literal["professional", "college", "high_school", "recreational"]] = Field(
        default=None,
        description="Playing level",
    )
    bat_weight: Optional[float] = Field(
        default=None,
        ge=16.0,
        le=36.0,
        description="Bat weight in oz (16-36)",
    )

    @field_validator("bat_length")
    @classmethod
    def validate_bat_length_range(cls, v: float) -> float:
        """Validate bat_length is in valid range: 24-36 inches OR 61-91 cm."""
        if not (24 <= v <= 36 or 61 <= v <= 91):
            raise ValueError(
                "bat_length must be between 24-36 inches or 61-91 cm"
            )
        return v


class UserProfileUpdate(BaseModel):
    """Schema for updating a user profile. All fields are optional."""

    height: Optional[float] = Field(
        default=None,
        ge=100.0,
        le=220.0,
        description="User height in cm (100-220)",
    )
    bat_length: Optional[float] = Field(
        default=None,
        description="Bat length in inches (24-36) or cm (61-91)",
    )
    batting_direction: Optional[Literal["left", "right"]] = Field(
        default=None,
        description="Batting direction: left-handed or right-handed",
    )
    weight: Optional[float] = Field(
        default=None,
        gt=0,
        description="Body weight in kg",
    )
    camera_direction: Optional[Literal["front", "side", "rear"]] = Field(
        default=None,
        description="Camera filming direction",
    )
    age_group: Optional[str] = Field(
        default=None,
        description="User's age group",
    )
    level: Optional[Literal["professional", "college", "high_school", "recreational"]] = Field(
        default=None,
        description="Playing level",
    )
    bat_weight: Optional[float] = Field(
        default=None,
        ge=16.0,
        le=36.0,
        description="Bat weight in oz (16-36)",
    )

    @field_validator("bat_length")
    @classmethod
    def validate_bat_length_range(cls, v: Optional[float]) -> Optional[float]:
        """Validate bat_length is in valid range: 24-36 inches OR 61-91 cm."""
        if v is not None and not (24 <= v <= 36 or 61 <= v <= 91):
            raise ValueError(
                "bat_length must be between 24-36 inches or 61-91 cm"
            )
        return v


class UserProfileResponse(BaseModel):
    """Schema for user profile API response."""

    id: str
    user_id: str
    height: float
    bat_length: float
    batting_direction: Literal["left", "right"]
    weight: Optional[float] = None
    camera_direction: Optional[Literal["front", "side", "rear"]] = None
    age_group: Optional[str] = None
    level: Optional[Literal["professional", "college", "high_school", "recreational"]] = None
    bat_weight: Optional[float] = None
