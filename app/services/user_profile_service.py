"""User profile service for CRUD operations and validation.

Handles profile creation, update (upsert), and retrieval.
Validates field ranges per Requirement 2.5.
"""

from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import UserProfileTable, UserTable
from app.models.user_profile import UserProfile
from app.schemas.user_profile import UserProfileCreate


class ValidationError(Exception):
    """Raised when profile data fails range validation."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(message)


class UserNotFoundError(Exception):
    """Raised when a profile is created for a user row that does not exist."""

    def __init__(self, user_id: UUID):
        self.user_id = user_id
        super().__init__(f"User {user_id} does not exist")


def validate_profile_ranges(data: UserProfileCreate) -> list[str]:
    """Validate profile field ranges per Requirement 2.5.

    Returns a list of error messages for any out-of-range values.

    Validation rules:
    - height: 100-220 cm
    - bat_length: 24-36 inches OR 61-91 cm
    - bat_weight (if provided): 16-36 oz
    """
    errors: list[str] = []

    # Height validation: 100-220 cm
    if data.height < 100 or data.height > 220:
        errors.append(
            f"height must be between 100 and 220 cm (received: {data.height})"
        )

    # Bat length validation: 24-36 inches OR 61-91 cm
    # Values 24-36 are treated as inches, 61-91 as cm
    # Values between 36 and 61 or above 91 are invalid
    if not (24 <= data.bat_length <= 36 or 61 <= data.bat_length <= 91):
        errors.append(
            f"bat_length must be between 24-36 inches or 61-91 cm (received: {data.bat_length})"
        )

    # Bat weight validation (optional): 16-36 oz
    if data.bat_weight is not None:
        if data.bat_weight < 16 or data.bat_weight > 36:
            errors.append(
                f"bat_weight must be between 16 and 36 oz (received: {data.bat_weight})"
            )

    return errors


def _table_to_domain(row: UserProfileTable) -> UserProfile:
    """Convert a database row to a domain model."""
    from app.models.enums import BattingDirection

    return UserProfile(
        height=row.height,
        bat_length=row.bat_length,
        batting_direction=BattingDirection(row.batting_direction),
        weight=row.weight,
        camera_direction=row.camera_direction,
        age_group=row.age_group,
        level=row.level,
        bat_weight=row.bat_weight,
    )


async def create_or_update_profile(
    db: AsyncSession, user_id: UUID, data: UserProfileCreate
) -> tuple[UserProfileTable, bool]:
    """Create or update a user profile (upsert).

    Args:
        db: Async database session
        user_id: The user's UUID
        data: Validated profile data

    Returns:
        Tuple of (profile_table_row, created) where created is True if new profile was created.

    Raises:
        ValueError: If field values are out of valid ranges.
    """
    # Validate ranges (Requirement 2.5, 2.6)
    range_errors = validate_profile_ranges(data)
    if range_errors:
        raise ValueError("; ".join(range_errors))

    # Check if profile already exists for this user
    stmt = select(UserProfileTable).where(UserProfileTable.user_id == user_id)
    result = await db.execute(stmt)
    existing_profile = result.scalar_one_or_none()

    if existing_profile:
        # Update existing profile
        existing_profile.height = data.height
        existing_profile.bat_length = data.bat_length
        existing_profile.batting_direction = data.batting_direction
        existing_profile.weight = data.weight
        existing_profile.camera_direction = data.camera_direction
        existing_profile.age_group = data.age_group
        existing_profile.level = data.level
        existing_profile.bat_weight = data.bat_weight
        await db.flush()
        return existing_profile, False
    else:
        # Avoid surfacing a database foreign-key violation as a generic 500.
        user_stmt = select(UserTable.id).where(UserTable.id == user_id)
        user_result = await db.execute(user_stmt)
        if user_result.scalar_one_or_none() is None:
            raise UserNotFoundError(user_id)

        # Create new profile
        new_profile = UserProfileTable(
            user_id=user_id,
            height=data.height,
            bat_length=data.bat_length,
            batting_direction=data.batting_direction,
            weight=data.weight,
            camera_direction=data.camera_direction,
            age_group=data.age_group,
            level=data.level,
            bat_weight=data.bat_weight,
        )
        db.add(new_profile)
        await db.flush()
        return new_profile, True


async def get_profile(db: AsyncSession, user_id: UUID) -> Optional[UserProfileTable]:
    """Retrieve a user's profile.

    Args:
        db: Async database session
        user_id: The user's UUID

    Returns:
        The user profile table row, or None if no profile exists.
    """
    stmt = select(UserProfileTable).where(UserProfileTable.user_id == user_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()
