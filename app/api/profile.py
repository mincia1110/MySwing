"""User profile API endpoints (Requirements 2.1-2.8).

Provides endpoints for creating/updating and retrieving user profiles.
Profiles store physical and batting characteristics used for swing analysis calibration.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id
from app.db.session import get_async_db
from app.schemas.user_profile import UserProfileCreate, UserProfileResponse
from app.services.user_profile_service import (
    UserNotFoundError,
    create_or_update_profile,
    get_profile,
)

router = APIRouter(tags=["profile"])


def _profile_response(profile) -> UserProfileResponse:
    return UserProfileResponse(
        id=str(profile.id),
        user_id=str(profile.user_id),
        height=profile.height,
        bat_length=profile.bat_length,
        batting_direction=profile.batting_direction,
        weight=profile.weight,
        camera_direction=profile.camera_direction,
        age_group=profile.age_group,
        level=profile.level,
        bat_weight=profile.bat_weight,
    )


@router.post(
    "/users/{user_id}/profile",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or update user profile",
    description="Create a new user profile or update an existing one (upsert). "
    "Required fields: height, bat_length, batting_direction. "
    "Returns 201 for new profiles, 200 for updates.",
    responses={
        200: {"description": "Profile updated successfully"},
        201: {"description": "Profile created successfully"},
        400: {"description": "Validation error - missing fields or out-of-range values"},
        422: {"description": "Request body validation error"},
    },
)
async def create_or_update_user_profile(
    user_id: UUID,
    profile_data: UserProfileCreate,
    db: AsyncSession = Depends(get_async_db),
) -> UserProfileResponse:
    """Create or update a user profile.

    Validates mandatory fields (height, bat_length, batting_direction) and
    range constraints per Requirement 2.5:
    - height: 100-220 cm
    - bat_length: 24-36 inches or 61-91 cm
    - bat_weight: 16-36 oz (if provided)

    If any mandatory field is missing, returns 422 with field details (Req 2.2).
    If values are out of range, returns 400 with acceptable range info (Req 2.6).
    """
    try:
        profile, created = await create_or_update_profile(db, user_id, profile_data)
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValueError as e:
        # Range validation error (Requirement 2.6)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    response = _profile_response(profile)

    if not created:
        # Return 200 for updates instead of 201
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content=response.model_dump(),
            status_code=status.HTTP_200_OK,
        )

    return response


@router.get(
    "/users/{user_id}/profile",
    response_model=UserProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get user profile",
    description="Retrieve a user's profile for pre-populating form fields (Requirement 2.7).",
    responses={
        200: {"description": "Profile retrieved successfully"},
        404: {"description": "No profile found for this user"},
    },
)
async def get_user_profile(
    user_id: UUID,
    db: AsyncSession = Depends(get_async_db),
) -> UserProfileResponse:
    """Retrieve a user's profile.

    Used for pre-populating profile fields with previously saved values (Req 2.7).
    Returns 404 if no profile exists for the given user.
    """
    profile = await get_profile(db, user_id)

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No profile found for user {user_id}",
        )

    return _profile_response(profile)


@router.post(
    "/me/profile",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or update current user's profile",
)
async def create_or_update_my_profile(
    profile_data: UserProfileCreate,
    current_user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_async_db),
) -> UserProfileResponse:
    """Create or update the current user's profile without trusting a path ID."""
    return await create_or_update_user_profile(current_user_id, profile_data, db)


@router.get(
    "/me/profile",
    response_model=UserProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current user's profile",
)
async def get_my_profile(
    current_user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_async_db),
) -> UserProfileResponse:
    """Retrieve the current user's profile without trusting a path ID."""
    return await get_user_profile(current_user_id, db)
