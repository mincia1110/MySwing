"""Shared API dependencies."""

from typing import Annotated
from uuid import UUID

from fastapi import Header, HTTPException, status

DEV_DEFAULT_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


async def get_current_user_id(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> UUID:
    """Resolve the current user for MVP/local development.

    This is intentionally small: production auth can replace this dependency
    without letting route handlers trust path/query/body user IDs.
    """
    if x_user_id is None:
        return DEV_DEFAULT_USER_ID

    try:
        return UUID(x_user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-User-Id header",
        )
