"""Create or reconcile the fixed user used by local development authentication."""

from typing import Literal
from uuid import UUID

from app.db.models import UserTable
from app.db.session import sync_session_factory

DEVELOPMENT_USER_ID = UUID("00000000-0000-0000-0000-000000000001")
DEVELOPMENT_USER_EMAIL = "development@myswing.local"
DEVELOPMENT_USER_NAME = "Development User"
DevelopmentUserResult = Literal["created", "updated", "unchanged"]


def create_development_user() -> DevelopmentUserResult:
    """Create or reconcile the fixed development user and report the outcome."""
    session = sync_session_factory()
    try:
        existing = session.query(UserTable).filter(UserTable.id == DEVELOPMENT_USER_ID).first()
        if existing:
            if (
                existing.email == DEVELOPMENT_USER_EMAIL
                and existing.name == DEVELOPMENT_USER_NAME
            ):
                print(
                    "Development user unchanged: "
                    f"id={DEVELOPMENT_USER_ID}, email={DEVELOPMENT_USER_EMAIL}"
                )
                return "unchanged"

            existing.email = DEVELOPMENT_USER_EMAIL
            existing.name = DEVELOPMENT_USER_NAME
            session.commit()
            print(
                "Updated development user: "
                f"id={DEVELOPMENT_USER_ID}, email={DEVELOPMENT_USER_EMAIL}"
            )
            return "updated"

        user = UserTable(
            id=DEVELOPMENT_USER_ID,
            email=DEVELOPMENT_USER_EMAIL,
            name=DEVELOPMENT_USER_NAME,
        )
        session.add(user)
        session.commit()
        print(
            "Created development user: "
            f"id={DEVELOPMENT_USER_ID}, email={DEVELOPMENT_USER_EMAIL}"
        )
        return "created"
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> None:
    create_development_user()


if __name__ == "__main__":
    main()
