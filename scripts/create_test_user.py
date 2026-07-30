"""Create the fixed user used by local development authentication."""

from uuid import UUID

from app.db.models import UserTable
from app.db.session import sync_session_factory

DEVELOPMENT_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


def create_development_user() -> bool:
    """Create the development user idempotently and return whether it was added."""
    session = sync_session_factory()
    try:
        existing = session.query(UserTable).filter(UserTable.id == DEVELOPMENT_USER_ID).first()
        if existing:
            print(f"User already exists: {existing.email}")
            return False

        user = UserTable(
            id=DEVELOPMENT_USER_ID,
            email="development@myswing.local",
            name="Development User",
        )
        session.add(user)
        session.commit()
        print(
            f"Created development user: id={DEVELOPMENT_USER_ID}, email=development@myswing.local"
        )
        return True
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> None:
    create_development_user()


if __name__ == "__main__":
    main()
