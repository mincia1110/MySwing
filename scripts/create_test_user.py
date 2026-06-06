"""Create a test user for local development."""
from app.db.session import sync_session_factory
from app.db.models import UserTable
import uuid

def main():
    session = sync_session_factory()
    try:
        user_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
        existing = session.query(UserTable).filter(UserTable.id == user_id).first()
        if existing:
            print(f"User already exists: {existing.email}")
            return
        
        user = UserTable(
            id=user_id,
            email="test@example.com",
            name="Test User",
        )
        session.add(user)
        session.commit()
        print(f"Created test user: id={user_id}, email=test@example.com")
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    main()
