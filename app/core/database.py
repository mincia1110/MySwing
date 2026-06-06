"""Database session and engine configuration.

This module re-exports from app.db.session for backward compatibility.
The canonical location for database setup is now app.db.session.
"""

from app.db.session import Base, get_db, get_engine, sync_session_factory

# Backward-compatible aliases
engine = get_engine()
SessionLocal = sync_session_factory

__all__ = [
    "Base",
    "SessionLocal",
    "engine",
    "get_db",
]
