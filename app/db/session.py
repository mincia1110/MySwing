"""Database engine and session configuration.

Provides both synchronous and asynchronous database access.
The async engine is the primary interface for the FastAPI application.
The sync engine is available for scripts and Celery workers.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class for all ORM models."""

    pass


def _get_async_url(url: str) -> str:
    """Convert a standard PostgreSQL URL to an asyncpg URL."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _get_sync_url(url: str) -> str:
    """Ensure URL uses standard psycopg2 driver for sync access."""
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url


# Sync engine (for scripts, Celery workers, Alembic)
_sync_engine = create_engine(
    _get_sync_url(settings.database_url),
    echo=settings.debug,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# Session factories
sync_session_factory = sessionmaker(
    bind=_sync_engine,
    autocommit=False,
    autoflush=False,
)


def get_engine():
    """Get the synchronous engine instance."""
    return _sync_engine


def _create_async_engine():
    """Lazily create the async engine (requires asyncpg to be installed)."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    engine = create_async_engine(
        _get_async_url(settings.database_url),
        echo=settings.debug,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return engine, factory


# Lazy async engine/session - created on first access
_async_engine_instance = None
_async_session_factory_instance = None


def _get_async_components():
    """Get or create async engine and session factory."""
    global _async_engine_instance, _async_session_factory_instance
    if _async_engine_instance is None:
        _async_engine_instance, _async_session_factory_instance = _create_async_engine()
    return _async_engine_instance, _async_session_factory_instance


def async_session_factory():
    """Get the async session factory (creates engine on first call)."""
    _, factory = _get_async_components()
    return factory


async def get_async_db() -> AsyncGenerator["AsyncSession", None]:
    """FastAPI dependency for async database sessions."""
    _, factory = _get_async_components()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_db() -> Generator[Session, None, None]:
    """Dependency for synchronous database sessions (scripts, workers)."""
    session = sync_session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
