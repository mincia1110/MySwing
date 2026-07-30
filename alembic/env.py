"""Alembic environment configuration."""

from logging.config import fileConfig

from sqlalchemy import create_engine, pool

# Import all models so they register with Base.metadata
import app.db.models  # noqa: F401
from alembic import context

# Import Base from the new db.session module and ensure all models are loaded
from app.core.config import settings
from app.db.session import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _migration_url() -> str:
    """Use the same configured database as the application and worker."""
    if settings.database_url.startswith("postgresql+asyncpg://"):
        return settings.database_url.replace(
            "postgresql+asyncpg://",
            "postgresql://",
            1,
        )
    return settings.database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    context.configure(
        url=_migration_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = create_engine(
        _migration_url(),
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
