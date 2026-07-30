"""Tests for Alembic URL precedence and migration wiring."""

from __future__ import annotations

import configparser
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPOSITORY_ROOT / "alembic" / "env.py"
INI_PATH = REPOSITORY_ROOT / "alembic.ini"


class _FakeConfig:
    def __init__(self, sqlalchemy_url: str | None) -> None:
        self.config_file_name = None
        self._sqlalchemy_url = sqlalchemy_url

    def get_main_option(self, name: str) -> str | None:
        assert name == "sqlalchemy.url"
        return self._sqlalchemy_url


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _load_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    alembic_url: str | None,
    settings_url: str,
    offline: bool,
) -> tuple[dict[str, object], SimpleNamespace, MagicMock, MagicMock, object]:
    config = _FakeConfig(alembic_url)
    transaction = MagicMock()
    transaction.__enter__.return_value = None
    transaction.__exit__.return_value = False

    context = SimpleNamespace(
        config=config,
        configure=MagicMock(),
        begin_transaction=MagicMock(return_value=transaction),
        run_migrations=MagicMock(),
        is_offline_mode=MagicMock(return_value=offline),
    )

    connection = object()
    connection_context = MagicMock()
    connection_context.__enter__.return_value = connection
    connection_context.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = connection_context
    create_engine = MagicMock(return_value=engine)
    null_pool = object()

    sqlalchemy_module = ModuleType("sqlalchemy")
    sqlalchemy_module.create_engine = create_engine  # type: ignore[attr-defined]
    sqlalchemy_module.pool = SimpleNamespace(NullPool=null_pool)  # type: ignore[attr-defined]

    alembic_module = ModuleType("alembic")
    alembic_module.context = context  # type: ignore[attr-defined]

    app_module = _package("app")
    app_db_module = _package("app.db")
    app_core_module = _package("app.core")
    models_module = ModuleType("app.db.models")
    session_module = ModuleType("app.db.session")
    session_module.Base = SimpleNamespace(metadata=object())  # type: ignore[attr-defined]
    config_module = ModuleType("app.core.config")
    config_module.settings = SimpleNamespace(database_url=settings_url)  # type: ignore[attr-defined]

    modules = {
        "sqlalchemy": sqlalchemy_module,
        "alembic": alembic_module,
        "app": app_module,
        "app.db": app_db_module,
        "app.db.models": models_module,
        "app.db.session": session_module,
        "app.core": app_core_module,
        "app.core.config": config_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    namespace = runpy.run_path(str(ENV_PATH))
    return namespace, context, create_engine, engine, connection


def test_checked_in_config_does_not_override_application_settings() -> None:
    parser = configparser.ConfigParser()
    parser.read(INI_PATH)

    assert parser.get("alembic", "sqlalchemy.url", fallback="").strip() == ""


def test_migration_url_falls_back_to_application_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, *_ = _load_env(
        monkeypatch,
        alembic_url=None,
        settings_url="postgresql://app/database",
        offline=True,
    )

    assert namespace["_migration_url"]() == "postgresql://app/database"  # type: ignore[operator]


def test_migration_url_prefers_explicit_alembic_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, *_ = _load_env(
        monkeypatch,
        alembic_url="sqlite:///alternate.db",
        settings_url="postgresql://app/database",
        offline=True,
    )

    assert namespace["_migration_url"]() == "sqlite:///alternate.db"  # type: ignore[operator]


@pytest.mark.parametrize(
    ("alembic_url", "settings_url", "expected"),
    [
        (None, "postgresql+asyncpg://app/database", "postgresql://app/database"),
        (
            "postgresql+asyncpg://override/database",
            "postgresql://app/database",
            "postgresql://override/database",
        ),
    ],
)
def test_migration_url_converts_asyncpg_after_selecting_effective_source(
    monkeypatch: pytest.MonkeyPatch,
    alembic_url: str | None,
    settings_url: str,
    expected: str,
) -> None:
    namespace, *_ = _load_env(
        monkeypatch,
        alembic_url=alembic_url,
        settings_url=settings_url,
        offline=True,
    )

    assert namespace["_migration_url"]() == expected  # type: ignore[operator]


def test_offline_migrations_use_shared_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    _, context, create_engine, _, _ = _load_env(
        monkeypatch,
        alembic_url="postgresql+asyncpg://override/database",
        settings_url="postgresql://app/database",
        offline=True,
    )

    create_engine.assert_not_called()
    assert context.configure.call_args.kwargs["url"] == "postgresql://override/database"
    context.run_migrations.assert_called_once()


def test_online_migrations_use_shared_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    _, context, create_engine, _, connection = _load_env(
        monkeypatch,
        alembic_url=None,
        settings_url="postgresql+asyncpg://app/database",
        offline=False,
    )

    assert create_engine.call_args.args == ("postgresql://app/database",)
    assert "poolclass" in create_engine.call_args.kwargs
    context.configure.assert_called_once()
    assert context.configure.call_args.kwargs["connection"] is connection
    context.run_migrations.assert_called_once()
