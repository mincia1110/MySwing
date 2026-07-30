"""Tests for local development user setup."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.create_test_user import (
    DEVELOPMENT_USER_EMAIL,
    DEVELOPMENT_USER_ID,
    DEVELOPMENT_USER_NAME,
    create_development_user,
)


@patch("scripts.create_test_user.sync_session_factory")
def test_create_development_user_creates_missing_user(mock_session_factory) -> None:
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = None
    mock_session_factory.return_value = session

    assert create_development_user() == "created"
    [user] = session.add.call_args.args
    assert user.id == DEVELOPMENT_USER_ID
    assert user.email == DEVELOPMENT_USER_EMAIL
    assert user.name == DEVELOPMENT_USER_NAME
    session.commit.assert_called_once()
    session.rollback.assert_not_called()
    session.close.assert_called_once()


@patch("scripts.create_test_user.sync_session_factory")
def test_create_development_user_leaves_canonical_user_unchanged(
    mock_session_factory,
) -> None:
    session = MagicMock()
    existing = SimpleNamespace(
        email=DEVELOPMENT_USER_EMAIL,
        name=DEVELOPMENT_USER_NAME,
    )
    session.query.return_value.filter.return_value.first.return_value = existing
    mock_session_factory.return_value = session

    assert create_development_user() == "unchanged"
    assert existing.email == DEVELOPMENT_USER_EMAIL
    assert existing.name == DEVELOPMENT_USER_NAME
    session.add.assert_not_called()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
    session.close.assert_called_once()


@patch("scripts.create_test_user.sync_session_factory")
def test_create_development_user_updates_mismatched_fixed_user(
    mock_session_factory,
) -> None:
    session = MagicMock()
    existing = SimpleNamespace(email="legacy@example.com", name="Legacy User")
    session.query.return_value.filter.return_value.first.return_value = existing
    mock_session_factory.return_value = session

    assert create_development_user() == "updated"
    assert existing.email == DEVELOPMENT_USER_EMAIL
    assert existing.name == DEVELOPMENT_USER_NAME
    session.add.assert_not_called()
    session.commit.assert_called_once()
    session.rollback.assert_not_called()
    session.close.assert_called_once()


@patch("scripts.create_test_user.sync_session_factory")
def test_create_development_user_rolls_back_query_failure(mock_session_factory) -> None:
    session = MagicMock()
    session.query.side_effect = RuntimeError("database unavailable")
    mock_session_factory.return_value = session

    with pytest.raises(RuntimeError, match="database unavailable"):
        create_development_user()

    session.rollback.assert_called_once()
    session.close.assert_called_once()


@patch("scripts.create_test_user.sync_session_factory")
def test_create_development_user_rolls_back_uniqueness_failure(
    mock_session_factory,
) -> None:
    session = MagicMock()
    existing = SimpleNamespace(email="legacy@example.com", name="Legacy User")
    session.query.return_value.filter.return_value.first.return_value = existing
    session.commit.side_effect = RuntimeError("duplicate key value")
    mock_session_factory.return_value = session

    with pytest.raises(RuntimeError, match="duplicate key value"):
        create_development_user()

    session.rollback.assert_called_once()
    session.close.assert_called_once()
