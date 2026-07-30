"""Tests for local development user setup."""

from unittest.mock import MagicMock, patch

import pytest

from scripts.create_test_user import DEVELOPMENT_USER_ID, create_development_user


@patch("scripts.create_test_user.sync_session_factory")
def test_create_development_user_is_idempotent(mock_session_factory) -> None:
    session = MagicMock()
    existing = MagicMock(email="development@myswing.local")
    session.query.return_value.filter.return_value.first.return_value = existing
    mock_session_factory.return_value = session

    assert create_development_user() is False
    session.add.assert_not_called()
    session.commit.assert_not_called()
    session.close.assert_called_once()


@patch("scripts.create_test_user.sync_session_factory")
def test_create_development_user_inserts_expected_identity(mock_session_factory) -> None:
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = None
    mock_session_factory.return_value = session

    assert create_development_user() is True
    [user] = session.add.call_args.args
    assert user.id == DEVELOPMENT_USER_ID
    assert user.email == "development@myswing.local"
    session.commit.assert_called_once()
    session.close.assert_called_once()


@patch("scripts.create_test_user.sync_session_factory")
def test_create_development_user_rolls_back_and_raises(mock_session_factory) -> None:
    session = MagicMock()
    session.query.side_effect = RuntimeError("database unavailable")
    mock_session_factory.return_value = session

    with pytest.raises(RuntimeError, match="database unavailable"):
        create_development_user()

    session.rollback.assert_called_once()
    session.close.assert_called_once()
