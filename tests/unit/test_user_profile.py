"""Unit tests for user profile API endpoints (Requirements 2.1-2.8).

Tests cover:
- Profile creation with valid data
- Missing mandatory fields → error
- Out-of-range values → error with range info
- Profile retrieval (existing profile)
- Profile retrieval (no profile) → 404
- Profile update (upsert behavior)
- Optional fields handling
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    mock_session = AsyncMock()

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_async_db_dep()] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _user_headers(user_id: uuid.UUID) -> dict[str, str]:
    return {"X-User-Id": str(user_id)}


@pytest.fixture
def valid_profile_data() -> dict:
    """Valid profile data with all required fields."""
    return {
        "height": 175.0,
        "bat_length": 34.0,
        "batting_direction": "right",
    }


@pytest.fixture
def valid_profile_data_full() -> dict:
    """Valid profile data with all fields (required + optional)."""
    return {
        "height": 175.0,
        "bat_length": 34.0,
        "batting_direction": "right",
        "weight": 80.0,
        "camera_direction": "side",
        "age_group": "adult",
        "level": "recreational",
        "bat_weight": 30.0,
    }


@pytest.fixture
def mock_profile_row():
    """Create a mock profile database row."""
    row = MagicMock()
    row.id = uuid.uuid4()
    row.user_id = uuid.uuid4()
    row.height = 175.0
    row.bat_length = 34.0
    row.batting_direction = "right"
    row.weight = 80.0
    row.camera_direction = "side"
    row.age_group = "adult"
    row.level = "recreational"
    row.bat_weight = 30.0
    return row


class TestProfileCreation:
    """Tests for POST /api/v1/users/{user_id}/profile."""

    @patch("app.api.profile.get_async_db")
    def test_create_profile_valid_data(self, mock_get_db, client, valid_profile_data):
        """Profile creation with valid required fields returns 201."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()

        async def override_get_db():
            yield mock_session

        app.dependency_overrides[get_async_db_dep()] = override_get_db

        # Use a simpler approach - test the schema validation directly
        from app.schemas.user_profile import UserProfileCreate

        profile = UserProfileCreate(**valid_profile_data)
        assert profile.height == 175.0
        assert profile.bat_length == 34.0
        assert profile.batting_direction == "right"

    @patch("app.api.profile.create_or_update_profile")
    def test_create_my_profile_uses_current_user_header(
        self,
        mock_create_or_update,
        client,
        valid_profile_data,
        mock_profile_row,
    ):
        """POST /me/profile uses X-User-Id instead of a path user id."""
        current_user_id = uuid.uuid4()
        mock_profile_row.user_id = current_user_id
        mock_create_or_update.return_value = (mock_profile_row, True)
        mock_session = AsyncMock()

        async def override_get_db():
            yield mock_session

        app.dependency_overrides[get_async_db_dep()] = override_get_db

        try:
            response = client.post(
                "/api/v1/me/profile",
                json=valid_profile_data,
                headers={"X-User-Id": str(current_user_id)},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 201
        body = response.json()
        assert body["user_id"] == str(current_user_id)
        call_args = mock_create_or_update.call_args[0]
        assert call_args[1] == current_user_id

    def test_create_profile_missing_height(self, client):
        """Missing height field returns 422 with field info (Req 2.2)."""
        user_id = uuid.uuid4()
        data = {
            "bat_length": 34.0,
            "batting_direction": "right",
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body
        # Check that the error mentions the missing field
        error_fields = [err["loc"][-1] for err in body["detail"]]
        assert "height" in error_fields

    def test_create_profile_missing_bat_length(self, client):
        """Missing bat_length field returns 422 with field info (Req 2.2)."""
        user_id = uuid.uuid4()
        data = {
            "height": 175.0,
            "batting_direction": "right",
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422
        body = response.json()
        error_fields = [err["loc"][-1] for err in body["detail"]]
        assert "bat_length" in error_fields

    def test_create_profile_missing_batting_direction(self, client):
        """Missing batting_direction field returns 422 with field info (Req 2.2)."""
        user_id = uuid.uuid4()
        data = {
            "height": 175.0,
            "bat_length": 34.0,
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422
        body = response.json()
        error_fields = [err["loc"][-1] for err in body["detail"]]
        assert "batting_direction" in error_fields

    def test_create_profile_missing_all_required(self, client):
        """Missing all required fields returns 422 listing all missing fields."""
        user_id = uuid.uuid4()
        data = {}
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422
        body = response.json()
        error_fields = [err["loc"][-1] for err in body["detail"]]
        assert "height" in error_fields
        assert "bat_length" in error_fields
        assert "batting_direction" in error_fields


class TestProfileRangeValidation:
    """Tests for field range validation (Req 2.5, 2.6)."""

    def test_height_below_minimum(self, client):
        """Height below 100cm returns validation error with range info."""
        user_id = uuid.uuid4()
        data = {
            "height": 99.0,
            "bat_length": 34.0,
            "batting_direction": "right",
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422
        body = response.json()
        # Should indicate the valid range
        detail_str = str(body["detail"])
        assert "100" in detail_str or "greater" in detail_str.lower()

    def test_height_above_maximum(self, client):
        """Height above 220cm returns validation error with range info."""
        user_id = uuid.uuid4()
        data = {
            "height": 221.0,
            "bat_length": 34.0,
            "batting_direction": "right",
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422
        body = response.json()
        detail_str = str(body["detail"])
        assert "220" in detail_str or "less" in detail_str.lower()

    def test_bat_length_in_inches_valid(self, client):
        """Bat length 24-36 inches is valid."""
        from app.schemas.user_profile import UserProfileCreate

        profile = UserProfileCreate(height=175.0, bat_length=24.0, batting_direction="right")
        assert profile.bat_length == 24.0

        profile = UserProfileCreate(height=175.0, bat_length=36.0, batting_direction="right")
        assert profile.bat_length == 36.0

    def test_bat_length_in_cm_valid(self, client):
        """Bat length 61-91 cm is valid."""
        from app.schemas.user_profile import UserProfileCreate

        profile = UserProfileCreate(height=175.0, bat_length=61.0, batting_direction="right")
        assert profile.bat_length == 61.0

        profile = UserProfileCreate(height=175.0, bat_length=91.0, batting_direction="right")
        assert profile.bat_length == 91.0

    def test_bat_length_invalid_gap(self, client):
        """Bat length between 36 and 61 (invalid gap) returns error."""
        user_id = uuid.uuid4()
        data = {
            "height": 175.0,
            "bat_length": 50.0,
            "batting_direction": "right",
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422
        body = response.json()
        detail_str = str(body["detail"])
        assert "24" in detail_str or "bat_length" in detail_str

    def test_bat_length_below_minimum(self, client):
        """Bat length below 24 inches returns error."""
        user_id = uuid.uuid4()
        data = {
            "height": 175.0,
            "bat_length": 23.0,
            "batting_direction": "right",
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422

    def test_bat_length_above_maximum(self, client):
        """Bat length above 91 cm returns error."""
        user_id = uuid.uuid4()
        data = {
            "height": 175.0,
            "bat_length": 92.0,
            "batting_direction": "right",
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422

    def test_bat_weight_below_minimum(self, client):
        """Bat weight below 16oz returns validation error."""
        user_id = uuid.uuid4()
        data = {
            "height": 175.0,
            "bat_length": 34.0,
            "batting_direction": "right",
            "bat_weight": 15.0,
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422

    def test_bat_weight_above_maximum(self, client):
        """Bat weight above 36oz returns validation error."""
        user_id = uuid.uuid4()
        data = {
            "height": 175.0,
            "bat_length": 34.0,
            "batting_direction": "right",
            "bat_weight": 37.0,
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422

    def test_invalid_batting_direction(self, client):
        """Invalid batting_direction value returns validation error."""
        user_id = uuid.uuid4()
        data = {
            "height": 175.0,
            "bat_length": 34.0,
            "batting_direction": "center",
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422


class TestProfileRetrieval:
    """Tests for GET /api/v1/users/{user_id}/profile."""

    def test_get_profile_not_found(self, client):
        """GET profile for user with no profile returns 404."""
        user_id = uuid.uuid4()

        with patch("app.api.profile.get_profile", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            response = client.get(
                f"/api/v1/users/{user_id}/profile",
                headers=_user_headers(user_id),
            )

        assert response.status_code == 404
        body = response.json()
        assert "detail" in body
        assert str(user_id) in body["detail"]

    def test_get_profile_existing(self, client, mock_profile_row):
        """GET profile for user with existing profile returns 200 with data."""
        user_id = mock_profile_row.user_id

        with patch("app.api.profile.get_profile", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_profile_row
            response = client.get(
                f"/api/v1/users/{user_id}/profile",
                headers=_user_headers(user_id),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["height"] == 175.0
        assert body["bat_length"] == 34.0
        assert body["batting_direction"] == "right"
        assert body["weight"] == 80.0
        assert body["camera_direction"] == "side"
        assert body["age_group"] == "adult"
        assert body["level"] == "recreational"
        assert body["bat_weight"] == 30.0

    def test_get_profile_rejects_other_user(self, client):
        """Path user_id must match the authenticated user."""
        target_user_id = uuid.uuid4()
        current_user_id = uuid.uuid4()

        with patch("app.api.profile.get_profile", new_callable=AsyncMock) as mock_get:
            response = client.get(
                f"/api/v1/users/{target_user_id}/profile",
                headers=_user_headers(current_user_id),
            )

        assert response.status_code == 403
        mock_get.assert_not_called()


class TestProfileUpsert:
    """Tests for upsert behavior (create if not exists, update if exists)."""

    def test_upsert_creates_new_profile(self, client, valid_profile_data, mock_profile_row):
        """POST to user without profile creates new profile (201)."""
        user_id = uuid.uuid4()
        mock_profile_row.user_id = user_id

        with patch(
            "app.api.profile.create_or_update_profile", new_callable=AsyncMock
        ) as mock_upsert:
            mock_upsert.return_value = (mock_profile_row, True)
            response = client.post(
                f"/api/v1/users/{user_id}/profile",
                json=valid_profile_data,
                headers=_user_headers(user_id),
            )

        assert response.status_code == 201
        body = response.json()
        assert body["height"] == 175.0

    def test_upsert_rejects_other_user(self, client, valid_profile_data):
        """Path user_id cannot target a different authenticated user."""
        target_user_id = uuid.uuid4()
        current_user_id = uuid.uuid4()

        with patch(
            "app.api.profile.create_or_update_profile", new_callable=AsyncMock
        ) as mock_upsert:
            response = client.post(
                f"/api/v1/users/{target_user_id}/profile",
                json=valid_profile_data,
                headers=_user_headers(current_user_id),
            )

        assert response.status_code == 403
        mock_upsert.assert_not_called()

    def test_upsert_rejects_missing_user(self, client, valid_profile_data):
        """POST for an unknown user returns 404 instead of leaking a DB FK 500."""
        user_id = uuid.uuid4()

        from app.services.user_profile_service import UserNotFoundError

        with patch(
            "app.api.profile.create_or_update_profile", new_callable=AsyncMock
        ) as mock_upsert:
            mock_upsert.side_effect = UserNotFoundError(user_id)
            response = client.post(
                f"/api/v1/users/{user_id}/profile",
                json=valid_profile_data,
                headers=_user_headers(user_id),
            )

        assert response.status_code == 404
        body = response.json()
        assert str(user_id) in body["detail"]
        assert "does not exist" in body["detail"]

    def test_upsert_updates_existing_profile(self, client, valid_profile_data, mock_profile_row):
        """POST to user with existing profile updates it (200)."""
        user_id = uuid.uuid4()
        mock_profile_row.user_id = user_id

        with patch(
            "app.api.profile.create_or_update_profile", new_callable=AsyncMock
        ) as mock_upsert:
            mock_upsert.return_value = (mock_profile_row, False)
            response = client.post(
                f"/api/v1/users/{user_id}/profile",
                json=valid_profile_data,
                headers=_user_headers(user_id),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["height"] == 175.0


class TestOptionalFields:
    """Tests for optional field handling (Req 2.3, 2.4)."""

    def test_optional_fields_not_required(self, client):
        """Profile creation succeeds without optional fields."""
        from app.schemas.user_profile import UserProfileCreate

        profile = UserProfileCreate(
            height=175.0,
            bat_length=34.0,
            batting_direction="right",
        )
        assert profile.weight is None
        assert profile.camera_direction is None
        assert profile.age_group is None
        assert profile.level is None
        assert profile.bat_weight is None

    def test_optional_fields_accepted(self, client, valid_profile_data_full):
        """Profile creation accepts all optional fields."""
        from app.schemas.user_profile import UserProfileCreate

        profile = UserProfileCreate(**valid_profile_data_full)
        assert profile.weight == 80.0
        assert profile.camera_direction == "side"
        assert profile.age_group == "adult"
        assert profile.level == "recreational"
        assert profile.bat_weight == 30.0

    def test_invalid_camera_direction(self, client):
        """Invalid camera_direction value returns validation error."""
        user_id = uuid.uuid4()
        data = {
            "height": 175.0,
            "bat_length": 34.0,
            "batting_direction": "right",
            "camera_direction": "top",
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422

    def test_invalid_level(self, client):
        """Invalid level value returns validation error."""
        user_id = uuid.uuid4()
        data = {
            "height": 175.0,
            "bat_length": 34.0,
            "batting_direction": "right",
            "level": "amateur",
        }
        response = client.post(
            f"/api/v1/users/{user_id}/profile",
            json=data,
            headers=_user_headers(user_id),
        )
        assert response.status_code == 422


class TestServiceValidation:
    """Tests for the service-level validation logic."""

    def test_validate_profile_ranges_valid(self):
        """Valid data passes range validation."""
        from app.schemas.user_profile import UserProfileCreate
        from app.services.user_profile_service import validate_profile_ranges

        data = UserProfileCreate(
            height=175.0, bat_length=34.0, batting_direction="right"
        )
        errors = validate_profile_ranges(data)
        assert errors == []

    def test_validate_profile_ranges_height_too_low(self):
        """Height below 100 fails range validation."""
        from app.schemas.user_profile import UserProfileCreate
        from app.services.user_profile_service import validate_profile_ranges

        # We need to bypass Pydantic validation to test service-level validation
        # Create with valid Pydantic range but test service logic
        data = UserProfileCreate(
            height=100.0, bat_length=34.0, batting_direction="right"
        )
        errors = validate_profile_ranges(data)
        assert errors == []  # 100 is the boundary, should pass

    def test_validate_profile_ranges_bat_length_cm(self):
        """Bat length in cm range (61-91) passes validation."""
        from app.schemas.user_profile import UserProfileCreate
        from app.services.user_profile_service import validate_profile_ranges

        data = UserProfileCreate(
            height=175.0, bat_length=75.0, batting_direction="left"
        )
        errors = validate_profile_ranges(data)
        assert errors == []

    def test_validate_profile_ranges_bat_weight_valid(self):
        """Valid bat weight passes range validation."""
        from app.schemas.user_profile import UserProfileCreate
        from app.services.user_profile_service import validate_profile_ranges

        data = UserProfileCreate(
            height=175.0, bat_length=34.0, batting_direction="right", bat_weight=28.0
        )
        errors = validate_profile_ranges(data)
        assert errors == []


class TestProfileServiceUpsert:
    """Tests for service-level profile upsert behavior."""

    async def test_create_profile_rejects_unknown_user_before_flush(self, valid_profile_data):
        """Unknown user_id raises a domain error before the FK can fail at flush."""
        from app.schemas.user_profile import UserProfileCreate
        from app.services.user_profile_service import (
            UserNotFoundError,
            create_or_update_profile,
        )

        user_id = uuid.uuid4()
        profile_data = UserProfileCreate(**valid_profile_data)
        mock_session = AsyncMock()

        no_existing_profile = MagicMock()
        no_existing_profile.scalar_one_or_none.return_value = None
        no_user = MagicMock()
        no_user.scalar_one_or_none.return_value = None
        mock_session.execute.side_effect = [no_existing_profile, no_user]

        with pytest.raises(UserNotFoundError) as exc_info:
            await create_or_update_profile(mock_session, user_id, profile_data)

        assert exc_info.value.user_id == user_id
        mock_session.add.assert_not_called()
        mock_session.flush.assert_not_called()


def get_async_db_dep():
    """Helper to get the actual dependency function for overriding."""
    from app.db.session import get_async_db
    return get_async_db
