"""Unit tests for SQLAlchemy ORM models (Task 1.3).

Tests model definitions, table metadata, column types, and relationships
without requiring a running database.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect

from app.db.models import (
    AnalysisResultTable,
    AnalysisTable,
    QualityCheckTable,
    ReferenceDataTable,
    UserProfileTable,
    UserTable,
    VideoTable,
)
from app.db.session import Base


class TestBaseMetadata:
    """Test that all expected tables are registered in Base.metadata."""

    def test_all_tables_registered(self):
        table_names = set(Base.metadata.tables.keys())
        expected = {
            "users",
            "user_profiles",
            "videos",
            "analyses",
            "analysis_results",
            "quality_checks",
            "reference_data",
        }
        assert expected == table_names


class TestUserTable:
    """Test UserTable model definition."""

    def test_tablename(self):
        assert UserTable.__tablename__ == "users"

    def test_columns(self):
        mapper = inspect(UserTable)
        column_names = {col.key for col in mapper.columns}
        assert column_names == {"id", "email", "name", "created_at"}

    def test_id_is_uuid_primary_key(self):
        mapper = inspect(UserTable)
        pk_cols = [col.key for col in mapper.columns if col.primary_key]
        assert pk_cols == ["id"]

    def test_email_is_unique(self):
        table = UserTable.__table__
        email_col = table.c.email
        assert email_col.unique is True

    def test_instance_creation(self):
        user = UserTable(
            id=uuid.uuid4(),
            email="test@example.com",
            name="Test User",
        )
        assert user.email == "test@example.com"
        assert user.name == "Test User"


class TestUserProfileTable:
    """Test UserProfileTable model definition."""

    def test_tablename(self):
        assert UserProfileTable.__tablename__ == "user_profiles"

    def test_columns(self):
        mapper = inspect(UserProfileTable)
        column_names = {col.key for col in mapper.columns}
        expected = {
            "id",
            "user_id",
            "height",
            "bat_length",
            "batting_direction",
            "weight",
            "camera_direction",
            "age_group",
            "level",
            "bat_weight",
            "created_at",
            "updated_at",
        }
        assert column_names == expected

    def test_user_id_foreign_key(self):
        table = UserProfileTable.__table__
        user_id_col = table.c.user_id
        fk = list(user_id_col.foreign_keys)[0]
        assert str(fk.column) == "users.id"

    def test_user_id_is_unique(self):
        table = UserProfileTable.__table__
        user_id_col = table.c.user_id
        assert user_id_col.unique is True

    def test_required_fields_not_nullable(self):
        table = UserProfileTable.__table__
        assert table.c.height.nullable is False
        assert table.c.bat_length.nullable is False
        assert table.c.batting_direction.nullable is False

    def test_optional_fields_nullable(self):
        table = UserProfileTable.__table__
        assert table.c.weight.nullable is True
        assert table.c.camera_direction.nullable is True
        assert table.c.age_group.nullable is True
        assert table.c.level.nullable is True
        assert table.c.bat_weight.nullable is True

    def test_instance_creation(self):
        profile = UserProfileTable(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            height=175.0,
            bat_length=33.0,
            batting_direction="right",
        )
        assert profile.height == 175.0
        assert profile.bat_length == 33.0
        assert profile.batting_direction == "right"


class TestVideoTable:
    """Test VideoTable model definition."""

    def test_tablename(self):
        assert VideoTable.__tablename__ == "videos"

    def test_columns(self):
        mapper = inspect(VideoTable)
        column_names = {col.key for col in mapper.columns}
        expected = {
            "id",
            "user_id",
            "file_key",
            "file_name",
            "file_size_bytes",
            "duration_seconds",
            "resolution_width",
            "resolution_height",
            "frame_rate",
            "format",
            "uploaded_at",
        }
        assert column_names == expected

    def test_user_id_foreign_key(self):
        table = VideoTable.__table__
        user_id_col = table.c.user_id
        fk = list(user_id_col.foreign_keys)[0]
        assert str(fk.column) == "users.id"

    def test_instance_creation(self):
        video = VideoTable(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            file_key="uploads/abc123.mp4",
            file_name="swing.mp4",
            file_size_bytes=50_000_000,
            duration_seconds=10.5,
            resolution_width=1920,
            resolution_height=1080,
            frame_rate=60.0,
            format="mp4",
        )
        assert video.file_size_bytes == 50_000_000
        assert video.resolution_width == 1920


class TestAnalysisTable:
    """Test AnalysisTable model definition."""

    def test_tablename(self):
        assert AnalysisTable.__tablename__ == "analyses"

    def test_columns(self):
        mapper = inspect(AnalysisTable)
        column_names = {col.key for col in mapper.columns}
        expected = {
            "id",
            "user_id",
            "video_id",
            "status",
            "error_message",
            "started_at",
            "completed_at",
            "created_at",
        }
        assert column_names == expected

    def test_foreign_keys(self):
        table = AnalysisTable.__table__
        user_fk = list(table.c.user_id.foreign_keys)[0]
        video_fk = list(table.c.video_id.foreign_keys)[0]
        assert str(user_fk.column) == "users.id"
        assert str(video_fk.column) == "videos.id"

    def test_status_default(self):
        analysis = AnalysisTable(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            video_id=uuid.uuid4(),
        )
        # Default is set at DB level via server_default, but Python default also works
        assert analysis.status is None or analysis.status == "pending"

    def test_nullable_fields(self):
        table = AnalysisTable.__table__
        assert table.c.error_message.nullable is True
        assert table.c.started_at.nullable is True
        assert table.c.completed_at.nullable is True


class TestAnalysisResultTable:
    """Test AnalysisResultTable model definition."""

    def test_tablename(self):
        assert AnalysisResultTable.__tablename__ == "analysis_results"

    def test_columns(self):
        mapper = inspect(AnalysisResultTable)
        column_names = {col.key for col in mapper.columns}
        expected = {
            "id",
            "analysis_id",
            "biomechanics_data",
            "swing_phases_data",
            "evaluations_data",
            "improvements_data",
            "drill_recommendations",
            "overlay_video_key",
            "processing_time_seconds",
            "created_at",
        }
        assert column_names == expected

    def test_analysis_id_unique(self):
        table = AnalysisResultTable.__table__
        assert table.c.analysis_id.unique is True

    def test_analysis_id_foreign_key(self):
        table = AnalysisResultTable.__table__
        fk = list(table.c.analysis_id.foreign_keys)[0]
        assert str(fk.column) == "analyses.id"

    def test_jsonb_columns_not_nullable(self):
        table = AnalysisResultTable.__table__
        assert table.c.biomechanics_data.nullable is False
        assert table.c.swing_phases_data.nullable is False
        assert table.c.evaluations_data.nullable is False
        assert table.c.improvements_data.nullable is False
        assert table.c.drill_recommendations.nullable is False


class TestQualityCheckTable:
    """Test QualityCheckTable model definition."""

    def test_tablename(self):
        assert QualityCheckTable.__tablename__ == "quality_checks"

    def test_columns(self):
        mapper = inspect(QualityCheckTable)
        column_names = {col.key for col in mapper.columns}
        expected = {
            "id",
            "video_id",
            "brightness_status",
            "framing_status",
            "resolution_status",
            "frame_rate_stability_status",
            "details",
            "checked_at",
        }
        assert column_names == expected

    def test_video_id_foreign_key(self):
        table = QualityCheckTable.__table__
        fk = list(table.c.video_id.foreign_keys)[0]
        assert str(fk.column) == "videos.id"

    def test_status_fields_not_nullable(self):
        table = QualityCheckTable.__table__
        assert table.c.brightness_status.nullable is False
        assert table.c.framing_status.nullable is False
        assert table.c.resolution_status.nullable is False
        assert table.c.frame_rate_stability_status.nullable is False


class TestReferenceDataTable:
    """Test ReferenceDataTable model definition."""

    def test_tablename(self):
        assert ReferenceDataTable.__tablename__ == "reference_data"

    def test_columns(self):
        mapper = inspect(ReferenceDataTable)
        column_names = {col.key for col in mapper.columns}
        expected = {
            "id",
            "level",
            "age_group",
            "metric_name",
            "min_value",
            "max_value",
            "optimal_min",
            "optimal_max",
            "source",
            "updated_at",
        }
        assert column_names == expected

    def test_unique_constraint(self):
        table = ReferenceDataTable.__table__
        unique_constraints = [
            c for c in table.constraints if hasattr(c, "name") and c.name == "uq_reference_data_key"
        ]
        assert len(unique_constraints) == 1
        constraint = unique_constraints[0]
        col_names = {col.name for col in constraint.columns}
        assert col_names == {"level", "age_group", "metric_name"}

    def test_instance_creation(self):
        ref = ReferenceDataTable(
            id=uuid.uuid4(),
            level="professional",
            age_group="adult",
            metric_name="bat_speed",
            min_value=110.0,
            max_value=130.0,
            optimal_min=115.0,
            optimal_max=125.0,
            source="Professional baseball average data",
        )
        assert ref.level == "professional"
        assert ref.metric_name == "bat_speed"
        assert ref.min_value == 110.0
        assert ref.optimal_max == 125.0


class TestSeedScript:
    """Test the seed script data generation."""

    def test_get_seed_records_returns_data(self):
        from scripts.seed_reference_data import get_seed_records

        records = get_seed_records()
        assert len(records) > 0

    def test_seed_records_have_all_levels(self):
        from scripts.seed_reference_data import get_seed_records

        records = get_seed_records()
        levels = {r["level"] for r in records}
        assert levels == {"professional", "college", "high_school", "recreational"}

    def test_seed_records_have_all_metrics(self):
        from scripts.seed_reference_data import get_seed_records

        records = get_seed_records()
        metrics = {r["metric_name"] for r in records}
        expected_metrics = {
            "bat_speed",
            "attack_angle",
            "hip_shoulder_separation",
            "hand_path_efficiency",
            "attack_angle",
        }
        assert expected_metrics == metrics

    def test_seed_records_have_required_fields(self):
        from scripts.seed_reference_data import get_seed_records

        records = get_seed_records()
        required_keys = {
            "level",
            "age_group",
            "metric_name",
            "min_value",
            "max_value",
            "optimal_min",
            "optimal_max",
            "source",
        }
        for record in records:
            assert set(record.keys()) == required_keys

    def test_seed_records_optimal_within_range(self):
        """Optimal range should be within the acceptable range."""
        from scripts.seed_reference_data import get_seed_records

        records = get_seed_records()
        for record in records:
            assert record["min_value"] <= record["optimal_min"], (
                f"{record['metric_name']} ({record['level']}): "
                f"optimal_min {record['optimal_min']} < min_value {record['min_value']}"
            )
            assert record["optimal_max"] <= record["max_value"], (
                f"{record['metric_name']} ({record['level']}): "
                f"optimal_max {record['optimal_max']} > max_value {record['max_value']}"
            )
            assert record["optimal_min"] <= record["optimal_max"]

    def test_professional_bat_speed_values(self):
        """Verify professional bat speed reference data matches spec."""
        from scripts.seed_reference_data import get_seed_records

        records = get_seed_records()
        pro_bat_speed = [
            r for r in records
            if r["level"] == "professional" and r["metric_name"] == "bat_speed"
        ]
        assert len(pro_bat_speed) == 1
        data = pro_bat_speed[0]
        assert data["min_value"] == 110.0
        assert data["max_value"] == 130.0
        assert data["optimal_min"] == 115.0
        assert data["optimal_max"] == 125.0
