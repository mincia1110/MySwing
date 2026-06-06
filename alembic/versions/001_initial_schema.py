"""Initial database schema.

Revision ID: 001_initial
Revises:
Create Date: 2024-01-01 00:00:00.000000

Creates all tables for the Baseball Swing Analysis service:
- users: User accounts
- user_profiles: Physical and batting characteristics
- videos: Uploaded video metadata
- analyses: Analysis job tracking
- analysis_results: Complete analysis output (JSONB)
- quality_checks: Video quality validation results
- reference_data: Professional player benchmark data
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables."""
    # Users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )

    # User profiles table
    op.create_table(
        "user_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("bat_length", sa.Float(), nullable=False),
        sa.Column("batting_direction", sa.String(10), nullable=False),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.Column("camera_direction", sa.String(10), nullable=True),
        sa.Column("age_group", sa.String(50), nullable=True),
        sa.Column("level", sa.String(20), nullable=True),
        sa.Column("bat_weight", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id"),
    )

    # Videos table
    op.create_table(
        "videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("file_key", sa.String(500), nullable=False),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("resolution_width", sa.Integer(), nullable=False),
        sa.Column("resolution_height", sa.Integer(), nullable=False),
        sa.Column("frame_rate", sa.Float(), nullable=False),
        sa.Column("format", sa.String(10), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )

    # Analyses table
    op.create_table(
        "analyses",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
    )

    # Analysis results table
    op.create_table(
        "analysis_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("biomechanics_data", postgresql.JSONB(), nullable=False),
        sa.Column("swing_phases_data", postgresql.JSONB(), nullable=False),
        sa.Column("evaluations_data", postgresql.JSONB(), nullable=False),
        sa.Column("improvements_data", postgresql.JSONB(), nullable=False),
        sa.Column("drill_recommendations", postgresql.JSONB(), nullable=False),
        sa.Column("overlay_video_key", sa.String(500), nullable=True),
        sa.Column("processing_time_seconds", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("analysis_id"),
    )

    # Quality checks table
    op.create_table(
        "quality_checks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("brightness_status", sa.String(10), nullable=False),
        sa.Column("framing_status", sa.String(10), nullable=False),
        sa.Column("resolution_status", sa.String(10), nullable=False),
        sa.Column("frame_rate_stability_status", sa.String(10), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("checked_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
    )

    # Reference data table
    op.create_table(
        "reference_data",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("level", sa.String(20), nullable=False),
        sa.Column("age_group", sa.String(50), nullable=False, server_default="adult"),
        sa.Column("metric_name", sa.String(100), nullable=False),
        sa.Column("min_value", sa.Float(), nullable=False),
        sa.Column("max_value", sa.Float(), nullable=False),
        sa.Column("optimal_min", sa.Float(), nullable=False),
        sa.Column("optimal_max", sa.Float(), nullable=False),
        sa.Column("source", sa.String(255), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("level", "age_group", "metric_name", name="uq_reference_data_key"),
    )


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.drop_table("reference_data")
    op.drop_table("quality_checks")
    op.drop_table("analysis_results")
    op.drop_table("analyses")
    op.drop_table("videos")
    op.drop_table("user_profiles")
    op.drop_table("users")
