"""Make video object keys unique for atomic metadata upserts.

Revision ID: 003_unique_video_file_key
Revises: 002_hot_path_indexes
Create Date: 2026-07-28 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003_unique_video_file_key"
down_revision: Union[str, None] = "002_hot_path_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Deduplicate same-owner rows, then enforce global object-key identity."""
    # Never auto-merge records across tenants. An operator must investigate and
    # resolve any legacy ownership conflict before this migration can continue.
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM videos
                    GROUP BY file_key
                    HAVING COUNT(DISTINCT user_id) > 1
                ) THEN
                    RAISE EXCEPTION
                        'videos.file_key has cross-user duplicates; resolve ownership first';
                END IF;
            END
            $$
            """
        )
    )

    # Keep the oldest row for each same-owner key and preserve all foreign-key
    # references before deleting duplicates.
    ranked_videos = """
        SELECT
            id,
            FIRST_VALUE(id) OVER (
                PARTITION BY file_key
                ORDER BY uploaded_at NULLS LAST, id
            ) AS keeper_id,
            ROW_NUMBER() OVER (
                PARTITION BY file_key
                ORDER BY uploaded_at NULLS LAST, id
            ) AS duplicate_rank
        FROM videos
    """
    op.execute(
        sa.text(
            f"""
            WITH ranked AS ({ranked_videos})
            UPDATE analyses AS target
            SET video_id = ranked.keeper_id
            FROM ranked
            WHERE target.video_id = ranked.id
              AND ranked.duplicate_rank > 1
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            WITH ranked AS ({ranked_videos})
            UPDATE quality_checks AS target
            SET video_id = ranked.keeper_id
            FROM ranked
            WHERE target.video_id = ranked.id
              AND ranked.duplicate_rank > 1
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            WITH ranked AS ({ranked_videos})
            DELETE FROM videos AS target
            USING ranked
            WHERE target.id = ranked.id
              AND ranked.duplicate_rank > 1
            """
        )
    )

    op.drop_index("ix_videos_file_key", table_name="videos")
    op.create_unique_constraint("uq_videos_file_key", "videos", ["file_key"])


def downgrade() -> None:
    """Restore the legacy non-unique lookup index."""
    op.drop_constraint("uq_videos_file_key", "videos", type_="unique")
    op.create_index("ix_videos_file_key", "videos", ["file_key"])
