"""Add indexes for hot query paths.

Revision ID: 002_hot_path_indexes
Revises: 001_initial
Create Date: 2026-06-17 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_hot_path_indexes"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create secondary indexes used by API list/look-up queries."""
    op.create_index("ix_videos_file_key", "videos", ["file_key"])
    op.create_index(
        "ix_analyses_user_id_created_at",
        "analyses",
        ["user_id", "created_at"],
    )
    op.create_index("ix_analyses_status", "analyses", ["status"])


def downgrade() -> None:
    """Drop secondary indexes."""
    op.drop_index("ix_analyses_status", table_name="analyses")
    op.drop_index("ix_analyses_user_id_created_at", table_name="analyses")
    op.drop_index("ix_videos_file_key", table_name="videos")
