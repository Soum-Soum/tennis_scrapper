"""rename match cols

Revision ID: 8f626612d8fa
Revises: 116d527390b1
Create Date: 2025-08-27 11:32:02.470676

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "8f626612d8fa"
down_revision: Union[str, Sequence[str], None] = "116d527390b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Rename columns instead of drop/add to preserve data
    op.alter_column("match", "atp_ranking_player_1", new_column_name="player_1_ranking")
    op.alter_column(
        "match", "atp_points_player_1", new_column_name="player_1_ranking_points"
    )
    op.alter_column("match", "atp_ranking_player_2", new_column_name="player_2_ranking")
    op.alter_column(
        "match", "atp_points_player_2", new_column_name="player_2_ranking_points"
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Revert renames
    op.alter_column("match", "player_1_ranking", new_column_name="atp_ranking_player_1")
    op.alter_column(
        "match", "player_1_ranking_points", new_column_name="atp_points_player_1"
    )
    op.alter_column("match", "player_2_ranking", new_column_name="atp_ranking_player_2")
    op.alter_column(
        "match", "player_2_ranking_points", new_column_name="atp_points_player_2"
    )
