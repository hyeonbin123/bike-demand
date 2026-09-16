"""create serving tables

Revision ID: 0001
Revises:
Create Date: 2026-09-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "stations",
        sa.Column("station_id", sa.String(length=20), primary_key=True),
        sa.Column("station_no", sa.Integer()),
        sa.Column("station_name", sa.String(length=200)),
        sa.Column("district", sa.String(length=50)),
        sa.Column("lat", sa.Double()),
        sa.Column("lon", sa.Double()),
        sa.Column("docks", sa.Integer()),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_table(
        "realtime_snapshots",
        sa.Column("station_id", sa.String(length=20), primary_key=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("bike_count", sa.Integer(), nullable=False),
        sa.Column("rack_count", sa.Integer()),
    )
    op.create_index("ix_realtime_snapshots_fetched_at", "realtime_snapshots", ["fetched_at"])
    op.create_table(
        "weather_forecasts",
        sa.Column("base_datetime", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("fcst_datetime", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("category", sa.String(length=10), primary_key=True),
        sa.Column("nx", sa.Integer(), primary_key=True),
        sa.Column("ny", sa.Integer(), primary_key=True),
        sa.Column("value", sa.String(length=50), nullable=False),
    )
    op.create_table(
        "predictions",
        sa.Column("station_id", sa.String(length=20), primary_key=True),
        sa.Column("hour_start", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("model_version", sa.String(length=100), primary_key=True),
        sa.Column("predicted_rentals", sa.Double(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_predictions_hour_start", "predictions", ["hour_start"])


def downgrade() -> None:
    op.drop_index("ix_predictions_hour_start", table_name="predictions")
    op.drop_table("predictions")
    op.drop_table("weather_forecasts")
    op.drop_index("ix_realtime_snapshots_fetched_at", table_name="realtime_snapshots")
    op.drop_table("realtime_snapshots")
    op.drop_table("stations")
