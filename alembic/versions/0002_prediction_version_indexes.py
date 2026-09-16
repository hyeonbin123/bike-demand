"""index predictions for picking the newest version and filtering by version and hour

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-17

API는 매 요청마다 created_at이 가장 늦은 버전을 찾고(docs/api.md), 그 버전의 시간 범위를 읽는다.
예측은 발표마다 약 13만 행씩 쌓이므로 인덱스 없이 훑으면 요청이 점점 느려진다.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_predictions_created_at_version", "predictions", ["created_at", "model_version"]
    )
    op.create_index("ix_predictions_version_hour", "predictions", ["model_version", "hour_start"])


def downgrade() -> None:
    op.drop_index("ix_predictions_version_hour", table_name="predictions")
    op.drop_index("ix_predictions_created_at_version", table_name="predictions")
