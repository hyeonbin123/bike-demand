"""index predictions for picking the newest version and filtering by version and hour

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-17

API는 매 요청마다 created_at이 가장 늦은 버전을 찾고(docs/api.md), 그 버전의 시간 범위를 읽는다.
예측은 발표마다 약 13만 행씩 쌓이므로 인덱스 없이 훑으면 요청이 점점 느려진다.
"""

# [파일 역할] predictions 테이블에 '가장 최근 버전 찾기'와 '버전 + 시간 범위 읽기'용 복합 인덱스 두
# 개를 더하는 마이그레이션.
# [근거] 커밋 b456909 메시지: 개발 DB에서 가장 최근 모델 버전을 찾는 조회가 테이블 전체를 훑어 165ms
# 걸렸고, 두 인덱스를 둔 뒤 0.09ms가 됐다. 같은 인덱스가 serving/models.py의 __table_args__에도 적혀
# 있어 ORM 정의와 DB가 같다.

from collections.abc import Sequence

from alembic import op

# 0001 다음에 적용되는 마이그레이션임을 Alembic에 알린다.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # (created_at, model_version): API의 'created_at이 가장 늦은 행의 model_version, 같으면 버전
    # 문자열이 큰 것' 규칙을 정렬된 인덱스의 끝에서 바로 읽게 한다.
    op.create_index(
        "ix_predictions_created_at_version", "predictions", ["created_at", "model_version"]
    )
    # (model_version, hour_start): 고른 버전 하나의 특정 시간 범위(앞으로 N시간, 하루 0~23시)만 읽는
    # 조회용. 버전이 앞에 있어 다른 버전 행을 건너뛴다.
    op.create_index("ix_predictions_version_hour", "predictions", ["model_version", "hour_start"])


def downgrade() -> None:
    # 되돌릴 때는 만든 두 인덱스만 지운다(데이터는 그대로).
    op.drop_index("ix_predictions_version_hour", table_name="predictions")
    op.drop_index("ix_predictions_created_at_version", table_name="predictions")
