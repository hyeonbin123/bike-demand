"""create serving tables

Revision ID: 0001
Revises:
Create Date: 2026-09-16
"""

# [파일 역할] 서비스용 PostgreSQL의 첫 마이그레이션. API가 바로 읽을 네 테이블(stations,
# realtime_snapshots, weather_forecasts, predictions)을 만든다.
# [설계 이유] 원본·정제 데이터(1억 4천만 건 대여이력)는 Parquet·DuckDB에 두고, Postgres에는 서비스에
# 필요한 것만 둔다(README 설계 판단). 컬럼 뜻과 적재 규칙은 docs/serving-schema.md에 있다.
# [멱등 적재와의 관계] 스냅샷·예보·예측 테이블은 '같은 행'을 정의하는 복합 기본 키를 둔다. 적재
# 코드는 이 키로 충돌을 처리해(스냅샷·예보는 건너뛰기, 예측은 값과 생성 시각 갱신, 대여소는 upsert)
# 같은 입력을 다시 넣어도 행이 늘지 않게 한다.

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# Alembic이 마이그레이션 순서를 잇는 식별자. down_revision이 None이라 첫 번째 마이그레이션이다.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# 네 테이블과 조회용 인덱스를 만든다. 시각 컬럼은 모두 timestamptz(timezone=True)라 저장은 절대
# 시각으로 하고, 애플리케이션 연결이 세션 시간대를 Asia/Seoul로 열어 +09:00으로
# 읽는다(docs/serving-schema.md 시간대).
def upgrade() -> None:
    # stations: 대여소 한 곳이 한 행. source는 'history'(dbt dim_stations에서 옴) 또는
    # 'realtime'(실시간 스냅샷에만 있는 새 대여소)을 구분한다. 실시간에만 있는 곳은 좌표·자치구가
    # 없을 수 있어 lat·lon·district는 비워 둘 수 있게 했다.
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
    # realtime_snapshots: (대여소, 수집 시각) 한 쌍이 한 행. fetched_at은 수집을 시작한 시각이라 한
    # 번 수집한 모든 대여소가 같은 값을 가진다. rack_count는 비워 둘 수 있게, bike_count는 반드시
    # 있게 했다.
    op.create_table(
        "realtime_snapshots",
        sa.Column("station_id", sa.String(length=20), primary_key=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("bike_count", sa.Integer(), nullable=False),
        sa.Column("rack_count", sa.Integer()),
    )
    # 가장 최근 스냅샷 시각(/health)이나 '최근 N분 안의 스냅샷'(/shortage-risk) 조회가 시각으로
    # 거르므로 fetched_at에 인덱스를 둔다.
    op.create_index("ix_realtime_snapshots_fetched_at", "realtime_snapshots", ["fetched_at"])
    # weather_forecasts: 기상청 단기예보를 API가 주는 긴 형식 그대로(발표 시각, 예보 시각, 항목,
    # 격자 x·y, 값) 저장한다. value가 문자열인 것은 '강수없음' 같은 글자 값이 있어서다. 한 발표의
    # 값은 바뀌지 않으므로 다섯 컬럼 키가 겹치면 건너뛴다.
    op.create_table(
        "weather_forecasts",
        sa.Column("base_datetime", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("fcst_datetime", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("category", sa.String(length=10), primary_key=True),
        sa.Column("nx", sa.Integer(), primary_key=True),
        sa.Column("ny", sa.Integer(), primary_key=True),
        sa.Column("value", sa.String(length=50), nullable=False),
    )
    # predictions: (대여소, 시간, 모델 버전) 한 쌍이 한 행. model_version을 기본 키에 넣어 같은
    # 시간을 여러 예측 묶음이 함께 가질 수 있게 했다(예보 발표마다 새 묶음). created_at은 DB가
    # now()로 채우고, API는 이 값이 가장 늦은 묶음을 고른다(docs/api.md).
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
    # 시간 범위(앞으로 몇 시간, 하루)로 예측을 읽는 조회를 위한 인덱스. 버전을 함께 거르는 인덱스는
    # 0002에서 더했다.
    op.create_index("ix_predictions_hour_start", "predictions", ["hour_start"])


# 만든 순서의 역순으로 인덱스와 테이블을 지운다(인덱스를 먼저 지우고 그 테이블을 지움).
def downgrade() -> None:
    op.drop_index("ix_predictions_hour_start", table_name="predictions")
    op.drop_table("predictions")
    op.drop_table("weather_forecasts")
    op.drop_index("ix_realtime_snapshots_fetched_at", table_name="realtime_snapshots")
    op.drop_table("realtime_snapshots")
    op.drop_table("stations")
