"""서비스용 테이블. 설명은 docs/serving-schema.md."""

# [파일 역할] 서비스 DB 테이블 4개(stations, realtime_snapshots, weather_forecasts, predictions)를
# SQLAlchemy ORM 클래스로 정의한다.
# 적재(serving/load.py)·예측(model/predict.py)·API(api/main.py)·측정 코드가 이 클래스로 쿼리를
# 만들고, alembic/env.py가 Base.metadata를 마이그레이션 비교 기준으로 쓴다.
# [설계] 원본과 정제 자료(1억 건대)는 DuckDB·Parquet에 두고, 여기에는 API가 바로 읽을 것만
# 둔다(README 설계 판단). 실제 테이블은 alembic/versions의 마이그레이션이 만들고, 이 정의와 같게
# 유지한다.
# [시각] 모든 시각 열은 timezone=True(timestamptz)다. 연결의 세션 시간대가 Asia/Seoul이라 +09:00으로
# 읽힌다(serving/db.py).

from __future__ import annotations

from datetime import datetime

# SQLAlchemy 2.0의 선언형 매핑(DeclarativeBase, Mapped, mapped_column)으로 열 타입을 파이썬 타입
# 힌트와 함께 적는다.
from sqlalchemy import DateTime, Double, Index, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# 모든 서비스 테이블이 공유하는 메타데이터의 뿌리.
class Base(DeclarativeBase):
    pass


# 대여소 한 곳 = 한 행. 과거 이력(warehouse의 dim_stations)에서 온 대여소는 source=history, 실시간
# 스냅샷에만 나온 새 대여소는 source=realtime으로 들어온다. 실시간에만 있는 곳은 자치구가 없을 수
# 있어 대부분 열이 빈 값을 허용한다.
# station_id는 'ST-xxx' 형식의 대여소ID로, 스냅샷·예측 테이블과 잇는 키다(외래 키 제약은 두지 않음).
class Station(Base):
    __tablename__ = "stations"

    station_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    station_no: Mapped[int | None] = mapped_column(Integer)
    station_name: Mapped[str | None] = mapped_column(String(200))
    district: Mapped[str | None] = mapped_column(String(50))
    lat: Mapped[float | None] = mapped_column(Double)
    lon: Mapped[float | None] = mapped_column(Double)
    docks: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(20))  # history | realtime
    # 넣을 때 DB 시각(now())이 기본값으로 들어가고, SQLAlchemy로 UPDATE할 때도 갱신된다. 적재의
    # upsert는 set_에 직접 now()를 넣는다.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# 실시간 대여정보 한 번 수집(10분마다)의 대여소별 자전거 수. (station_id, fetched_at)이 기본키라
# 같은 스냅샷을 다시 적재해도 행이 늘지 않는다.
# fetched_at 인덱스: 기본키는 station_id가 앞이라, 시각 범위로 스냅샷을 찾는 조회(최근 스냅샷, 기준
# 시각 이전 스냅샷)를 위해 시각만의 인덱스를 따로 둔다.
class RealtimeSnapshot(Base):
    __tablename__ = "realtime_snapshots"
    __table_args__ = (Index("ix_realtime_snapshots_fetched_at", "fetched_at"),)

    station_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    # bike_count는 빈 값을 허용하지 않는다(적재가 값 없는 행을 건너뜀). 거치대 밖 주차가 있어
    # rack_count보다 클 수 있다(docs/serving-schema.md).
    bike_count: Mapped[int] = mapped_column(Integer)
    rack_count: Mapped[int | None] = mapped_column(Integer)


# 기상청 단기예보를 받은 긴 형식(한 행 = 발표 시각 × 예보 시각 × 항목 × 격자) 그대로 둔다. 다섯 열
# 전체가 기본키라 같은 발표를 다시 받아도 중복되지 않는다.
# value를 문자열로 두는 이유: 강수량 같은 항목에 '강수없음'처럼 숫자가 아닌 값이 오기 때문이다. 숫자
# 변환은 읽는 쪽(예측의 예보 날씨 변환)이 한다.
class WeatherForecast(Base):
    __tablename__ = "weather_forecasts"

    base_datetime: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    fcst_datetime: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    category: Mapped[str] = mapped_column(String(10), primary_key=True)
    nx: Mapped[int] = mapped_column(Integer, primary_key=True)
    ny: Mapped[int] = mapped_column(Integer, primary_key=True)
    value: Mapped[str] = mapped_column(String(50))


# 대여소 × 예측 시간(정시 시작 1시간) × 모델 버전마다 예상 대여 수 한 개. model_version은 '서비스
# 모델 세대 이름 + -fcst + 예보 발표 시각'이라 같은 시간을 여러 발표로 예측한 값이 함께 쌓인다.
# API는 한 응답에 한 버전만 쓴다(README 설계 판단).
# 인덱스 세 개: hour_start(시간 범위 조회), (created_at, model_version)(가장 최근에 만든 버전 찾기),
# (model_version, hour_start)(그 버전의 시간 범위 읽기). 예측이 발표마다 약 13만 행씩 쌓여 인덱스
# 없이 훑으면 요청이 점점 느려지므로 마이그레이션 0002에서 뒤의 둘을 더했다.
class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_hour_start", "hour_start"),
        Index("ix_predictions_created_at_version", "created_at", "model_version"),
        Index("ix_predictions_version_hour", "model_version", "hour_start"),
    )

    station_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    hour_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(100), primary_key=True)
    predicted_rentals: Mapped[float] = mapped_column(Double)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
