"""서비스 DB 테스트용 임시 PostgreSQL 데이터베이스.

`docker compose up -d db`로 띄운 서버에 테스트마다 임시 DB를 만들고 마이그레이션을 적용한 뒤 지운다.
개발 DB(bike_demand)의 데이터는 건드리지 않는다. 서버에 접속할 수 없으면 해당 테스트는 건너뛴다.
"""

# [파일 역할] pytest가 모든 테스트 파일에 자동으로 주는 공용 fixture 모음. 두 종류다.
# - 서비스 DB 테스트용: pg_engine(테스트마다 새로 만들고 지우는 임시 PostgreSQL DB).
#   적재·API·예측·실시간 수집 테스트가 쓴다.
# - 모델 테스트용: small_warehouse·two_year_warehouse(메모리 DuckDB에 dbt 결과 테이블을 흉내 낸 작은
#   warehouse). 실제 1억 건대 warehouse 없이 특징·수준 계산 SQL을 검증한다.
# [설계 이유] 개발 DB의 수집 자료를 건드리지 않으려고 같은 서버에 이름이 무작위인 DB를 따로 만들고,
# 실제 alembic 마이그레이션을 적용해 운영과 같은 스키마로 잰다. DB가 꺼져 있으면 실패 대신 건너뛰어,
# DB 없는 환경(새 클론 등)에서도 나머지 테스트는 돈다.

from __future__ import annotations

import uuid
from contextlib import contextmanager

# duckdb: 메모리 warehouse fixture. sqlalchemy: 임시 DB를 만들고 지우는 관리용 연결. make_url: 접속
# 문자열에서 DB 이름만 바꾸기.
import duckdb
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

# 앱과 같은 접속 주소·엔진 설정(세션 시간대 KST, 5초 연결 제한)을 써야 테스트가 운영과 같은 조건이
# 된다.
from bike_demand.serving.db import database_url, make_engine


# 임시 DB를 만들 수 없을 때(서버 꺼짐, 접속 거부 등) 내는 예외. pg_engine이 이 예외만 잡아
# 건너뛰기로 바꾼다. 마이그레이션 실패 같은 다른 오류는 그대로 테스트 실패로 드러난다.
class DatabaseUnavailable(Exception):
    pass


# 임시 DB 하나의 수명: 만들기 → alembic upgrade head → 앱과 같은 엔진으로 내주기 → (성공·실패와
# 상관없이) 지우기.
# 호출 위치: pg_engine fixture, tests/test_serving_load.py와
# tests/test_review_regressions.py(마이그레이션·엔진 생성이 실패해도 DB가 지워지는지 확인).
@contextmanager
def temporary_database(upgrade=None):
    """같은 서버에 임시 DB를 만들고 마이그레이션한 엔진을 준다. 어느 단계에서 실패해도 지운다.

    upgrade: 테스트에서 마이그레이션 실패를 흉내 낼 때만 바꿔 넣는다.
    """
    from alembic import command
    from alembic.config import Config

    base = make_url(database_url())
    # CREATE/DROP DATABASE는 트랜잭션 안에서 실행할 수 없으므로 관리용 연결은 기본 DB(postgres)에
    # AUTOCOMMIT으로 연다.
    admin = create_engine(
        base.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        # DB가 꺼져 있으면 Windows의 psycopg가 거부된 연결을 알아채지 못하고 멈춘다.
        # make_engine과 같은 5초 제한을 둔다
        connect_args={"connect_timeout": 5},
    )
    # 테스트마다 무작위 이름이라 테스트끼리, 또 개발 DB(bike_demand)와 섞이지 않는다.
    name = f"bike_demand_test_{uuid.uuid4().hex[:12]}"
    try:
        with admin.connect() as conn:
            conn.execute(text(f'create database "{name}"'))
    except Exception as exc:  # noqa: BLE001 - 접속 실패 종류와 상관없이 건너뜀
        admin.dispose()
        raise DatabaseUnavailable(type(exc).__name__) from None

    # alembic Config는 configparser 값이라 '%'를 보간 문자로 읽는다. 비밀번호 등에 '%'가 있어도
    # 깨지지 않게 '%%'로 바꿔 넣는다. configure_logger=False는 alembic이 테스트 로깅 설정을 덮어쓰지
    # 않게 한다.
    url = base.set(database=name).render_as_string(hide_password=False)
    engine = None
    try:  # 생성 직후부터 정리를 보장한다(마이그레이션·엔진 생성 실패 포함)
        config = Config("alembic.ini", attributes={"configure_logger": False})
        config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
        (upgrade or command.upgrade)(config, "head")
        engine = make_engine(url)
        yield engine
    finally:
        # 엔진 연결을 먼저 정리한 뒤 DROP한다. WITH (FORCE)는 남은 연결이 있어도 끊고 지워, 실패한
        # 테스트가 임시 DB를 남기지 않게 한다.
        if engine is not None:
            engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'drop database if exists "{name}" with (force)'))
        admin.dispose()


# 한 번 접속에 실패하면 이번 실행의 나머지 DB 테스트는 제한 시간을 다시 기다리지 않고 바로 건너뛴다
_db_unavailable: str | None = None


# DB 테스트가 받는 fixture. 테스트마다 빈 임시 DB(스키마만 있음)를 준다. 서버에 접속할 수 없으면
# 이유(예외 이름)를 기억해 두고, 이번 실행의 나머지 DB 테스트는 다시 5초씩 기다리지 않고 바로
# 건너뛴다(DB가 꺼진 상태에서 테스트가 오래 멈추던 문제를 고친 것, test_review_regressions.py가
# 검증).
@pytest.fixture
def pg_engine():
    global _db_unavailable
    if _db_unavailable is not None:
        pytest.skip(f"PostgreSQL에 접속할 수 없음 (docker compose up -d db): {_db_unavailable}")
    try:
        with temporary_database() as engine:
            yield engine
    except DatabaseUnavailable as exc:
        _db_unavailable = str(exc)
        pytest.skip(f"PostgreSQL에 접속할 수 없음 (docker compose up -d db): {exc}")


# 특징 SQL을 검증하는 작은 warehouse(2024-01 한 달). dbt의
# dim_stations·dim_hours·int_station_hour_grid와 같은 열 이름만 갖춘다.
# 학습 기간과 평가 기간의 대여 수를 크게 다르게(8시 2대 → 100대) 만들어, 평가 기간 값이 대여소 패턴
# 특징에 새면 바로 드러나게 한다. ST-2는 평가 기간에만 있어 '학습에 없던 새 대여소' 처리를 확인하는
# 데 쓴다.
# 호출 위치: test_model_frames.py, test_model_predict.py, test_model_forecast_gap.py.
@pytest.fixture
def small_warehouse():
    """작은 warehouse: 대여소 ST-1은 학습·평가 기간 모두, ST-2는 평가 기간에만 있다."""
    db = duckdb.connect()
    db.execute("""
        create table dim_stations as
        select * from (values ('ST-1', 1, '강남구', 10, 37.5, 127.0),
                              ('ST-2', 2, '마포구', 12, 37.6, 126.9))
            t(station_id, station_no, district, docks, lat, lon)
    """)
    db.execute("""
        create table dim_hours as
        select h as hour_start, hour(h) as hour_of_day, isodow(h) as day_of_week,
               month(h) as month, dayofyear(h) as day_of_year,
               false as is_holiday, isodow(h) >= 6 as is_offday,
               10.0 as temp_c, 0.0 as rain_mm, 1.0 as wind_ms, 50.0 as humidity_pct,
               false as is_new_snow
        from unnest(generate_series(timestamp '2024-01-01', timestamp '2024-01-31 23:00:00',
                                    interval 1 hour)) t(h)
    """)
    # 학습 기간(1/1~1/15)에는 ST-1이 8시마다 2대, 평가 기간(1/16~)에는 8시마다 100대
    db.execute("""
        create table int_station_hour_grid as
        select 'ST-1' as station_id, hour_start,
               case when hour(hour_start) = 8 and not is_offday
                    then (case when hour_start < '2024-01-16' then 2 else 100 end) else 0 end
                   as rentals
        from dim_hours
        union all
        select 'ST-2', hour_start, case when hour(hour_start) = 8 then 5 else 0 end
        from dim_hours where hour_start >= '2024-01-16'
    """)
    return db


# v2 수준 특징(공개 반기 평균·비율)을 검증하는 warehouse. 2023-01 ~ 2025-06 격자에서 대여소 하나가
# 반기마다 시간당 1·2·3·4·5대를 빌리므로, 어느 반기를 '공개됨'으로 골랐는지가 특징 값으로 바로
# 보인다.
# 호출 위치: test_model_levels.py, test_model_predict.py, test_review_regressions.py.
@pytest.fixture
def two_year_warehouse():
    """대여소 하나가 반기마다 시간당 1, 2, 3, 4, 5대를 빌리는 2023-01 ~ 2025-06 격자."""
    db = duckdb.connect()
    db.execute("""
        create table dim_stations as
        select 'ST-1' as station_id, 1 as station_no, '강남구' as district, 10 as docks,
               37.5 as lat, 127.0 as lon
    """)
    db.execute("""
        create table dim_hours as
        select h as hour_start, hour(h) as hour_of_day, isodow(h) as day_of_week,
               month(h) as month, dayofyear(h) as day_of_year,
               false as is_holiday, isodow(h) >= 6 as is_offday,
               10.0 as temp_c, 0.0 as rain_mm, 1.0 as wind_ms, 50.0 as humidity_pct,
               false as is_new_snow
        from unnest(generate_series(timestamp '2023-01-01', timestamp '2025-06-30 23:00:00',
                                    interval 1 hour)) t(h)
    """)
    # 반기 번호: (연도 − 2023) × 2 + (7월 이후면 1) + 1 → 2023 상반기 1, 하반기 2, …, 2025 상반기 5.
    db.execute("""
        create table int_station_hour_grid as
        select 'ST-1' as station_id, hour_start,
               ((year(hour_start) - 2023) * 2 + (month(hour_start) > 6)::int + 1) as rentals
        from dim_hours
    """)
    return db
