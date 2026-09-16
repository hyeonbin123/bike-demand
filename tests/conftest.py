"""서비스 DB 테스트용 임시 PostgreSQL 데이터베이스.

`docker compose up -d db`로 띄운 서버에 테스트마다 임시 DB를 만들고 마이그레이션을 적용한 뒤 지운다.
개발 DB(bike_demand)의 데이터는 건드리지 않는다. 서버에 접속할 수 없으면 해당 테스트는 건너뛴다.
"""

from __future__ import annotations

import uuid

import duckdb
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from bike_demand.serving.db import database_url, make_engine


@pytest.fixture
def pg_engine():
    base = make_url(database_url())
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    name = f"bike_demand_test_{uuid.uuid4().hex[:12]}"
    try:
        with admin.connect() as conn:
            conn.execute(text(f'create database "{name}"'))
    except Exception as exc:  # noqa: BLE001 - 접속 실패 종류와 상관없이 건너뜀
        admin.dispose()
        pytest.skip(f"PostgreSQL에 접속할 수 없음 (docker compose up -d db): {type(exc).__name__}")

    url = base.set(database=name).render_as_string(hide_password=False)
    from alembic import command
    from alembic.config import Config

    engine = None
    try:  # 마이그레이션이 실패해도 임시 DB를 지운다
        config = Config("alembic.ini", attributes={"configure_logger": False})
        config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
        command.upgrade(config, "head")
        engine = make_engine(url)
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'drop database if exists "{name}" with (force)'))
        admin.dispose()


@pytest.fixture
def small_warehouse():
    """작은 warehouse: 대여소 ST-1은 학습·평가 기간 모두, ST-2는 평가 기간에만 있다."""
    db = duckdb.connect()
    db.execute("""
        create table dim_stations as
        select * from (values ('ST-1', '강남구', 10, 37.5, 127.0),
                              ('ST-2', '마포구', 12, 37.6, 126.9))
            t(station_id, district, docks, lat, lon)
    """)
    db.execute("""
        create table dim_hours as
        select h as hour_start, hour(h) as hour_of_day, isodow(h) as day_of_week,
               month(h) as month, dayofyear(h) as day_of_year,
               false as is_holiday, isodow(h) >= 6 as is_offday,
               10.0 as temp_c, 0.0 as rain_mm, 1.0 as wind_ms, 50.0 as humidity_pct,
               0.0 as snow_cm
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


@pytest.fixture
def two_year_warehouse():
    """대여소 하나가 반기마다 시간당 1, 2, 3, 4, 5대를 빌리는 2023-01 ~ 2025-06 격자."""
    db = duckdb.connect()
    db.execute("""
        create table dim_stations as
        select 'ST-1' as station_id, '강남구' as district, 10 as docks, 37.5 as lat, 127.0 as lon
    """)
    db.execute("""
        create table dim_hours as
        select h as hour_start, hour(h) as hour_of_day, isodow(h) as day_of_week,
               month(h) as month, dayofyear(h) as day_of_year,
               false as is_holiday, isodow(h) >= 6 as is_offday,
               10.0 as temp_c, 0.0 as rain_mm, 1.0 as wind_ms, 50.0 as humidity_pct,
               0.0 as snow_cm
        from unnest(generate_series(timestamp '2023-01-01', timestamp '2025-06-30 23:00:00',
                                    interval 1 hour)) t(h)
    """)
    db.execute("""
        create table int_station_hour_grid as
        select 'ST-1' as station_id, hour_start,
               ((year(hour_start) - 2023) * 2 + (month(hour_start) > 6)::int + 1) as rentals
        from dim_hours
    """)
    return db
