from datetime import date, datetime, timedelta, timezone

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import select, text

from bike_demand.ingest import forecast, realtime
from bike_demand.serving import load
from bike_demand.serving.models import RealtimeSnapshot, Station, WeatherForecast

KST = timezone(timedelta(hours=9))


def test_split_station_name():
    assert load.split_station_name("102. 망원역 1번출구 앞") == (102, "망원역 1번출구 앞")
    assert load.split_station_name("이름만") == (None, "이름만")
    assert load.split_station_name(None) == (None, None)


def make_warehouse(path, name="강남역"):
    con = duckdb.connect(str(path))
    con.execute(f"""
        create or replace table dim_stations as select * from (values
            ('ST-1', 1, '{name}', '강남구', 37.49, 127.02, 20),
            ('ST-2', 2, '좌표없음', null, null, null, null)
        ) t(station_id, station_no, station_name, district, lat, lon, docks)
    """)
    con.close()


def test_stations_upsert_is_idempotent_and_updates(pg_engine, tmp_path):
    warehouse = tmp_path / "wh.duckdb"
    make_warehouse(warehouse)
    assert load.load_stations_from_warehouse(pg_engine, warehouse) == 2
    make_warehouse(warehouse, name="강남역 2번출구")
    load.load_stations_from_warehouse(pg_engine, warehouse)

    with pg_engine.connect() as conn:
        rows = {s.station_id: s for s in conn.execute(select(Station)).all()}
    assert len(rows) == 2
    assert rows["ST-1"].station_name == "강남역 2번출구"
    assert rows["ST-2"].lat is None and rows["ST-2"].source == "history"


def write_realtime(bronze, day, fetched_at, rows):
    path = bronze / f"date={day:%Y-%m-%d}" / "snapshots.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{"fetched_at": fetched_at, **r} for r in rows]
    pq.write_table(pa.Table.from_pylist(records, schema=realtime.SCHEMA), path)


def test_realtime_load_adds_new_stations_and_skips_duplicates(pg_engine, tmp_path):
    warehouse = tmp_path / "wh.duckdb"
    make_warehouse(warehouse)
    load.load_stations_from_warehouse(pg_engine, warehouse)

    bronze = tmp_path / "bronze"
    day = date(2026, 9, 16)
    row = {"rack_count": "20", "shared": "10", "lat": "37.49", "lon": "127.02"}
    write_realtime(
        bronze,
        day,
        "2026-09-16T20:37:45.505669+09:00",
        [
            {**row, "station_id": "ST-1", "station_name": "1. 강남역 이름 바뀜", "bike_count": "3"},
            {**row, "station_id": "ST-9", "station_name": "9. 새 대여소", "bike_count": "7"},
        ],
    )
    assert load.load_realtime(pg_engine, bronze, day) == (2, 1)
    assert load.load_realtime(pg_engine, bronze, day) == (0, 0)

    with pg_engine.connect() as conn:
        stations = {s.station_id: s for s in conn.execute(select(Station)).all()}
        snapshot = conn.execute(
            select(RealtimeSnapshot).where(RealtimeSnapshot.station_id == "ST-9")
        ).one()
        server_tz = conn.execute(text("show timezone")).scalar_one()
    # 과거 이력의 대여소는 실시간 이름으로 덮어쓰지 않음
    assert stations["ST-1"].station_name == "강남역" and stations["ST-1"].source == "history"
    assert (stations["ST-9"].station_no, stations["ST-9"].station_name) == (9, "새 대여소")
    assert stations["ST-9"].source == "realtime" and stations["ST-9"].docks == 20
    assert snapshot.bike_count == 7
    assert snapshot.fetched_at == datetime(2026, 9, 16, 20, 37, 45, 505669, tzinfo=KST)
    assert snapshot.fetched_at.utcoffset() == timedelta(hours=9)
    assert server_tz == "Asia/Seoul"


def test_forecast_load_is_idempotent(pg_engine, tmp_path):
    path = tmp_path / "vilage_fcst.parquet"
    records = [
        {"base_datetime": "2026-09-16T20:00:00+09:00", "fcst_datetime": "2026-09-16T21:00:00+09:00",
         "category": c, "value": v, "nx": "60", "ny": "127",
         "fetched_at": "2026-09-16T20:37:00+09:00"}
        for c, v in [("TMP", "22"), ("PCP", "강수없음")]
    ]  # fmt: skip
    pq.write_table(pa.Table.from_pylist(records, schema=forecast.SCHEMA), path)
    assert load.load_forecasts(pg_engine, path) == 2
    assert load.load_forecasts(pg_engine, path) == 0
    with pg_engine.connect() as conn:
        pcp = conn.execute(select(WeatherForecast).where(WeatherForecast.category == "PCP")).one()
    assert pcp.value == "강수없음"
    assert pcp.fcst_datetime == datetime(2026, 9, 16, 21, tzinfo=KST)


@pytest.mark.parametrize("revision", ["base", "head"])
def test_migration_round_trip(pg_engine, revision):
    from alembic import command
    from alembic.config import Config

    config = Config("alembic.ini", attributes={"configure_logger": False})
    config.set_main_option(
        "sqlalchemy.url", pg_engine.url.render_as_string(hide_password=False).replace("%", "%%")
    )
    command.downgrade(config, "base")
    with pg_engine.connect() as conn:
        assert conn.execute(text("select to_regclass('stations')")).scalar_one() is None
    if revision == "head":
        command.upgrade(config, "head")
        with pg_engine.connect() as conn:
            assert conn.execute(text("select to_regclass('stations')")).scalar_one() == "stations"
