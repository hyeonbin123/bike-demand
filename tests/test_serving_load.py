# [무엇을 검증하나] 서비스 DB 적재(src/bike_demand/serving/load.py)가 멱등인지와 각 테이블의 규칙을
# 실제 PostgreSQL(임시 DB)에서 확인한다.
# - 대여소: warehouse에서 upsert(다시 넣으면 고친 값으로 바뀌고 행은 늘지 않음), 실시간에만 나온
#   대여소는 추가만 하고 과거 이력 대여소는 덮어쓰지 않음, '102. 이름' 꼴 이름 나누기
# - 스냅샷·예보: 같은 키는 건너뛰어 두 번째 적재는 0행, 시각은 +09:00(KST)으로 저장·조회
# - 마이그레이션: base까지 내렸다가 head로 올리기, 마이그레이션이 실패해도 임시 DB가 지워짐
# [왜 중요한가] 수집 작업은 10분마다 그날 Parquet 전체를 다시 적재하므로 멱등이 깨지면 스냅샷이
# 곧바로 중복된다(docs/serving-schema.md 적재 규칙).

from datetime import date, datetime, timedelta, timezone

# duckdb: 작은 warehouse 파일(dim_stations만) 만들기. pyarrow: ingest 모듈의 SCHEMA(모든 열
# 문자열)로 bronze Parquet 만들기. sqlalchemy: 적재 결과 조회.
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import select, text

from bike_demand.ingest import forecast, realtime
from bike_demand.serving import load
from bike_demand.serving.models import RealtimeSnapshot, Station, WeatherForecast

KST = timezone(timedelta(hours=9))


# 실시간 API 이름 '102. 망원역 1번출구 앞'은 번호 102와 이름으로 나누고, 번호가 없으면 이름만, 값이
# 없으면 둘 다 None.
def test_split_station_name():
    assert load.split_station_name("102. 망원역 1번출구 앞") == (102, "망원역 1번출구 앞")
    assert load.split_station_name("이름만") == (None, "이름만")
    assert load.split_station_name(None) == (None, None)


# dim_stations만 있는 warehouse 파일. ST-1은 이름을 바꿔 다시 만들 수 있고, ST-2는
# 좌표·자치구·거치대 수가 없는 대여소다.
def make_warehouse(path, name="강남역"):
    con = duckdb.connect(str(path))
    con.execute(f"""
        create or replace table dim_stations as select * from (values
            ('ST-1', 1, '{name}', '강남구', 37.49, 127.02, 20),
            ('ST-2', 2, '좌표없음', null, null, null, null)
        ) t(station_id, station_no, station_name, district, lat, lon, docks)
    """)
    con.close()


# 같은 warehouse를 두 번 적재해도 2행이고, 두 번째에 바뀐 이름이 반영된다(upsert). 좌표가 없는
# 대여소도 null로 들어가고 출처는 history.
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


# 실시간 수집이 만드는 날짜별 bronze Parquet(date=YYYY-MM-DD/snapshots.parquet)을 직접 쓴다.
def write_realtime(bronze, day, fetched_at, rows):
    path = bronze / f"date={day:%Y-%m-%d}" / "snapshots.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{"fetched_at": fetched_at, **r} for r in rows]
    pq.write_table(pa.Table.from_pylist(records, schema=realtime.SCHEMA), path)


# ST-1(과거 이력 대여소)과 ST-9(실시간에만 있는 새 대여소)의 스냅샷을 적재하면 스냅샷 2행·새 대여소
# 1곳, 같은 파일을 다시 적재하면 (0, 0).
# ST-1의 이름은 실시간 이름으로 덮어쓰지 않고(history 유지), ST-9는 이름 앞 번호 9와 거치대 수
# 20으로 source=realtime 추가된다. 마이크로초까지 있는 fetched_at이 그대로 KST(+09:00) 시각으로
# 읽히고, 세션 시간대가 Asia/Seoul인지도 확인한다(serving/db.py의 연결 설정).
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


# 단기예보 두 항목을 두 번 적재하면 처음 2행, 다음 0행. '강수없음' 같은 문자열 값은 그대로 저장되고
# 예보 시각은 KST 시각이다.
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


# alembic downgrade base로 모든 표가 지워지고(stations 없음), 다시 upgrade head로 만들어지는지 본다.
# 마이그레이션의 downgrade 경로도 깨지지 않았는지 확인하는 것이다. 접속 문자열의 '%'는 configparser
# 보간을 피하려고 '%%'로 바꾼다.
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


# 실제 서버에서 확인: 마이그레이션이 실패하도록 만든 임시 DB 이름을 기록해 두고, 실패 뒤
# pg_database에 그 이름이 남아 있지 않은지 관리 연결로 조회한다(실패한 테스트가 임시 DB를 남기지
# 않음).
def test_temporary_database_is_dropped_when_migration_fails(pg_engine):
    """pg_engine이 쓰는 임시 DB 도구가 마이그레이션 실패에도 DB를 지우는지(T30)."""
    from conftest import temporary_database
    from sqlalchemy.engine import make_url

    from bike_demand.serving.db import database_url

    def failing_upgrade(config, revision):
        created.append(make_url(config.get_main_option("sqlalchemy.url")).database)
        raise RuntimeError("migration failed")

    created: list[str] = []
    with pytest.raises(RuntimeError, match="migration failed"):
        with temporary_database(upgrade=failing_upgrade):
            pass
    admin_url = make_url(database_url()).set(database="postgres")
    from sqlalchemy import create_engine

    admin = create_engine(admin_url)
    with admin.connect() as conn:
        left = conn.execute(
            text("select count(*) from pg_database where datname = :n"), {"n": created[0]}
        ).scalar_one()
    admin.dispose()
    assert left == 0
