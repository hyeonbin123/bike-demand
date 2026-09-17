"""서비스 DB에 넣기. 모든 함수는 멱등이다(다시 돌려도 행이 늘지 않음). docs/serving-schema.md

- stations: warehouse의 dim_stations(과거 이력)에서 upsert. 실시간 스냅샷에만 있는 새 대여소는
  load_realtime이 source=realtime으로 추가한다(이미 있으면 건드리지 않음)
- realtime_snapshots: 날짜별 bronze Parquet에서 추가 (station_id, fetched_at이 같으면 건너뜀)
- weather_forecasts: 단기예보 bronze Parquet에서 추가 (발표·예보 시각·항목·격자가 같으면 건너뜀)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from datetime import date, datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

from bike_demand.serving.models import RealtimeSnapshot, Station, WeatherForecast

BATCH = 5000
# 실시간 API의 대여소 이름은 "102. 망원역 1번출구 앞"처럼 번호가 앞에 붙는다.
_NAME_WITH_NO = re.compile(r"^\s*(\d+)\.\s*(.*?)\s*$")


def _batches(rows: list[dict], size: int = BATCH) -> Iterator[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _count(engine: Engine, table) -> int:
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar_one()


def _insert_ignore(engine: Engine, table, rows: Iterable[dict], keys: list[str]) -> int:
    rows = list(rows)
    before = _count(engine, table)
    with engine.begin() as conn:
        for batch in _batches(rows):
            conn.execute(insert(table).on_conflict_do_nothing(index_elements=keys), batch)
    return _count(engine, table) - before


def _int(value: str | None) -> int | None:
    return int(float(value)) if value not in (None, "") else None


def _float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def load_stations_from_warehouse(engine: Engine, warehouse: Path) -> int:
    """dim_stations 전체를 upsert하고 넣거나 고친 행 수를 돌려준다."""
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        records = con.execute(
            """select station_id, station_no, station_name, district, lat, lon, docks
               from dim_stations order by station_id"""
        ).fetchall()
    finally:
        con.close()
    columns = ["station_id", "station_no", "station_name", "district", "lat", "lon", "docks"]
    rows = [{**dict(zip(columns, r, strict=True)), "source": "history"} for r in records]
    table = Station.__table__
    with engine.begin() as conn:
        for batch in _batches(rows):
            statement = insert(table)
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=["station_id"],
                    set_={
                        **{c: statement.excluded[c] for c in columns[1:]},
                        "source": "history",
                        "updated_at": func.now(),
                    },
                ),
                batch,
            )
    return len(rows)


def split_station_name(name: str | None) -> tuple[int | None, str | None]:
    if not name:
        return None, None
    match = _NAME_WITH_NO.match(name)
    if not match:
        return None, name.strip()
    return int(match.group(1)), match.group(2)


def load_realtime(engine: Engine, bronze_dir: Path, day: date) -> tuple[int, int]:
    """(새로 넣은 스냅샷 행 수, 새로 추가한 대여소 수). 그날의 bronze Parquet을 적재한다."""
    return load_realtime_file(engine, bronze_dir / f"date={day:%Y-%m-%d}" / "snapshots.parquet")


def load_realtime_file(engine: Engine, path: Path) -> tuple[int, int]:
    """실시간 스냅샷 Parquet 파일 하나를 적재한다(수집 작업은 실행마다 따로 만든 파일을 넘긴다)."""
    records = pq.read_table(path).to_pylist()
    snapshots = [
        {
            "station_id": r["station_id"],
            "fetched_at": datetime.fromisoformat(r["fetched_at"]),
            "bike_count": _int(r["bike_count"]),
            "rack_count": _int(r["rack_count"]),
        }
        for r in records
        if r["station_id"] and r["bike_count"] not in (None, "")
    ]
    stations: dict[str, dict] = {}
    for r in records:
        if not r["station_id"]:
            continue
        station_no, station_name = split_station_name(r["station_name"])
        stations[r["station_id"]] = {
            "station_id": r["station_id"],
            "station_no": station_no,
            "station_name": station_name,
            "lat": _float(r["lat"]),
            "lon": _float(r["lon"]),
            "docks": _int(r["rack_count"]),
            "source": "realtime",
        }
    new_stations = _insert_ignore(engine, Station.__table__, stations.values(), ["station_id"])
    new_snapshots = _insert_ignore(
        engine, RealtimeSnapshot.__table__, snapshots, ["station_id", "fetched_at"]
    )
    return new_snapshots, new_stations


def load_forecasts(engine: Engine, parquet_path: Path) -> int:
    rows = [
        {
            "base_datetime": datetime.fromisoformat(r["base_datetime"]),
            "fcst_datetime": datetime.fromisoformat(r["fcst_datetime"]),
            "category": r["category"],
            "nx": int(r["nx"]),
            "ny": int(r["ny"]),
            "value": r["value"],
        }
        for r in pq.read_table(parquet_path).to_pylist()
    ]
    return _insert_ignore(
        engine,
        WeatherForecast.__table__,
        rows,
        ["base_datetime", "fcst_datetime", "category", "nx", "ny"],
    )


if __name__ == "__main__":
    import argparse

    from bike_demand.serving.db import make_engine

    parser = argparse.ArgumentParser(description="서비스 DB에 넣기")
    sub = parser.add_subparsers(dest="target", required=True)
    s = sub.add_parser("stations")
    s.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    r = sub.add_parser("realtime")
    r.add_argument("--day", type=date.fromisoformat, required=True, help="YYYY-MM-DD (KST)")
    r.add_argument("--bronze", type=Path, default=Path("data/bronze/realtime/bikelist"))
    f = sub.add_parser("forecasts")
    f.add_argument("--parquet", type=Path, default=Path("data/bronze/weather/vilage_fcst.parquet"))
    args = parser.parse_args()

    engine = make_engine()
    if args.target == "stations":
        print("stations upserted", load_stations_from_warehouse(engine, args.warehouse))
    elif args.target == "realtime":
        snapshots, stations = load_realtime(engine, args.bronze, args.day)
        print("snapshots inserted", snapshots, "new stations", stations)
    else:
        print("forecasts inserted", load_forecasts(engine, args.parquet))
