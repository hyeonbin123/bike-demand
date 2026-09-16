"""Airflow 작업 하나가 부르는 묶음 명령.

- latest-trip-month: 대여이력 원본 중 가장 최근 달(YYYY-MM, --last-day면 그달 마지막 날).
  반기 갱신 DAG가 날씨 수집과 달력(dbt var calendar_end)에 같은 끝을 넘기는 데 쓴다(T32).
- collect-realtime: 실시간 스냅샷 수집 → 그날 Parquet 재생성 → 서비스 DB 적재.
  적재할 날짜를 **수집 시각**(KST)에서 정한다. 수집과 적재를 따로 돌리면 23:59대에 수집한 스냅샷을
  자정 뒤에 적재할 때 날짜가 달라져 빠질 수 있다(T23).
"""

from __future__ import annotations

import calendar
import os
from datetime import datetime
from pathlib import Path

import httpx
from sqlalchemy import Engine

from bike_demand.ingest import realtime, trips
from bike_demand.serving import load


def latest_trip_month(raw_dir: Path) -> str:
    sources = trips.discover_sources(raw_dir)
    if not sources:
        raise SystemExit(f"대여이력 원본이 없음: {raw_dir}")
    return sources[-1].month


def last_day(month: str) -> str:
    year, mon = map(int, month.split("-"))
    return f"{month}-{calendar.monthrange(year, mon)[1]:02d}"


def collect_realtime(
    engine: Engine,
    raw_dir: Path,
    bronze_dir: Path,
    service_key: str,
    now: datetime | None = None,
    client: httpx.Client | None = None,
) -> dict:
    collected_at = realtime._kst(now or datetime.now(realtime.KST))
    day = collected_at.date()
    path, status = realtime.download(raw_dir, service_key, collected_at, client=client)
    parquet = bronze_dir / f"date={day:%Y-%m-%d}" / "snapshots.parquet"
    rows = realtime.to_parquet(raw_dir, parquet, day)
    snapshots, stations = load.load_realtime(engine, bronze_dir, day)
    return {
        "day": day.isoformat(),
        "raw": str(path),
        "status": status,
        "parquet_rows": rows,
        "snapshots_inserted": snapshots,
        "new_stations": stations,
    }


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    from bike_demand.serving.db import make_engine

    parser = argparse.ArgumentParser(description="Airflow 작업용 묶음 명령")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("collect-realtime")
    r.add_argument("--raw", type=Path, default=Path("data/raw/realtime/bikelist"))
    r.add_argument("--bronze", type=Path, default=Path("data/bronze/realtime/bikelist"))
    m = sub.add_parser("latest-trip-month")
    m.add_argument("--raw", type=Path, default=Path("data/raw/trips"))
    m.add_argument("--last-day", action="store_true")
    args = parser.parse_args()

    if args.command == "latest-trip-month":
        month = latest_trip_month(args.raw)
        print(last_day(month) if args.last_day else month)
        raise SystemExit(0)

    load_dotenv()
    key = os.environ.get("SEOUL_OPEN_API_KEY")
    if not key:
        raise SystemExit(".env에 SEOUL_OPEN_API_KEY가 없음 (.env.example 참고)")
    try:
        print(collect_realtime(make_engine(), args.raw, args.bronze, key))
    except realtime.ApiError as exc:
        raise SystemExit(str(exc)) from None
