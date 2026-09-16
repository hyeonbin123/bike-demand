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
from datetime import datetime, timedelta
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
    """수집 → 그날 Parquet 재생성 → 적재. 0시대에 돌면 전날도 다시 만들고 적재한다.

    23:59대에 수집은 됐지만 적재가 실패하고, 재시도가 자정을 넘기면 그 스냅샷은 전날 폴더에만
    남는다(T35). 적재는 이미 있는 행을 건너뛰므로 전날을 다시 적재해도 중복되지 않는다.
    """
    collected_at = realtime._kst(now or datetime.now(realtime.KST))
    day = collected_at.date()
    path, status = realtime.download(raw_dir, service_key, collected_at, client=client)
    days = [day - timedelta(days=1), day] if collected_at.hour == 0 else [day]
    result: dict = {"day": day.isoformat(), "raw": str(path), "status": status, "days": {}}
    for target in days:
        if not (raw_dir / f"date={target:%Y-%m-%d}").exists():
            continue
        parquet = bronze_dir / f"date={target:%Y-%m-%d}" / "snapshots.parquet"
        rows = realtime.to_parquet(raw_dir, parquet, target)
        snapshots, stations = load.load_realtime(engine, bronze_dir, target)
        result["days"][target.isoformat()] = {
            "parquet_rows": rows,
            "snapshots_inserted": snapshots,
            "new_stations": stations,
        }
    today = result["days"][day.isoformat()]
    result.update(
        parquet_rows=today["parquet_rows"],
        snapshots_inserted=sum(d["snapshots_inserted"] for d in result["days"].values()),
        new_stations=sum(d["new_stations"] for d in result["days"].values()),
    )
    return result


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
