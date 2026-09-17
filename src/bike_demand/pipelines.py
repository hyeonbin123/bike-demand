"""Airflow 작업 하나가 부르는 묶음 명령.

- latest-trip-month: 대여이력 원본 중 가장 최근 달(YYYY-MM, --last-day면 그달 마지막 날).
  반기 갱신 DAG가 날씨 수집과 달력(dbt var calendar_end)에 같은 끝을 넘기는 데 쓴다(T32).
- collect-realtime: 실시간 스냅샷 수집 → 그날 Parquet 재생성 → 서비스 DB 적재.
  적재할 날짜를 **수집 시각**(KST)에서 정한다. 수집과 적재를 따로 돌리면 23:59대에 수집한 스냅샷을
  자정 뒤에 적재할 때 날짜가 달라져 빠질 수 있다(T23). 적재가 실패해 남은 지난 날의 원본은
  다음 정상 실행이 회수한다(T35, T39).
"""

from __future__ import annotations

import calendar
import os
from datetime import date, datetime, timedelta
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


# 적재에 성공한 원본 파일 이름 목록. 원본 목록과 다르면 그날은 아직 다 적재되지 않은 것이다.
LOADED_MARKER = "loaded.txt"
# 이보다 오래된 날은 매번 살피지 않는다. 그런 날은 원본 폴더를 보고 직접 다시 적재한다.
RECOVER_DAYS = 7


def _raw_names(raw_dir: Path, day: date) -> list[str]:
    return sorted(path.name for path in (raw_dir / f"date={day:%Y-%m-%d}").glob("*.json"))


def _marker(bronze_dir: Path, day: date) -> Path:
    return bronze_dir / f"date={day:%Y-%m-%d}" / LOADED_MARKER


def _loaded_names(bronze_dir: Path, day: date) -> list[str] | None:
    path = _marker(bronze_dir, day)
    return path.read_text(encoding="utf-8").split() if path.exists() else None


def _mark_loaded(bronze_dir: Path, day: date, names: list[str]) -> None:
    path = _marker(bronze_dir, day)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(f"{name}\n" for name in names), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def collect_realtime(
    engine: Engine,
    raw_dir: Path,
    bronze_dir: Path,
    service_key: str,
    now: datetime | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """수집 → 그날 Parquet 재생성 → 적재. 덜 적재된 지난 날도 다시 만들고 적재한다.

    적재가 실패하면 그 날의 스냅샷은 원본 폴더에만 남는다. 23:59대 수집분의 재시도가 자정을 넘기거나
    DB 장애가 몇 시간 이어지면 수집 날짜가 이미 지나 있다(T35, T39). 그래서 날마다 적재에 성공한
    원본 이름을 bronze 폴더의 loaded.txt에 남기고, 최근 RECOVER_DAYS일 중 원본 목록과 기록이 다른
    날을 오늘과 함께 다시 적재한다. 적재는 이미 있는 행을 건너뛰므로 중복되지 않는다.
    """
    collected_at = realtime._kst(now or datetime.now(realtime.KST))
    day = collected_at.date()
    path, status = realtime.download(raw_dir, service_key, collected_at, client=client)
    past = [day - timedelta(days=n) for n in range(RECOVER_DAYS, 0, -1)]
    pending = [d for d in past if _raw_names(raw_dir, d) not in ([], _loaded_names(bronze_dir, d))]
    result: dict = {"day": day.isoformat(), "raw": str(path), "status": status, "days": {}}
    for target in [*pending, day]:
        names = _raw_names(raw_dir, target)
        parquet = bronze_dir / f"date={target:%Y-%m-%d}" / "snapshots.parquet"
        rows = realtime.to_parquet(raw_dir, parquet, target)
        snapshots, stations = load.load_realtime(engine, bronze_dir, target)
        _mark_loaded(bronze_dir, target, names)
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
