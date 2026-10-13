"""Airflow 작업 하나가 부르는 묶음 명령.

- latest-trip-month: 대여이력 원본 중 가장 최근 달(YYYY-MM, --last-day면 그달 마지막 날).
  반기 갱신 DAG가 날씨 수집과 달력(dbt var calendar_end)에 같은 끝을 넘기는 데 쓴다(T32).
- collect-realtime: 실시간 스냅샷 수집 → 그날 Parquet 재생성 → 서비스 DB 적재.
  적재할 날짜를 **수집 시각**(KST)에서 정한다. 수집과 적재를 따로 돌리면 23:59대에 수집한 스냅샷을
  자정 뒤에 적재할 때 날짜가 달라져 빠질 수 있다(T23). 적재가 실패해 남은 지난 날의 원본은
  다음 정상 실행이 회수한다(T35, T39).
"""

# [파일 역할] Airflow DAG의 BashOperator가 `python -m bike_demand.pipelines <명령>`으로 부르는
# 진입점이다. 여러 단계를 한 프로세스 안에서 묶어 실행해야 순서·날짜가 어긋나지 않는 작업만 여기에
# 둔다.
# [왜 묶나] 수집·Parquet 변환·DB 적재를 Airflow 작업 셋으로 나누면 각 작업이 '오늘 날짜'를 따로
# 계산해 자정 무렵 날짜가 달라질 수 있다(모듈 docstring). 한 함수 안에서 수집 시각 하나로 날짜를
# 정하면 이 틈이 없어진다.
# [연결] collect-realtime은 airflow/dags/collect_realtime.py가 10분마다, latest-trip-month는
# airflow/dags/refresh_history.py가 날씨 수집 범위와 dbt calendar_end를 정할 때 부른다. 실제
# 수집·변환은 ingest/realtime.py, 적재는 serving/load.py가 한다.

from __future__ import annotations

import calendar
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

# httpx: 실시간 API 호출용 클라이언트를 테스트에서 가짜로 넣을 수 있게 타입으로 받는다. Engine:
# 서비스 DB(PostgreSQL) 연결.
import httpx
from sqlalchemy import Engine

# realtime: 따릉이 실시간 대여정보 다운로드·날짜별 Parquet 변환. trips: 대여이력 원본 목록(가장 최근
# 달 찾기). load: Parquet을 서비스 DB에 적재.
from bike_demand.ingest import realtime, trips
from bike_demand.serving import load


# 대여이력 원본 폴더(CSV·ZIP)에서 가장 최근 달을 'YYYY-MM'으로 돌려준다. trips.discover_sources가 달
# 순으로 정렬해 주므로 마지막 항목이 최신이다. 원본이 하나도 없으면 SystemExit로 멈춘다.
def latest_trip_month(raw_dir: Path) -> str:
    sources = trips.discover_sources(raw_dir)
    if not sources:
        raise SystemExit(f"대여이력 원본이 없음: {raw_dir}")
    return sources[-1].month


# 'YYYY-MM'을 그달 말일 'YYYY-MM-DD'로 바꾼다(윤년 2월 포함, calendar.monthrange). dbt의
# calendar_end var에 넘기는 값이다.
def last_day(month: str) -> str:
    year, mon = map(int, month.split("-"))
    return f"{month}-{calendar.monthrange(year, mon)[1]:02d}"


# 적재에 성공한 원본 파일 이름 목록. 원본 목록과 다르면 그날은 아직 다 적재되지 않은 것이다.
LOADED_MARKER = "loaded.txt"
# 이보다 오래된 날은 매번 살피지 않는다. 그런 날은 원본 폴더를 보고 직접 다시 적재한다.
RECOVER_DAYS = 7


# 재생성·적재·기록 구간의 잠금. 겹친 실행이 같은 날을 두 번 다시 만드는 낭비를 줄이는 용도다.
# 적재가 빠지지 않는 것은 잠금에 기대지 않는다: 실행마다 자기 Parquet 파일을 만들어 그 파일을
# 적재하고, loaded.txt에는 그 파일을 만들기 전에 본 원본 이름만 적는다(원본은 지워지지 않으므로
# 적힌 이름은 모두 적재된 파일 안에 있다). 그래서 오래 걸린 실행의 잠금을 다른 실행이 가져가도
# 기록이 적재보다 앞서지 않는다(T42, T44).
LOCK_NAME = ".collect-realtime.lock"
# 한 번 실행은 1분 안쪽이다. 이보다 오래된 잠금은 중단된 실행이 남긴 것으로 보고 가져간다.
STALE_LOCK_SECONDS = 30 * 60


# 잠금을 얻지 못했을 때 내는 예외. 원본 수집은 이미 끝난 뒤라 이번 스냅샷은 잃지 않고, 적재는
# Airflow 재시도나 다음 실행의 회수 단계가 맡는다.
class LoadBusy(RuntimeError):
    """다른 실행이 적재 중이다. Airflow 재시도나 다음 실행이 이어받는다."""


# bronze 폴더에 잠금 파일을 만들어 한 번에 한 실행만 재생성·적재·기록 구간에 들어가게 한다.
# - O_CREAT | O_EXCL: 파일이 없을 때만 원자적으로 만든다. 이미 있으면 다른 실행이 들어가 있는 것이라
#   LoadBusy.
# - STALE_LOCK_SECONDS보다 오래된 잠금은 중단된 실행(컨테이너 종료 등)이 남긴 것으로 보고 지운다.
#   이렇게 하지 않으면 잠금 하나가 남아 수집이 영원히 막힌다.
# - 잠금에 무작위 토큰을 적고, 끝날 때 토큰이 자기 것일 때만 지운다. 오래 걸려 잠금을 빼앗긴 실행이
#   끝나면서 새 주인의 잠금을 지우는 일을 막는다.
@contextmanager
def _load_lock(bronze_dir: Path) -> Iterator[None]:
    """잠금에 자기 표식을 적고, 풀 때는 표식이 그대로일 때만 지운다(가져간 실행의 잠금 보존)."""
    bronze_dir.mkdir(parents=True, exist_ok=True)
    lock = bronze_dir / LOCK_NAME
    try:
        if time.time() - lock.stat().st_mtime > STALE_LOCK_SECONDS:
            lock.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    token = uuid.uuid4().hex
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise LoadBusy("다른 실시간 적재가 실행 중") from None
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(token)
    try:
        yield
    finally:
        try:
            if lock.read_text(encoding="utf-8") == token:
                lock.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


# 그날 원본 폴더(date=YYYY-MM-DD)의 JSON 파일 이름 목록(정렬). loaded.txt와 비교하는 기준이다.
def _raw_names(raw_dir: Path, day: date) -> list[str]:
    return sorted(path.name for path in (raw_dir / f"date={day:%Y-%m-%d}").glob("*.json"))


# 그날 bronze 폴더 안의 loaded.txt 경로.
def _marker(bronze_dir: Path, day: date) -> Path:
    return bronze_dir / f"date={day:%Y-%m-%d}" / LOADED_MARKER


# loaded.txt에 적힌 원본 이름 목록. 파일이 없으면 None(한 번도 적재에 성공하지 않은 날).
def _loaded_names(bronze_dir: Path, day: date) -> list[str] | None:
    path = _marker(bronze_dir, day)
    return path.read_text(encoding="utf-8").split() if path.exists() else None


# 적재에 성공한 원본 이름을 loaded.txt에 쓴다. 임시 파일에 다 쓴 뒤 os.replace로 바꿔 끼워, 쓰다
# 중단돼도 반쯤 쓴 기록이 남지 않게 한다(원자적 교체). 줄 끝은 LF로 고정한다.
def _mark_loaded(bronze_dir: Path, day: date, names: list[str]) -> None:
    path = _marker(bronze_dir, day)
    tmp = path.with_name(f"{LOADED_MARKER}.{uuid.uuid4().hex}.tmp")
    tmp.write_text("".join(f"{name}\n" for name in names), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# collect-realtime 명령의 본체. 한 번 수집하고, 오늘과 덜 적재된 최근 날들의 Parquet을 다시 만들어
# 서비스 DB에 적재한 뒤 결과 요약 사전을 돌려준다.
# now와 client는 테스트에서 시각과 HTTP 응답을 고정하려고 받는다(운영에서는 None).
# 주의: 실행마다 그날 원본 전체로 Parquet을 다시 만들고 적재를 시도한다(이미 있는 행은 적재 단계가
# 건너뜀). 덜 적재된 지난 날을 오늘보다 먼저 처리하므로, 지난 날 하나가 계속 실패하면 오늘 적재도 그
# 실행에서는 진행되지 않는다.
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
    수집한 원본은 먼저 저장하고, 재생성부터 기록까지는 보통 한 번에 한 실행만 한다(겹치면 LoadBusy).
    7일보다 오래된 날은 매 실행이 살피지 않으므로 `realtime.to_parquet`로 Parquet을 다시 만든 뒤
    `serving.load realtime --day`로 직접 적재한다. 이 수동 적재는 이미 있는 행을 건너뛸 뿐
    loaded.txt를 쓰지 않으므로 수집 작업과 겹쳐도 기록이 어긋나지 않는다.
    """
    # 수집 시각을 KST로 고정하고 그 날짜를 '오늘'로 삼는다. 이후 모든 단계가 이 날짜 하나를 쓴다.
    collected_at = realtime._kst(now or datetime.now(realtime.KST))
    day = collected_at.date()
    # 원본 JSON은 잠금 밖에서 먼저 저장한다. 다른 실행이 적재 중이어도 이번 스냅샷 원본은 남아
    # 나중에 회수된다.
    path, status = realtime.download(raw_dir, service_key, collected_at, client=client)
    result: dict = {"day": day.isoformat(), "raw": str(path), "status": status, "days": {}}
    with _load_lock(bronze_dir):
        # 회수 대상: 최근 7일(오래된 날부터) 중 원본이 있고, 원본 목록이 loaded.txt 기록과 다른 날.
        # 원본이 없는 날([])과 기록이 원본과 같은 날은 건너뛴다.
        past = [day - timedelta(days=n) for n in range(RECOVER_DAYS, 0, -1)]
        pending = [
            d for d in past if _raw_names(raw_dir, d) not in ([], _loaded_names(bronze_dir, d))
        ]
        # names는 Parquet을 만들기 전에 읽는다. 그 뒤에 새 원본이 생겨도 Parquet에는 들어가고
        # 기록에는 빠질 뿐이라, 기록이 실제 적재보다 앞서는 일은 없다(LOCK_NAME 위 주석).
        for target in [*pending, day]:
            names = _raw_names(raw_dir, target)
            parquet = bronze_dir / f"date={target:%Y-%m-%d}" / "snapshots.parquet"
            # 실행마다 자기 이름(uuid)의 임시 Parquet을 만들어 그 파일을 적재한다. 다른 실행이 같은
            # 날의 snapshots.parquet을 바꿔 끼워도 이 실행이 적재하는 내용은 바뀌지 않는다. 적재에
            # 성공한 뒤에야 정식 이름으로 교체하고, 실패하면 임시 파일만 지워진다.
            own = parquet.with_name(f"snapshots.{uuid.uuid4().hex}.parquet.part")
            try:
                rows = realtime.to_parquet(raw_dir, own, target)
                snapshots, stations = load.load_realtime_file(engine, own)
                os.replace(own, parquet)
            finally:
                own.unlink(missing_ok=True)
            # 적재가 예외 없이 끝난 날만 기록한다. 실패한 날은 기록이 원본과 달라 다음 실행이 다시
            # 회수한다.
            _mark_loaded(bronze_dir, target, names)
            result["days"][target.isoformat()] = {
                "parquet_rows": rows,
                "snapshots_inserted": snapshots,
                "new_stations": stations,
            }
    # 요약: Parquet 행 수는 오늘 것, 넣은 스냅샷·새 대여소 수는 회수한 날까지 합친 값. Airflow
    # 로그에 이 사전이 찍힌다.
    today = result["days"][day.isoformat()]
    result.update(
        parquet_rows=today["parquet_rows"],
        snapshots_inserted=sum(d["snapshots_inserted"] for d in result["days"].values()),
        new_stations=sum(d["new_stations"] for d in result["days"].values()),
    )
    return result


# 명령줄: `python -m bike_demand.pipelines collect-realtime [--raw --bronze]` 또는
# `latest-trip-month [--raw] [--last-day]`. latest-trip-month는 결과를 표준 출력으로만 내고 끝나므로
# 인증키나 DB 없이 돈다(refresh_history DAG가 셸 $(...)로 그 출력을 받아 쓴다).
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

    # 서울 열린데이터광장 인증키는 .env에서 읽는다. 키는 코드에 두지 않는다.
    load_dotenv()
    key = os.environ.get("SEOUL_OPEN_API_KEY")
    if not key:
        raise SystemExit(".env에 SEOUL_OPEN_API_KEY가 없음 (.env.example 참고)")
    # API 오류는 스택 추적 없이 메시지만 남기고 실패 종료해 Airflow가 작업 실패로 표시하게 한다.
    try:
        print(collect_realtime(make_engine(), args.raw, args.bronze, key))
    except realtime.ApiError as exc:
        raise SystemExit(str(exc)) from None
