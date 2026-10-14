"""서비스 DB에 넣기. 모든 함수는 멱등이다(다시 돌려도 행이 늘지 않음). docs/serving-schema.md

- stations: warehouse의 dim_stations(과거 이력)에서 upsert. 실시간 스냅샷에만 있는 새 대여소는
  load_realtime이 source=realtime으로 추가한다(이미 있으면 건드리지 않음)
- realtime_snapshots: 날짜별 bronze Parquet에서 추가 (station_id, fetched_at이 같으면 건너뜀)
- weather_forecasts: 단기예보 bronze Parquet에서 추가 (발표·예보 시각·항목·격자가 같으면 건너뜀)
"""

# [파일 역할] bronze Parquet·warehouse에 있는 자료를 서비스 DB(PostgreSQL)의
# stations·realtime_snapshots·weather_forecasts에 넣는다.
# [왜 멱등인가] 모든 적재는 같은 입력을 다시 넣어도 행이 늘지 않아야 한다(docs/serving-schema.md
# 적재 규칙). 실시간 수집은 10분마다 그날 Parquet 전체를 다시 적재하고, 예보 적재도 발표 이력 전체를
# 다시 넣으므로, 기본키 충돌 시 건너뛰기(ON CONFLICT DO NOTHING)와 upsert로 중복을 막는다.
# [연결] stations는 refresh_history DAG의 load_stations 작업, forecasts는 forecast_and_predict DAG,
# 실시간 스냅샷은 pipelines.collect_realtime이 load_realtime_file로 부른다. 명령줄(`python -m
# bike_demand.serving.load stations|realtime --day|forecasts`)로 손으로도 돌릴 수 있다.

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from datetime import date, datetime
from pathlib import Path

# duckdb: warehouse의 dim_stations를 읽기 전용으로 읽는다. pyarrow.parquet: bronze Parquet(모든 열이
# 문자열)을 읽는다. postgresql.insert: PostgreSQL 전용 INSERT … ON CONFLICT를 만든다(표준 insert에는
# 없음).
import duckdb
import pyarrow.parquet as pq
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

# 적재 대상 테이블 정의.
from bike_demand.serving.models import RealtimeSnapshot, Station, WeatherForecast

# 한 번에 보내는 행 수. 하루 스냅샷(대여소 약 2,700곳 × 수집 횟수)처럼 수십만 행을 한 문장으로
# 보내지 않고 나눠 보내 한 번에 보내는 양을 제한한다.
BATCH = 5000
# 실시간 API의 대여소 이름은 "102. 망원역 1번출구 앞"처럼 번호가 앞에 붙는다.
_NAME_WITH_NO = re.compile(r"^\s*(\d+)\.\s*(.*?)\s*$")


# 행 목록을 size개씩 잘라 차례로 내준다.
def _batches(rows: list[dict], size: int = BATCH) -> Iterator[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


# 테이블 전체 행 수. 새로 넣은 행 수를 적재 전후 차이로 세는 데 쓴다.
def _count(engine: Engine, table) -> int:
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar_one()


# keys(기본키 열)가 같은 행이 이미 있으면 건너뛰고 새 행만 넣은 뒤, 새로 들어간 행 수를 돌려준다.
# 모든 배치를 한 트랜잭션(engine.begin)으로 넣어 중간에 실패하면 전부 되돌린다.
# 새 행 수를 전후 count(*) 차이로 세므로, 같은 테이블에 적재를 동시에 돌리지 않는다는 전제가
# 있다(docs/serving-schema.md 적재 규칙, Airflow에서 작업별 동시 실행 1). 주의: 테이블이 커질수록
# count(*) 두 번의 비용도 커진다.
def _insert_ignore(engine: Engine, table, rows: Iterable[dict], keys: list[str]) -> int:
    rows = list(rows)
    before = _count(engine, table)
    with engine.begin() as conn:
        for batch in _batches(rows):
            conn.execute(insert(table).on_conflict_do_nothing(index_elements=keys), batch)
    return _count(engine, table) - before


# bronze는 모든 값을 문자열로 두므로 숫자로 바꾼다. '3.0'처럼 소수 꼴로 온 정수도 받으려고 float을
# 거친다. 빈 문자열·None은 None(DB의 NULL).
def _int(value: str | None) -> int | None:
    return int(float(value)) if value not in (None, "") else None


# 문자열 → float, 빈 값은 None.
def _float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


# warehouse의 dim_stations(대여이력에 나온 모든 대여소, 이름·위치는 가장 최근 대여소 정보 스냅샷
# 기준)를 stations에 upsert한다. 이미 있는 대여소는 값을 history 쪽으로 덮어쓰고 source=history,
# updated_at=now()로 바꾼다. 실시간 스냅샷으로 먼저 들어와 있던 대여소도 공식 대여소 정보가 나오면
# 그 값으로 바뀐다(docs/serving-schema.md).
# 돌려주는 값은 넣거나 고친 행 수(=dim_stations 행 수).
def load_stations_from_warehouse(engine: Engine, warehouse: Path) -> int:
    """dim_stations 전체를 upsert하고 넣거나 고친 행 수를 돌려준다."""
    # warehouse는 dbt와 동시에 쓰면 안 되므로 읽기 전용으로 열고, 읽자마자 닫는다.
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
            # ON CONFLICT (station_id) DO UPDATE: excluded는 '넣으려던 새 값'을 가리킨다.
            # station_id를 뺀 열을 모두 새 값으로 바꾼다.
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


# 실시간 API의 대여소 이름 '102. 망원역 1번출구 앞'을 (102, '망원역 1번출구 앞')으로 나눈다. 번호가
# 없는 이름은 (None, 앞뒤 공백을 뗀 이름), 빈 이름은 (None, None). 실시간에만 나오는 새 대여소의
# station_no를 채우는 데 쓴다.
def split_station_name(name: str | None) -> tuple[int | None, str | None]:
    if not name:
        return None, None
    match = _NAME_WITH_NO.match(name)
    if not match:
        return None, name.strip()
    return int(match.group(1)), match.group(2)


# 날짜 하나의 정식 Parquet(date=YYYY-MM-DD/snapshots.parquet)을 적재한다. 명령줄 `load realtime
# --day`(7일보다 오래된 날을 손으로 다시 적재할 때)가 쓴다.
def load_realtime(engine: Engine, bronze_dir: Path, day: date) -> tuple[int, int]:
    """(새로 넣은 스냅샷 행 수, 새로 추가한 대여소 수). 그날의 bronze Parquet을 적재한다."""
    return load_realtime_file(engine, bronze_dir / f"date={day:%Y-%m-%d}" / "snapshots.parquet")


# Parquet 파일 하나를 받아 처음 보는 대여소를 stations에 추가하고, 스냅샷을 realtime_snapshots에
# 넣는다. 둘 다 이미 있으면 건너뛴다. 출력은 (새 스냅샷 행 수, 새 대여소 수).
# 파일 경로를 직접 받는 이유: 수집 작업(pipelines.collect_realtime)은 실행마다 자기 임시 Parquet을
# 만들어 그 파일을 적재한다. 다른 실행이 정식 파일을 바꿔 끼워도 이 실행이 적재하는 내용이 바뀌지
# 않게 하려는 것이다.
def load_realtime_file(engine: Engine, path: Path) -> tuple[int, int]:
    """실시간 스냅샷 Parquet 파일 하나를 적재한다(수집 작업은 실행마다 따로 만든 파일을 넘긴다)."""
    records = pq.read_table(path).to_pylist()
    # 대여소ID가 없거나 자전거 수가 빈 행은 버린다(bike_count는 NOT NULL). fetched_at은
    # 오프셋(+09:00)이 붙은 ISO 문자열이라 시간대가 있는 datetime이 된다.
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
    # 같은 대여소가 하루 여러 스냅샷에 나오므로 대여소ID로 하나만 남긴다(파일 뒤쪽 값이 이김).
    # 실시간 정보에는 자치구가 없어 district는 비운다.
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
    # 대여소는 DO NOTHING으로 넣는다. 이미 있는 대여소(특히 history 출처)를 실시간 값으로 덮어쓰지
    # 않기 위해서다.
    new_stations = _insert_ignore(engine, Station.__table__, stations.values(), ["station_id"])
    new_snapshots = _insert_ignore(
        engine, RealtimeSnapshot.__table__, snapshots, ["station_id", "fetched_at"]
    )
    return new_snapshots, new_stations


# 단기예보 bronze Parquet(발표·예보 시각·항목·격자·값의 긴 형식) 전체를 weather_forecasts에 넣고 새
# 행 수를 돌려준다. 다섯 키가 같은 행은 건너뛴다. 한 발표의 값은 바뀌지 않으므로 덮어쓸 필요가
# 없다(docs/serving-schema.md).
# value는 '강수없음' 같은 문자열이 있어 변환하지 않고 그대로 넣는다.
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


# 명령줄: stations [--warehouse], realtime --day YYYY-MM-DD [--bronze], forecasts [--parquet].
# Airflow DAG도 이 명령을 그대로 부른다.
if __name__ == "__main__":
    import argparse

    from bike_demand.serving.db import make_engine

    parser = argparse.ArgumentParser(description="서비스 DB에 넣기")
    sub = parser.add_subparsers(dest="target", required=True)
    s = sub.add_parser("stations")
    s.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    r = sub.add_parser("realtime")
    r.add_argument(
        "--day",
        type=date.fromisoformat,
        required=True,
        help="적재할 수집 날짜 YYYY-MM-DD(KST 기준). 그날의 정식 snapshots.parquet을 넣는다",
    )
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
