"""서비스 DB에 넣기. 모든 함수는 멱등이다(다시 돌려도 행이 늘지 않음). docs/serving-schema.md

- stations: warehouse의 dim_stations(과거 이력)에서 upsert. 실시간 스냅샷에만 있는 새 대여소는
  load_realtime이 source=realtime으로 추가한다(이미 있으면 건드리지 않음)
- realtime_snapshots: 날짜별 bronze Parquet에서 추가 (station_id, fetched_at이 같으면 건너뜀).
  값은 행마다 검사해, DB에 넣을 수 없는 대여소ID·자전거 수의 행은 건너뛰고 거치대 수·좌표·이름처럼
  비워도 되는 값은 그 값만 비운다(realtime_rows). 한 행 때문에 그날 적재 전체가 실패하지 않는다
- weather_forecasts: 단기예보 bronze Parquet에서 추가 (발표·예보 시각·항목·격자가 같으면 건너뜀)
- demo: 고정 시각 데모 묶음(대여소·스냅샷·예보·예측 Parquet + manifest.json)을 네 테이블이 모두 빈
  DB에만 한 트랜잭션으로 넣는다. 비어 있지 않으면 아무것도 넣지 않고 거부한다(개발 DB에 섞이지 않게)
"""

# [파일 역할] bronze Parquet·warehouse에 있는 자료를 서비스 DB(PostgreSQL)의
# stations·realtime_snapshots·weather_forecasts에 넣는다. 고정 시각 데모 묶음의
# 형식(쓰기·읽기·검사)과 그 묶음을 빈 DB에 넣는 적재도 여기 둔다(묶음을 만드는 쪽은
# scripts/build_demo_bundle.py).
# [왜 멱등인가] 모든 적재는 같은 입력을 다시 넣어도 행이 늘지 않아야 한다(docs/serving-schema.md
# 적재 규칙). 실시간 수집은 10분마다 그날 Parquet 전체를 다시 적재하고, 예보 적재도 발표 이력 전체를
# 다시 넣으므로, 기본키 충돌 시 건너뛰기(ON CONFLICT DO NOTHING)와 upsert로 중복을 막는다.
# [연결] stations는 refresh_history DAG의 load_stations 작업, forecasts는 forecast_and_predict DAG,
# 실시간 스냅샷은 pipelines.collect_realtime이 load_realtime_file로 부른다. 명령줄(`python -m
# bike_demand.serving.load stations|realtime --day|forecasts|demo --bundle`)로 손으로도 돌릴 수
# 있다.

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

# duckdb: warehouse의 dim_stations를 읽기 전용으로 읽는다. pyarrow: bronze Parquet(모든 열이
# 문자열)을 읽고 데모 묶음 Parquet을 쓴다. postgresql.insert: PostgreSQL 전용 INSERT … ON CONFLICT를
# 만든다(표준 insert에는 없음).
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

# 수집기의 bronze 열 구성(모든 열 문자열). 데모 묶음의 스냅샷·예보 파일이 이 구성 그대로라, 적재가
# 서비스와 같은 변환(realtime_rows, forecast_rows)을 거친다.
from bike_demand.ingest.forecast import SCHEMA as FORECAST_SCHEMA
from bike_demand.ingest.realtime import SCHEMA as REALTIME_SCHEMA

# 적재 대상 테이블 정의.
from bike_demand.serving.models import Prediction, RealtimeSnapshot, Station, WeatherForecast

# 한 번에 보내는 행 수. 하루 스냅샷(대여소 약 2,700곳 × 수집 횟수)처럼 수십만 행을 한 문장으로
# 보내지 않고 나눠 보내 한 번에 보내는 양을 제한한다.
BATCH = 5000
# 실시간 API의 대여소 이름은 "102. 망원역 1번출구 앞"처럼 번호가 앞에 붙는다. 이 정규식은 앞
# 공백과 번호·점까지만 찾고(re.match로 문자열 처음에 고정), 나머지 이름의 앞뒤 공백은
# str.strip()으로 뗀다. 예전에는 한 정규식 `^\s*(\d+)\.\s*(.*?)\s*$`로 이름 끝 공백까지 맞췄는데,
# `(.*?)\s*$`는 이름 가운데의 긴 공백에서 길이의 제곱에 비례해 느려졌다(공백 4만 개에 약 12초).
# 나누는 결과는 예전과 같다(split_station_name 주석).
_NAME_NUMBER = re.compile(r"\s*(\d+)\.")
# DB 열이 받을 수 있는 범위(serving/models.py): integer 열의 상한, station_id varchar(20),
# station_name varchar(200). 넘는 값을 그대로 보내면 PostgreSQL이 그 문장 전체를 거부한다.
INT32_MAX = 2_147_483_647
STATION_ID_MAX = 20
STATION_NAME_MAX = 200

# 적재 경고(건너뛴 행·비운 값)를 남기는 로거. 명령줄·Airflow 실행에는 로깅 설정이 없어 파이썬 기본
# 처리기가 WARNING 이상을 표준 오류로 내보내므로 작업 로그에 그대로 보인다.
logger = logging.getLogger(__name__)


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


# bronze는 모든 값을 문자열로 두므로 숫자로 바꾼다. 개수(자전거·거치대)는 0 이상 int32 상한 이하의
# 정수만 받는다. '3.0'처럼 소수 꼴로 온 정수도 받으려고 float을 거친다. 빈 값, 숫자가 아닌 값('-',
# 'N/A'), float으로 무한·NaN이 되는 값('1e400', 'nan'), 소수부가 있는 값, 음수, 상한 초과는 모두
# None이다. 전에는 int(float(value))가 '-'에서 ValueError, '1e400'에서 OverflowError를 내 파일 하나
# 전체의 적재가 실패했다. 빈 값인지(원래 결측)와 잘못된 값인지는 호출하는 쪽이 원래 값으로 가린다.
# 이름 앞 대여소번호(split_station_name의 숫자 문자열)도 같은 integer 열 규칙이라 이 함수로 바꾼다.
# float은 int()와 달리 자릿수 한도가 없어 아주 긴 숫자도 예외 없이 무한대 → None이 된다.
def _nonnegative_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer() or not 0 <= number <= INT32_MAX:
        return None
    return int(number)


# 좌표 같은 실수: 유한한 숫자만 받고 빈 값·숫자가 아님·무한·NaN은 None. double 열은 NaN·무한대를
# 받지만, 그 대여소를 담는 API 응답이 실패한다(JSON은 NaN·무한대를 담을 수 없어 FastAPI 기본
# JSONResponse가 allow_nan=False로 거부).
def _finite_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# varchar(limit) 열에 그대로 넣을 수 있는 문자열인지. PostgreSQL 문자열에는 NUL(\x00)을 넣을 수
# 없고, 길이를 넘으면 그 문장 전체가 거부된다.
def _storable_text(value: object, limit: int) -> bool:
    return isinstance(value, str) and len(value) <= limit and "\x00" not in value


# warehouse dim_stations에서 stations 표로 옮기는 열(source·updated_at은 적재가 채움).
STATION_COLUMNS = ("station_id", "station_no", "station_name", "district", "lat", "lon", "docks")


# warehouse의 dim_stations를 STATION_COLUMNS 열의 사전 목록으로 읽는다(station_id 순). 서비스 적재
# (load_stations_from_warehouse)와 데모 묶음 만들기(scripts/build_demo_bundle.py)가 같은 행을 쓰도록
# 한 곳에 둔다. warehouse는 dbt와 동시에 쓰면 안 되므로 읽기 전용으로 열고, 읽자마자 닫는다.
def read_warehouse_stations(warehouse: Path) -> list[dict]:
    """dim_stations의 대여소 행(station_id 순). 값은 DuckDB가 준 파이썬 값 그대로."""
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        records = con.execute(
            """select station_id, station_no, station_name, district, lat, lon, docks
               from dim_stations order by station_id"""
        ).fetchall()
    finally:
        con.close()
    return [dict(zip(STATION_COLUMNS, r, strict=True)) for r in records]


# warehouse의 dim_stations(대여이력에 나온 모든 대여소, 이름·위치는 가장 최근 대여소 정보 스냅샷
# 기준)를 stations에 upsert한다. 이미 있는 대여소는 값을 history 쪽으로 덮어쓰고 source=history,
# updated_at=now()로 바꾼다. 실시간 스냅샷으로 먼저 들어와 있던 대여소도 공식 대여소 정보가 나오면
# 그 값으로 바뀐다(docs/serving-schema.md).
# 돌려주는 값은 넣거나 고친 행 수(=dim_stations 행 수).
def load_stations_from_warehouse(engine: Engine, warehouse: Path) -> int:
    """dim_stations 전체를 upsert하고 넣거나 고친 행 수를 돌려준다."""
    rows = [{**r, "source": "history"} for r in read_warehouse_stations(warehouse)]
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
                        **{c: statement.excluded[c] for c in STATION_COLUMNS[1:]},
                        "source": "history",
                        "updated_at": func.now(),
                    },
                ),
                batch,
            )
    return len(rows)


# 실시간 API의 대여소 이름 '102. 망원역 1번출구 앞'을 ('102', '망원역 1번출구 앞')으로 나눈다.
# 번호가 없는 이름은 (None, 앞뒤 공백을 뗀 이름), 빈 이름은 (None, None). 실시간에만 나오는 새
# 대여소의 station_no를 채우는 데 쓴다.
# - 번호는 숫자 문자열 그대로 돌려준다(앞의 0 포함). 정수로 바꾸고 DB 범위(int32)를 확인하는 일은
#   realtime_rows가 다른 정수 값과 같은 규칙(_nonnegative_int)으로 한다. 여기서 int()를 부르면
#   4,300자리를 넘는 숫자(앞의 0 포함)에서 파이썬이 ValueError를 내 그 파일 전체의 적재가
#   실패했다.
# - 나누는 규칙은 예전 정규식 `^\s*(\d+)\.\s*(.*?)\s*$`와 같다: 앞 공백 뒤의 숫자와 바로 뒤 점이
#   번호, 나머지의 앞뒤 공백(줄바꿈 포함)을 뗀 것이 이름이다. 예전 `(.*?)`는 줄바꿈을 건너지
#   못했으므로, 떼고 남은 이름 가운데에 줄바꿈이 있으면 번호가 없는 이름으로 본다. 정규식의 \s와
#   str.strip()은 같은 유니코드 공백 정의(str.isspace)를 쓴다. tests/test_serving_load.py가 경계
#   사례를 고정한다.
def split_station_name(name: str | None) -> tuple[str | None, str | None]:
    if not name:
        return None, None
    match = _NAME_NUMBER.match(name)
    if match is not None:
        rest = name[match.end() :].strip()
        if "\n" not in rest:
            return match.group(1), rest
    return None, name.strip()


# realtime_rows의 결과. snapshots는 realtime_snapshots에, stations는 stations에 넣을 행이다.
# skipped_rows: 스냅샷으로 넣지 못하고 건너뛴 행 수(대여소ID나 자전거 수를 DB에 넣을 수 없음).
# nulled_values: 비워도 되는 열(거치대 수·좌표·대여소번호·이름)에 넣을 수 없는 값이 있어 None으로
# 바꾼 값의 수. 원래 빈 값(결측)은 어느 쪽에도 세지 않는다.
@dataclass(frozen=True)
class RealtimeRows:
    snapshots: list[dict]
    stations: list[dict]
    skipped_rows: int
    nulled_values: int


# 실시간 bronze 행(모든 값이 문자열)을 DB에 넣을 사전으로 바꾸는 순수 함수(DB를 쓰지 않음). 행마다
# 검사해 넣을 수 없는 값이 있는 행만 건너뛰거나 그 값만 비운다.
# - 대여소ID: 비었거나 varchar(20)보다 길거나 NUL을 담으면 키로 쓸 수 없어 그 행을 건너뛰고
#   대여소로도 등록하지 않는다(수집은 ID가 빈 응답을 이미 거부하므로 원본에서는 드묾).
# - 자전거 수(bike_count, NOT NULL): 0 이상 int32 정수가 아니면 그 행은 스냅샷에서 건너뛴다. 그 행의
#   대여소 정보(이름·거치대·좌표)는 대여소 등록에 그대로 쓴다.
# - 거치대 수(rack_count, 대여소의 docks)·좌표·이름 앞 대여소번호·이름: 넣을 수 없으면 그 값만 None.
# 왜 행 단위인가: 수집 작업은 실행마다 그날 원본 전체로 Parquet을 다시 만들어 적재하므로, 전에는 값
# 하나가 이상한 응답 한 번에 그날 자정까지의 모든 적재가 실패했다. 받을 때 그 쪽을 거부하는 방법은
# 대여소 하나가 계속 이상한 값을 보내면 수집 전체가 멈추므로 쓰지 않는다.
# 같은 대여소가 하루 여러 스냅샷에 나오므로 대여소 정보는 대여소ID로 하나만 남긴다(뒤쪽 행이 이김).
# 실시간 정보에는 자치구가 없어 district는 비운다. fetched_at은 수집기가 쓴 오프셋(+09:00) 붙은 ISO
# 문자열이라 시간대가 있는 datetime이 된다(이 값이 깨진 파일은 행 단위가 아니라 파일 문제라 예외).
def realtime_rows(records: Iterable[dict]) -> RealtimeRows:
    snapshots: list[dict] = []
    stations: dict[str, dict] = {}
    skipped = nulled = 0
    for r in records:
        station_id = r.get("station_id")
        if not station_id or not _storable_text(station_id, STATION_ID_MAX):
            skipped += 1
            continue
        rack_count = _nonnegative_int(r.get("rack_count"))
        lat, lon = _finite_float(r.get("lat")), _finite_float(r.get("lon"))
        # 이름 앞 번호(숫자 문자열)도 station_no integer 열이라 개수와 같은 규칙으로 바꾼다: 0
        # 이상 int32 상한 이하만 받고 넘으면 None. float을 거치므로 int()의 4,300자리 한도에
        # 걸리지 않고, 아주 긴 숫자는 무한대가 되어 None이다. 숫자만으로 된 문자열이라 소수·음수는
        # 나오지 않는다.
        number, station_name = split_station_name(r.get("station_name"))
        station_no = _nonnegative_int(number)
        # 넣을 수 없는 값만 비운다. 원래 값이 비어 있지 않았는데 None이 된 것만 센다.
        nulled += sum(
            parsed is None and raw not in (None, "")
            for parsed, raw in (
                (rack_count, r.get("rack_count")),
                (lat, r.get("lat")),
                (lon, r.get("lon")),
                (station_no, number),
            )
        )
        if station_name is not None and not _storable_text(station_name, STATION_NAME_MAX):
            station_name, nulled = None, nulled + 1
        stations[station_id] = {
            "station_id": station_id,
            "station_no": station_no,
            "station_name": station_name,
            "lat": lat,
            "lon": lon,
            "docks": rack_count,
            "source": "realtime",
        }
        bike_count = _nonnegative_int(r.get("bike_count"))
        if bike_count is None:
            skipped += 1
            continue
        snapshots.append(
            {
                "station_id": station_id,
                "fetched_at": datetime.fromisoformat(r["fetched_at"]),
                "bike_count": bike_count,
                "rack_count": rack_count,
            }
        )
    return RealtimeRows(snapshots, list(stations.values()), skipped, nulled)


# load_realtime·load_realtime_file의 결과. 튜플처럼 풀어 쓸 수 있다(snapshots, stations, skipped =
# load_realtime_file(...)).
class RealtimeLoadResult(NamedTuple):
    snapshots_inserted: int  # 새로 넣은 스냅샷 행 수(이미 있던 행은 세지 않음)
    new_stations: int  # 새로 추가한 대여소 수
    skipped_rows: int  # 넣을 수 없는 값 때문에 건너뛴 행 수(파일 전체 기준)


# 날짜 하나의 정식 Parquet(date=YYYY-MM-DD/snapshots.parquet)을 적재한다. 명령줄 `load realtime
# --day`(7일보다 오래된 날을 손으로 다시 적재할 때)가 쓴다.
def load_realtime(engine: Engine, bronze_dir: Path, day: date) -> RealtimeLoadResult:
    """그날의 bronze Parquet을 적재한다. (새 스냅샷 행 수, 새 대여소 수, 건너뛴 행 수)."""
    return load_realtime_file(engine, bronze_dir / f"date={day:%Y-%m-%d}" / "snapshots.parquet")


# Parquet 파일 하나를 받아 처음 보는 대여소를 stations에 추가하고, 스냅샷을 realtime_snapshots에
# 넣는다. 둘 다 이미 있으면 건너뛴다. 값 검사는 realtime_rows가 하고, 건너뛴 행이나 비운 값이
# 있으면 날짜 폴더 이름과 함께 WARNING 로그를 남긴다. 수집 작업은 그날 Parquet 전체를 실행마다 다시
# 적재하므로 같은 나쁜 행이 그날 남은 실행마다 다시 건너뛰어지고 다시 경고된다(행이 늘지는 않음).
# 출력은 RealtimeLoadResult(새 스냅샷 행 수, 새 대여소 수, 건너뛴 행 수).
# 파일 경로를 직접 받는 이유: 수집 작업(pipelines.collect_realtime)은 실행마다 자기 임시 Parquet을
# 만들어 그 파일을 적재한다. 다른 실행이 정식 파일을 바꿔 끼워도 이 실행이 적재하는 내용이 바뀌지
# 않게 하려는 것이다.
def load_realtime_file(engine: Engine, path: Path) -> RealtimeLoadResult:
    """실시간 스냅샷 Parquet 파일 하나를 적재한다(수집 작업은 실행마다 따로 만든 파일을 넘긴다)."""
    records = pq.read_table(path).to_pylist()
    rows = realtime_rows(records)
    if rows.skipped_rows or rows.nulled_values:
        logger.warning(
            "실시간 적재 %s: 넣을 수 없는 값 때문에 행 %d개를 건너뛰고 값 %d개를 비움 (전체 %d행)",
            path.parent.name,
            rows.skipped_rows,
            rows.nulled_values,
            len(records),
        )
    # 대여소는 DO NOTHING으로 넣는다. 이미 있는 대여소(특히 history 출처)를 실시간 값으로 덮어쓰지
    # 않기 위해서다.
    new_stations = _insert_ignore(engine, Station.__table__, rows.stations, ["station_id"])
    new_snapshots = _insert_ignore(
        engine, RealtimeSnapshot.__table__, rows.snapshots, ["station_id", "fetched_at"]
    )
    return RealtimeLoadResult(new_snapshots, new_stations, rows.skipped_rows)


# 단기예보 bronze 행(모든 값이 문자열)을 weather_forecasts에 넣을 사전으로 바꾸는 순수 함수(DB를
# 쓰지 않음). 발표·예보 시각은 수집기가 쓴 오프셋(+09:00) 붙은 ISO 문자열이라 시간대가 있는
# datetime이 되고, 격자는 정수다. value는 '강수없음' 같은 문자열이 있어 변환하지 않고 그대로 둔다.
# fetched_at은 표에 없어 버린다. 서비스 적재(load_forecasts)와 데모 적재(load_demo)가 같은 변환을
# 쓴다.
def forecast_rows(records: Iterable[dict]) -> list[dict]:
    return [
        {
            "base_datetime": datetime.fromisoformat(r["base_datetime"]),
            "fcst_datetime": datetime.fromisoformat(r["fcst_datetime"]),
            "category": r["category"],
            "nx": int(r["nx"]),
            "ny": int(r["ny"]),
            "value": r["value"],
        }
        for r in records
    ]


# 단기예보 bronze Parquet(발표·예보 시각·항목·격자·값의 긴 형식) 전체를 weather_forecasts에 넣고 새
# 행 수를 돌려준다. 다섯 키가 같은 행은 건너뛴다. 한 발표의 값은 바뀌지 않으므로 덮어쓸 필요가
# 없다(docs/serving-schema.md).
def load_forecasts(engine: Engine, parquet_path: Path) -> int:
    rows = forecast_rows(pq.read_table(parquet_path).to_pylist())
    return _insert_ignore(
        engine,
        WeatherForecast.__table__,
        rows,
        ["base_datetime", "fcst_datetime", "category", "nx", "ny"],
    )


# --- 고정 시각 데모 묶음 ---------------------------------------------------------------------
# [무엇] 서비스가 한 시각에 갖고 있던 자료(대여소 표, 스냅샷 한 번, 그 시각의 최신 예보 발표, 그
# 발표로 만든 예측 한 버전)를 작은 Parquet 묶음으로 두고, 빈 서비스 DB에 넣어 API·대시보드로 그
# 시각의 화면을 다시 보여 준다(api/main.py의 데모 시계 BIKE_DEMAND_DEMO_NOW와 함께 쓴다).
# [왜] 실시간 API는 지금 상태만 주므로 지난 시각의 스냅샷은 다시 받을 수 없고, 화면을 다시 만들려면
# 인증키·Airflow·서비스 모델(model.txt 약 63 MB)이 필요하다. 묶음은 이것 없이 서비스 DB만 있으면
# 된다.
# [형식] 폴더 하나에 stations·snapshots·forecasts·predictions.parquet와 manifest.json(형식 이름·판,
# 파일마다 SHA-256·바이트 수·행 수, 그 밖에 만든 쪽이 넘긴 정보: 데모 시각, 모델 버전, 원천 지문
# 등). 스냅샷·예보는 수집기 bronze 형식(모든 열 문자열) 그대로라 적재가 서비스와 같은 변환을 거친다.
# 예측의 hour_start도 bronze처럼 오프셋 붙은 ISO 문자열이다(시간대 이름 없이 순간을 정확히 담음).
DEMO_FORMAT = "bike-demand-demo"
DEMO_FORMAT_VERSION = 1
DEMO_MANIFEST = "manifest.json"
# 묶음 파일 이름(확장자 앞)과 열 구성. 파일은 '<이름>.parquet'이다. stations는 warehouse
# dim_stations의 일곱 열(read_warehouse_stations), predictions는 predictions 표의 키·값(created_at은
# 넣을 때 DB 시각).
DEMO_SCHEMAS = {
    "stations": pa.schema(
        [
            ("station_id", pa.string()),
            ("station_no", pa.int64()),
            ("station_name", pa.string()),
            ("district", pa.string()),
            ("lat", pa.float64()),
            ("lon", pa.float64()),
            ("docks", pa.int64()),
        ]
    ),
    "snapshots": REALTIME_SCHEMA,
    "forecasts": FORECAST_SCHEMA,
    "predictions": pa.schema(
        [
            ("station_id", pa.string()),
            ("hour_start", pa.string()),
            ("model_version", pa.string()),
            ("predicted_rentals", pa.float64()),
        ]
    ),
}
# 넣는 순서와 같은 네 서비스 테이블. 데모 적재는 이 넷이 모두 비었을 때만 한다.
DEMO_TABLES = (Station, RealtimeSnapshot, WeatherForecast, Prediction)


# 묶음 폴더나 manifest가 형식에 맞지 않을 때(없음, 다른 형식·판, 지문 불일치, 열 구성 다름). DB에
# 닿기 전에 낸다.
class DemoBundleError(ValueError):
    pass


# 데모 적재 대상 DB의 서비스 테이블에 이미 행이 있을 때. 아무것도 넣지 않은 상태에서 낸다.
class DemoLoadRefused(RuntimeError):
    pass


# read_demo_bundle의 결과: manifest 사전과 파일별 행(pyarrow to_pylist, 쓴 값 그대로).
@dataclass(frozen=True)
class DemoBundle:
    manifest: dict
    stations: list[dict]
    snapshots: list[dict]
    forecasts: list[dict]
    predictions: list[dict]


# 파일 하나의 SHA-256(16진수). 1 MiB씩 읽는다.
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 묶음을 쓴다: tables의 네 목록(키 = DEMO_SCHEMAS의 이름)을 열 구성대로 zstd Parquet으로 쓰고,
# 파일마다 SHA-256·바이트 수·행 수와 info를 담은 manifest.json을 LF·UTF-8로 쓴다. 같은 행이면 같은
# 바이트가 나온다(pyarrow 판이 같을 때). info에는 형식이 쓰는 키(format, format_version, files)를
# 넣을 수 없다. 폴더가 없으면 만들고, 있으면 그 다섯 파일만 덮어쓴다(빈 새 폴더에 쓰는 것은 부르는
# 쪽 책임).
# 호출 위치: scripts/build_demo_bundle.py, tests/test_demo.py.
def write_demo_bundle(directory: Path, tables: dict[str, list[dict]], info: dict) -> dict:
    """묶음 파일 네 개와 manifest.json을 쓰고 manifest 사전을 돌려준다."""
    reserved = {"format", "format_version", "files"} & set(info)
    if reserved:
        raise ValueError(f"info에 묶음 형식 키를 넣을 수 없음: {sorted(reserved)}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, schema in DEMO_SCHEMAS.items():
        path = directory / f"{name}.parquet"
        table = pa.Table.from_pylist(tables[name], schema=schema)
        pq.write_table(table, path, compression="zstd")
        files[path.name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "rows": table.num_rows,
        }
    manifest = {
        "format": DEMO_FORMAT,
        "format_version": DEMO_FORMAT_VERSION,
        **info,
        "files": files,
    }
    # LF로 쓴다(.gitattributes와 같은 줄 끝, Windows 기본 줄 끝은 CRLF)
    with open(directory / DEMO_MANIFEST, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


# 묶음을 읽고 검사한다: manifest의 형식 이름·판, 파일 네 개가 모두 있고 크기·SHA-256이 manifest와
# 같은지(덜 복사되거나 바뀐 파일 거부), 열 구성과 행 수가 형식과 같은지. 하나라도 다르면
# DemoBundleError. 통과한 행만 돌려주므로 load_demo는 DB를 열기 전에 묶음 전체를 확인한 셈이다.
def read_demo_bundle(directory: Path) -> DemoBundle:
    """묶음 폴더 → DemoBundle. 형식·지문·열 구성이 다르면 DemoBundleError."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / DEMO_MANIFEST).read_text("utf-8"))
    except FileNotFoundError:
        raise DemoBundleError(f"{DEMO_MANIFEST}이 없음: {directory}") from None
    except ValueError:
        raise DemoBundleError(f"{DEMO_MANIFEST}을 JSON으로 읽을 수 없음: {directory}") from None
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != DEMO_FORMAT
        or manifest.get("format_version") != DEMO_FORMAT_VERSION
        or not isinstance(manifest.get("files"), dict)
    ):
        raise DemoBundleError(
            f"{DEMO_MANIFEST}의 format이 {DEMO_FORMAT} {DEMO_FORMAT_VERSION}판이 아님: {directory}"
        )
    rows = {}
    for name, schema in DEMO_SCHEMAS.items():
        file_name = f"{name}.parquet"
        entry = manifest["files"].get(file_name)
        path = directory / file_name
        if not isinstance(entry, dict) or not path.is_file():
            raise DemoBundleError(f"{file_name}: 파일이나 manifest 항목이 없음")
        if path.stat().st_size != entry.get("bytes") or _sha256(path) != entry.get("sha256"):
            raise DemoBundleError(
                f"{file_name}: 크기·SHA-256이 manifest와 다름(바뀌었거나 덜 복사된 파일)"
            )
        table = pq.read_table(path)
        if not table.schema.equals(schema, check_metadata=False) or table.num_rows != entry.get(
            "rows"
        ):
            raise DemoBundleError(f"{file_name}: 열 구성이나 행 수가 묶음 형식과 다름")
        rows[name] = table.to_pylist()
    return DemoBundle(manifest=manifest, **rows)


# load_demo의 결과(넣은 행 수). demo_now·model_version은 manifest에 있으면 그 값(명령줄이 API를 띄울
# 데모 시각을 안내하는 데 쓴다).
class DemoLoadResult(NamedTuple):
    stations: int  # 과거 이력 대여소(source=history)
    realtime_stations: (
        int  # 스냅샷에만 있는 대여소(source=realtime, 서비스 실시간 적재와 같은 규칙)
    )
    snapshots: int  # 넣은 스냅샷 행
    skipped_rows: int  # 넣을 수 없는 값 때문에 건너뛴 스냅샷 행(realtime_rows 규칙)
    forecasts: int
    predictions: int
    demo_now: str | None = None
    model_version: str | None = None


# 묶음을 빈 서비스 DB에 넣는다. 순서: 묶음 검사(read_demo_bundle, DB 전) → 한 트랜잭션 안에서 네
# 테이블 행 수 확인 → 하나라도 행이 있으면 DemoLoadRefused(아무것도 넣지 않음) → 대여소(history) →
# 스냅샷에만 있는 대여소(realtime) → 스냅샷 → 예보 → 예측.
# 변환은 서비스 적재와 같다: 스냅샷은 realtime_rows(넣을 수 없는 값의 행만 건너뛰고 비워도 되는 값만
# 비움, 이미 있는 대여소는 실시간 값으로 덮어쓰지 않음), 예보는 forecast_rows. ON CONFLICT 없이 그냥
# 넣으므로 같은 키가 겹치면(동시에 두 번 실행 등) 오류가 나고 전부 되돌린다.
# 왜 빈 DB에만: 개발 DB(수집 자료가 있는 bike_demand)에 섞이면 데모 시각 뒤의 자료와 함께 쓰여 데모
# 화면이 그 시각의 화면이 아니게 되고, 지울 때 어느 행이 데모인지 가릴 수 없다. 테이블이 없는
# DB(alembic upgrade head 전)에서는 행 수 확인이 SQL 오류로 멈춘다.
# 주의: 확인과 넣기 사이에 다른 작업(수집기)이 같은 DB에 쓰는 것까지 막지는 않는다(잠금 없음).
# 수집을 켜지 않은 DB에서 쓴다.
def load_demo(engine: Engine, directory: Path) -> DemoLoadResult:
    """빈 DB에만 묶음을 넣고 넣은 행 수를 돌려준다. 비어 있지 않으면 DemoLoadRefused."""
    bundle = read_demo_bundle(directory)
    history = [{**{c: r[c] for c in STATION_COLUMNS}, "source": "history"} for r in bundle.stations]
    history_ids = {r["station_id"] for r in history}
    realtime = realtime_rows(bundle.snapshots)
    realtime_only = [s for s in realtime.stations if s["station_id"] not in history_ids]
    forecasts = forecast_rows(bundle.forecasts)
    predictions = [
        {
            "station_id": r["station_id"],
            "hour_start": datetime.fromisoformat(r["hour_start"]),
            "model_version": r["model_version"],
            "predicted_rentals": r["predicted_rentals"],
        }
        for r in bundle.predictions
    ]
    with engine.begin() as conn:
        counts = {
            model.__tablename__: conn.execute(
                select(func.count()).select_from(model.__table__)
            ).scalar_one()
            for model in DEMO_TABLES
        }
        filled = {name: count for name, count in counts.items() if count}
        if filled:
            listed = ", ".join(f"{name} {count:,}행" for name, count in filled.items())
            raise DemoLoadRefused(
                f"데모 묶음은 서비스 테이블 넷이 모두 빈 DB에만 넣는다. 이미 행이 있음: {listed}. "
                "빈 데이터베이스를 따로 만들어 DATABASE_URL로 가리키고 "
                "alembic upgrade head부터 한다"
            )
        # 대여소는 history와 realtime 행의 열 구성이 달라(realtime에는 district가 없음) 따로 넣는다
        for model, rows in (
            (Station, history),
            (Station, realtime_only),
            (RealtimeSnapshot, realtime.snapshots),
            (WeatherForecast, forecasts),
            (Prediction, predictions),
        ):
            for batch in _batches(rows):
                conn.execute(insert(model.__table__), batch)
    if realtime.skipped_rows or realtime.nulled_values:
        logger.warning(
            "데모 적재: 넣을 수 없는 값 때문에 스냅샷 행 %d개를 건너뛰고 값 %d개를 비움",
            realtime.skipped_rows,
            realtime.nulled_values,
        )
    return DemoLoadResult(
        stations=len(history),
        realtime_stations=len(realtime_only),
        snapshots=len(realtime.snapshots),
        skipped_rows=realtime.skipped_rows,
        forecasts=len(forecasts),
        predictions=len(predictions),
        demo_now=bundle.manifest.get("demo_now"),
        model_version=bundle.manifest.get("model_version"),
    )


# 명령줄: stations [--warehouse], realtime --day YYYY-MM-DD [--bronze], forecasts [--parquet],
# demo --bundle <폴더>. Airflow DAG도 앞의 셋을 그대로 부른다.
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
    d = sub.add_parser(
        "demo",
        help="고정 시각 데모 묶음을 빈 서비스 DB에 넣는다",
        description="고정 시각 데모 묶음(scripts/build_demo_bundle.py가 만든 폴더)을 넣는다. "
        "stations·realtime_snapshots·weather_forecasts·predictions가 모두 빈 DB에만 한 "
        "트랜잭션으로 넣고, 하나라도 행이 있으면 아무것도 넣지 않고 멈춘다. 접속할 DB는 "
        "DATABASE_URL(없으면 로컬 개발 DB). 넣은 뒤 API를 BIKE_DEMAND_DEMO_NOW와 함께 띄운다",
    )
    d.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="묶음 폴더(manifest.json과 Parquet 네 개). 파일 지문이 manifest와 다르면 넣지 않는다",
    )
    args = parser.parse_args()

    engine = make_engine()
    if args.target == "stations":
        print("stations upserted", load_stations_from_warehouse(engine, args.warehouse))
    elif args.target == "realtime":
        # 건너뛴 행 수는 넣을 수 없는 값(숫자가 아닌 자전거 수 등) 때문에 빠진 스냅샷 행이다.
        snapshots, stations, skipped = load_realtime(engine, args.bronze, args.day)
        print("snapshots inserted", snapshots, "new stations", stations, "skipped rows", skipped)
    elif args.target == "demo":
        # 데모 시계 변수 이름은 API 모듈 한 곳에 둔다(명령줄을 실행할 때만 불러온다)
        from bike_demand.api.main import DEMO_NOW_ENV

        try:
            result = load_demo(engine, args.bundle)
        except (DemoBundleError, DemoLoadRefused) as exc:
            raise SystemExit(str(exc)) from None
        print(
            "demo loaded: stations",
            result.stations,
            "realtime stations",
            result.realtime_stations,
            "snapshots",
            result.snapshots,
            "skipped rows",
            result.skipped_rows,
            "forecasts",
            result.forecasts,
            "predictions",
            result.predictions,
            "model_version",
            result.model_version,
        )
        if result.demo_now:
            print(
                f"API: {DEMO_NOW_ENV}={result.demo_now} "
                "uv run uvicorn bike_demand.api.main:app --host 127.0.0.1 --port 8000"
            )
    else:
        print("forecasts inserted", load_forecasts(engine, args.parquet))
