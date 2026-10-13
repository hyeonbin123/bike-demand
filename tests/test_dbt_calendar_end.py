"""작은 합성 bronze로 dbt build 전체를 돌려 달력 끝(calendar_end) 경로를 확인한다.

refresh_history DAG는 새 반기를 넣은 뒤
`dbt build --vars "{calendar_end: <원본 최신 달 말일>}"`을 돌린다.
실제 원본·warehouse는 건드리지 않는다: 자료, warehouse, profiles, target, 로그를 모두
임시 폴더에 둔다.
dbt build 한 번에 20~30초 걸린다.
"""

# [무엇을 검증하나] 반기 갱신 경로의 두 가지 실패를 실제 dbt build로 막는다. (1) refresh_history처럼
# calendar_end를 새 반기 끝(2026-12-31)으로 넘기면 단위 테스트까지 포함한 build가 끝까지 통과하고 새
# 반기 대여가 격자에 들어간다. (2) calendar_end를 빠뜨리면 새 반기가 조용히 잘리지 않고
# assert_calendar_covers_rentals 하나만 실패한다.
# [왜 중요한가] 고치기 전에는 격자 단위 테스트가 기본 calendar_end에 묶여 있어 새 반기 build가
# 실패하고 격자가 옛 표로 남았고(ed06760), var 없이 빌드하면 달력과 격자가 함께 잘려 새 반기가
# 사라져도 아무 검사도 실패하지 않았다(29a4094).
# [방식] dbt 모델·테스트는 저장소의 것을 그대로 쓰되, 입력 bronze·warehouse·profiles·target·로그를
# 모두 pytest 임시 폴더에 둔다. 실제 원본과 data/warehouse는 건드리지 않는다.

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# duckdb: build 결과 warehouse를 읽어 확인. pyarrow: ingest 모듈과 같은 스키마(모든 열 문자열)의
# 합성 bronze Parquet을 쓴다.
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# 각 ingest 모듈의 SCHEMA를 그대로 써서 합성 bronze가 실제 변환 결과와 같은 열을 갖게 한다(staging
# 모델이 읽는 열 이름이 어긋나지 않게).
from bike_demand.ingest import stations, trips, weather

# 저장소의 dbt 프로젝트 폴더. dbt 실행 파일은 PATH가 아니라 지금 pytest를 돌리는 가상환경에서 찾아,
# 다른 설치본의 dbt가 잡히지 않게 한다.
PROJECT_DIR = Path(__file__).resolve().parents[1] / "dbt"
DBT = shutil.which("dbt", path=str(Path(sys.executable).parent))  # 같은 가상환경의 dbt


# 대여 한 건짜리 bronze 행(ST-1에서 빌려 ST-2에 반납). 나머지 열은 None.
def _trip(rented_at: str, returned_at: str) -> dict:
    row = dict.fromkeys(trips.SCHEMA.names)
    row.update(
        bike_no="B1",
        rented_at=rented_at,
        returned_at=returned_at,
        rent_station_no="1",
        return_station_no="2",
        duration_min="10",
        distance_m="100",
        rent_station_id="ST-1",
        return_station_id="ST-2",
        source_file="synthetic",
    )
    return row


# 주어진 스키마로 Parquet 파일 하나를 쓴다(빈 목록이면 열만 있는 빈 파일).
def _write(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


# 두 테스트가 같은 bronze를 읽기만 하므로 모듈 단위로 한 번 만든다. 2026-06-30 밤 대여는 기존 달력
# 마지막 날, 2026-12-15 대여는 '새로 변환한 반기'를 흉내 낸다. 대여소·날씨는 열만 있는 빈 파일로
# 둔다(이 테스트는 대여의 달력 경계만 본다).
@pytest.fixture(scope="module")
def new_half_year(tmp_path_factory):
    """2026 상반기 마지막 날 대여 하나와 새로 변환한 하반기(2026-12) 대여 하나가 있는 bronze."""
    data = tmp_path_factory.mktemp("dbt-data")
    bronze = data / "bronze"
    _write(
        bronze / "trips" / "month=2026-06" / "trips.parquet",
        [_trip("2026-06-30 22:10:00", "2026-06-30 22:30:00")],
        trips.SCHEMA,
    )
    _write(
        bronze / "trips" / "month=2026-12" / "trips.parquet",
        [_trip("2026-12-15 08:10:00", "2026-12-15 08:20:00")],
        trips.SCHEMA,
    )
    _write(bronze / "stations" / "stations.parquet", [], stations.SCHEMA)
    _write(bronze / "weather" / "asos_hourly.parquet", [], weather.SCHEMA)
    return data


# 임시 profile·target·log 경로로 dbt build를 subprocess로 돌린다. data_dir var로 합성 bronze 폴더를
# 가리키고, 추가 var(calendar_end 등)는 그대로 넘긴다. 돌려주는 상태 사전은 run_results.json의 노드
# 이름 → success/pass/fail 등.
def dbt_build(data: Path, run: Path, **dbt_vars) -> tuple[subprocess.CompletedProcess, dict]:
    """run 폴더의 새 warehouse에 dbt build를 돌리고 (프로세스 결과, 노드별 상태)를 돌려준다."""
    warehouse = run / "bike_demand.duckdb"
    # 저장소의 profiles.yml은 환경 변수가 없으면 실제 warehouse를 가리키므로 임시 profile을 쓴다
    (run / "profiles.yml").write_text(
        "bike_demand:\n  target: dev\n  outputs:\n    dev:\n      type: duckdb\n"
        f'      path: "{warehouse.as_posix()}"\n      threads: 4\n',
        encoding="utf-8",
    )
    # --target-path는 dbt/ 기준, --log-path는 현재 폴더 기준이라 둘 다 절대 경로로 준다
    command = [
        DBT, "build",
        "--project-dir", str(PROJECT_DIR),
        "--profiles-dir", str(run),
        "--target-path", str(run / "target"),
        "--log-path", str(run / "logs"),
        "--vars", json.dumps({"data_dir": data.as_posix(), **dbt_vars}),
    ]  # fmt: skip
    # 모델 SQL에 한글 주석이 있어 Windows에서는 PYTHONUTF8=1이 필요하다. 테스트에서 dbt 사용 통계를
    # 외부로 보내지 않는다.
    env = {**os.environ, "PYTHONUTF8": "1", "DBT_SEND_ANONYMOUS_USAGE_STATS": "false"}
    proc = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8")
    results = run / "target" / "run_results.json"
    status = {}
    if results.exists():  # 파싱에서 멈추면 없다
        for node in json.loads(results.read_text(encoding="utf-8"))["results"]:
            status[node["unique_id"].split(".")[-1]] = node["status"]
    return proc, status


# 통과가 아닌 노드만 골라 실패 메시지를 읽기 쉽게 한다.
def _not_ok(status: dict) -> dict:
    return {name: s for name, s in status.items() if s not in ("success", "pass", "warn")}


def test_dag_build_with_a_later_calendar_end_passes(new_half_year, tmp_path):
    """refresh_history처럼 calendar_end를 새 반기 끝으로 넘겨도 build가 끝까지 통과한다.

    격자 단위 테스트가 기본 calendar_end(2026-06-30)에 묶여 있으면 여기서 실패하고
    격자는 옛 표로 남는다.
    """
    # 격자 단위 테스트(grid_stops_at_the_calendar_end)는 overrides로 자기 calendar_end(2026-06-30)를
    # 고정해 두었으므로, 바깥 var가 바뀌어도 통과해야 한다. 달력이 12-31 23시까지 늘고, 12월 대여가
    # 격자에 1건으로 들어갔는지 warehouse를 열어 확인한다.
    proc, status = dbt_build(new_half_year, tmp_path, calendar_end="2026-12-31")
    assert proc.returncode == 0, (_not_ok(status), proc.stdout[-3000:])
    assert status["grid_stops_at_the_calendar_end"] == "pass"
    with duckdb.connect(str(tmp_path / "bike_demand.duckdb"), read_only=True) as db:
        last_hour = db.sql("select max(hour_start) from dim_hours").fetchone()[0]
        december = db.sql(
            "select rentals from int_station_hour_grid "
            "where station_id = 'ST-1' and hour_start = timestamp '2026-12-15 08:00:00'"
        ).fetchone()
    assert last_hour == datetime(2026, 12, 31, 23)
    assert december == (1,)


def test_build_without_the_new_calendar_end_fails_instead_of_dropping_the_half_year(
    new_half_year, tmp_path
):
    """새 반기를 넣고 calendar_end 없이(기본 2026-06-30) 빌드하면 조용히 잘리지 않고 실패한다.

    달력과 격자가 같은 경계로 함께 잘려 assert_grid_within_calendar로는 잡히지 않는다.
    """
    # var 없이(기본 2026-06-30) 빌드하면 12월 대여가 달력 밖이 된다. 실패한 노드가 정확히
    # assert_calendar_covers_rentals 하나이고 나머지 노드는 모두 통과해야 한다.
    proc, status = dbt_build(new_half_year, tmp_path)
    assert proc.returncode != 0, proc.stdout[-3000:]
    assert _not_ok(status) == {"assert_calendar_covers_rentals": "fail"}
