"""작은 합성 bronze로 dbt build 전체를 돌려 달력 끝(calendar_end) 경로를 확인한다.

refresh_history DAG는 새 반기를 넣은 뒤
`dbt build --vars "{calendar_end: <원본 최신 달 말일>}"`을 돌린다.
실제 원본·warehouse는 건드리지 않는다: 자료, warehouse, profiles, target, 로그를 모두
임시 폴더에 둔다.
dbt build 한 번에 20~30초 걸린다.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bike_demand.ingest import stations, trips, weather

PROJECT_DIR = Path(__file__).resolve().parents[1] / "dbt"
DBT = shutil.which("dbt", path=str(Path(sys.executable).parent))  # 같은 가상환경의 dbt


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


def _write(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


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
    env = {**os.environ, "PYTHONUTF8": "1", "DBT_SEND_ANONYMOUS_USAGE_STATS": "false"}
    proc = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8")
    results = run / "target" / "run_results.json"
    status = {}
    if results.exists():  # 파싱에서 멈추면 없다
        for node in json.loads(results.read_text(encoding="utf-8"))["results"]:
            status[node["unique_id"].split(".")[-1]] = node["status"]
    return proc, status


def _not_ok(status: dict) -> dict:
    return {name: s for name, s in status.items() if s not in ("success", "pass", "warn")}


def test_dag_build_with_a_later_calendar_end_passes(new_half_year, tmp_path):
    """refresh_history처럼 calendar_end를 새 반기 끝으로 넘겨도 build가 끝까지 통과한다.

    격자 단위 테스트가 기본 calendar_end(2026-06-30)에 묶여 있으면 여기서 실패하고
    격자는 옛 표로 남는다.
    """
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
    proc, status = dbt_build(new_half_year, tmp_path)
    assert proc.returncode != 0, proc.stdout[-3000:]
    assert _not_ok(status) == {"assert_calendar_covers_rentals": "fail"}
