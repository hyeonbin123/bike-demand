"""반기마다 새 대여이력이 공개되면 수동으로 실행한다.

미리 새 원본 파일을 data/raw/trips, data/raw/stations에 넣어 둔다. 변환은 월 단위로 통째로 바꾸고
dbt는 전체를 다시 만들므로 여러 번 실행해도 중복되지 않는다.
대여이력 전체 dbt 빌드는 수십 분 걸린다.
날씨 수집의 끝 달과 달력(dim_hours)의 끝 날은 원본 중 가장 최근 달에서 정해 같은 값을 넘긴다(T32).
모델 재학습(bike_demand.model.final)은 측정 규칙을 거쳐야 하므로 이 DAG에 넣지 않는다.
"""

from __future__ import annotations

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

from bike_common import DEFAULT_ARGS, KST, PREFIX, PROJECT, PYTHON, py

LATEST_MONTH = f"$({PYTHON} -m bike_demand.pipelines latest-trip-month)"
LATEST_DAY = f"$({PYTHON} -m bike_demand.pipelines latest-trip-month --last-day)"
DBT = f"{PROJECT}/.venv/bin/dbt build --project-dir dbt --profiles-dir dbt"

with DAG(
    dag_id="refresh_history",
    description="대여이력·대여소 정보 적재 → dbt build → 서비스 DB 대여소 갱신 (수동)",
    schedule=None,
    start_date=pendulum.datetime(2026, 9, 16, tz=KST),
    catchup=False,
    max_active_runs=1,
    default_args={**DEFAULT_ARGS, "retries": 0},
    tags=["bike-demand", "history"],
) as dag:
    trips = BashOperator(task_id="convert_trips", bash_command=py("bike_demand.ingest.trips"))
    stations = BashOperator(
        task_id="convert_stations", bash_command=py("bike_demand.ingest.stations")
    )
    weather = BashOperator(
        task_id="fetch_asos",
        bash_command=py(f"bike_demand.ingest.weather --end {LATEST_MONTH}"),
    )
    dbt_build = BashOperator(
        task_id="dbt_build",
        bash_command=PREFIX + DBT + f" --vars \"{{calendar_end: '{LATEST_DAY}'}}\"",
    )
    load_stations = BashOperator(
        task_id="load_stations", bash_command=py("bike_demand.serving.load stations")
    )
    [trips, stations, weather] >> dbt_build >> load_stations
