"""따릉이 실시간 대여정보를 10분마다 받아 서비스 DB에 넣는다.

수집과 적재를 한 작업으로 돌려, 적재 날짜를 수집 시각에서 정한다(자정 경계에서 빠지지 않게, T23).
"""

from __future__ import annotations

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

from bike_common import DEFAULT_ARGS, KST, py

with DAG(
    dag_id="collect_realtime",
    description="bikeList 스냅샷 수집 → Parquet → realtime_snapshots 적재",
    schedule="*/10 * * * *",
    start_date=pendulum.datetime(2026, 9, 16, tz=KST),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["bike-demand", "collect"],
) as dag:
    BashOperator(
        task_id="collect_and_load",
        bash_command=py("bike_demand.pipelines collect-realtime"),
        max_active_tis_per_dag=1,
    )
