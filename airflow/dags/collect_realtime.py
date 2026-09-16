"""따릉이 실시간 대여정보를 10분마다 받아 서비스 DB에 넣는다."""

from __future__ import annotations

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

from bike_common import DEFAULT_ARGS, KST, py

with DAG(
    dag_id="collect_realtime",
    description="bikeList 스냅샷 수집 → realtime_snapshots 적재",
    schedule="*/10 * * * *",
    start_date=pendulum.datetime(2026, 9, 16, tz=KST),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["bike-demand", "collect"],
) as dag:
    fetch = BashOperator(task_id="fetch_snapshot", bash_command=py("bike_demand.ingest.realtime"))
    load = BashOperator(
        task_id="load_snapshot",
        bash_command=py("bike_demand.serving.load realtime --day $(TZ=Asia/Seoul date +%F)"),
        max_active_tis_per_dag=1,
    )
    fetch >> load
