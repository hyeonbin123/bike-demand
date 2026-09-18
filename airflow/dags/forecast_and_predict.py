"""단기예보 발표(02·05·08·11·14·17·20·23시) 15분 뒤: 예보 수집 → 적재 → 앞으로 48시간 예측."""

from __future__ import annotations

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

from bike_common import DEFAULT_ARGS, KST, py

with DAG(
    dag_id="forecast_and_predict",
    description="단기예보 수집 → weather_forecasts 적재 → predictions 저장",
    schedule="15 2,5,8,11,14,17,20,23 * * *",
    start_date=pendulum.datetime(2026, 9, 16, tz=KST),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["bike-demand", "collect", "predict"],
) as dag:
    fetch = BashOperator(
        task_id="fetch_forecast", bash_command=py("bike_demand.ingest.forecast --catch-up")
    )
    load = BashOperator(
        task_id="load_forecast",
        bash_command=py("bike_demand.serving.load forecasts"),
        max_active_tis_per_dag=1,
    )
    predict = BashOperator(
        task_id="predict",
        bash_command=py("bike_demand.model.predict --hours 48"),
        max_active_tis_per_dag=1,
    )
    fetch >> load >> predict
