"""단기예보 발표(02·05·08·11·14·17·20·23시) 15분 뒤: 예보 수집 → 적재 → 앞으로 48시간 예측."""

# [파일 역할] 기상청 단기예보가 발표될 때마다 예보를 받아 앞으로 48시간의 대여소별 시간당 예상 대여
# 수를 새로 만드는 DAG다. /shortage-risk가 쓰는 `predictions`를 채우는 유일한 경로다.
# [왜 발표 15분 뒤인가] 단기예보는 02·05·08·11·14·17·20·23시에 발표되지만 API로 받을 수 있게 되는
# 것은 약 10~15분 뒤다(docs/experiments.md v4 측정 전 보충). 그래서 발표 시각 + 15분에 돈다. 이
# 시각은 v4·v5 측정의 기준 시각 as_of(발표 + 15분)의 근거이기도 하다.
# [흐름] fetch_forecast(예보 JSON 저장) → load_forecast(Postgres weather_forecasts) →
# predict(CURRENT 서비스 모델로 48시간 예측, predictions에 새 model_version으로 저장).

from __future__ import annotations

# Airflow 3 DAG와 BashOperator. 프로젝트 명령은 별도 가상환경의 파이썬으로 돌린다(bike_common.py).
import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# 공통 재시도 설정, KST 시간대, 명령 조립 함수.
from bike_common import DEFAULT_ARGS, KST, py

# catchup=False: PC가 꺼져 있던 동안의 실행을 몰아서 돌리지 않는다. 대신 빠진 예보 발표는 아래 fetch
# 작업의 --catch-up이 하루 안쪽까지 채운다.
# max_active_runs=1: 예측 작업이 겹쳐 같은 시각에 두 묶음을 만드는 일을 막는다.
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
    # 예보 수집. --catch-up은 가장 최근 발표와 그 앞 일곱 발표 중 아직 없는 것을 받는다. 2026-09-17
    # 밤 PC가 꺼져 23·02·05시 발표를 못 받은 일이 있었고, API가 전날 발표까지 주므로 다음 실행에서
    # 채우도록 한 것이다(커밋 e3c788a). 이미 있는 파일은 API를 부르지 않고 건너뛰며, 옛 발표 실패는
    # 보고만 하고 최신 발표 실패만 작업 실패로 친다.
    fetch = BashOperator(
        task_id="fetch_forecast", bash_command=py("bike_demand.ingest.forecast --catch-up")
    )
    # 예보 JSON을 weather_forecasts에 적재. 같은 키는 건너뛰어(ON CONFLICT DO NOTHING) 다시 돌려도
    # 행이 늘지 않는다. 행 수 차이로 적재 수를 세므로 동시 실행은 1로 묶는다.
    load = BashOperator(
        task_id="load_forecast",
        bash_command=py("bike_demand.serving.load forecasts"),
        max_active_tis_per_dag=1,
    )
    # DB의 가장 최근 발표 예보로 앞으로 48시간(단기예보가 덮는 범위 안)을 예측해 predictions에
    # 저장한다. model_version은 '모델 세대 이름 + 예보 발표 시각'이라 발표마다 새 묶음이 쌓이고,
    # API는 가장 최근 묶음 하나만 쓴다(docs/api.md).
    predict = BashOperator(
        task_id="predict",
        bash_command=py("bike_demand.model.predict --hours 48"),
        max_active_tis_per_dag=1,
    )
    # 순서 의존: 예보가 적재된 뒤에만 예측한다. 앞 작업이 (재시도 후에도) 실패하면 뒤 작업은 돌지
    # 않아 옛 예보로 새 묶음을 만들지 않는다.
    fetch >> load >> predict
