"""따릉이 실시간 대여정보를 10분마다 받아 서비스 DB에 넣는다.

수집과 적재를 한 작업으로 돌려, 적재 날짜를 수집 시각에서 정한다(자정 경계에서 빠지지 않게, T23).
"""

# [파일 역할] 서비스의 '지금 자전거 수'를 공급하는 수집 DAG다. 10분마다 서울 열린데이터광장 실시간
# 대여정보(bikeList)를 받아 원본 JSON → 그날 Parquet → Postgres `realtime_snapshots`까지 한 번에
# 넣는다.
# [설계 이유] 처음에는 수집과 적재가 따로 된 작업이었는데, 수집은 자기 실행 날짜 폴더에 쓰고 적재는
# 자기 시작 날짜 폴더를 읽어서 23:59에 받은 스냅샷이 자정 뒤 적재에서 빠질 수 있었다(Codex 검토
# 라운드 3, 커밋 d19a3e8). 그래서 `pipelines collect-realtime` 한 명령이 수집 시각으로 날짜를 정해
# 수집·재생성·적재를 모두 한다.
# [연결] 실제 일은 src/bike_demand/pipelines.py의 collect_realtime이 한다(덜 적재된 최근 7일 회수
# 포함). /shortage-risk와 대시보드가 이 표의 최신 스냅샷을 읽는다.

from __future__ import annotations

# Airflow 3의 DAG(airflow.sdk)와 BashOperator(standard provider)를 쓴다. 파이썬 함수를 직접 부르는
# PythonOperator 대신 BashOperator를 쓰는 것은 프로젝트 코드를 Airflow와 다른 가상환경에서 돌리기
# 위해서다(bike_common.py 참고).
import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# DAG들이 함께 쓰는 기본 재시도 설정, KST 시간대, 명령 조립 함수.
from bike_common import DEFAULT_ARGS, KST, py

# 일정: 10분마다(cron 분 칸이 10분 간격, KST).
# catchup=False: 꺼져 있던 동안의 지난 실행을 몰아서 돌리지 않는다. 실시간 API는 지금 상태만 주므로
# 지난 시각의 스냅샷은 소급 실행으로도 되살릴 수 없다(README 한계).
# max_active_runs=1: 한 번에 한 실행만 돌게 해 같은 날 Parquet을 동시에 다시 만드는 일을 줄인다(겹친
# 경우는 pipelines의 잠금이 LoadBusy로 막는다).
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
    # 작업 하나: 수집 → 그날 Parquet 재생성 → 적재(+ 덜 적재된 최근 7일 회수).
    # max_active_tis_per_dag=1: 같은 작업 인스턴스가 동시에 둘 이상 돌지 않게 한다. 적재 행 수를
    # 적재 전후 행 수 차이로 세므로 같은 표에 적재를 동시에 돌리지 않는다는
    # 전제다(docs/serving-schema.md 적재 규칙).
    BashOperator(
        task_id="collect_and_load",
        bash_command=py("bike_demand.pipelines collect-realtime"),
        max_active_tis_per_dag=1,
    )
