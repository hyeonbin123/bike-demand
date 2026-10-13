"""반기마다 새 대여이력이 공개되면 수동으로 실행한다.

미리 새 원본 파일을 data/raw/trips, data/raw/stations에 넣어 둔다. 변환은 월 단위로 통째로 바꾸고
dbt는 전체를 다시 만들므로 여러 번 실행해도 중복되지 않는다.
대여이력 전체 dbt 빌드는 수십 분 걸린다.
날씨 수집의 끝 달과 달력(dim_hours)의 끝 날은 원본 중 가장 최근 달에서 정해 같은 값을 넘긴다(T32).
모델 재학습(bike_demand.model.final)은 측정 규칙을 거쳐야 하므로 이 DAG에 넣지 않는다.
"""

# [파일 역할] 반기마다 공개되는 따릉이 대여이력과 대여소 정보를 warehouse(DuckDB)와 서비스 DB에
# 반영하는 수동 DAG다.
# [흐름] (대여이력 변환, 대여소 정보 변환, ASOS 날씨 수집)을 동시에 → dbt build로 staging·marts 전체
# 재생성 → Postgres stations 갱신.
# [설계 이유] 처음에는 날씨와 달력(dim_hours)이 기본 끝(2026-06)에 머물러 새 반기 대여가 날씨·달력
# 행 없이 특징 결합에서 빠질 수 있었다(Codex 검토 라운드 3, 커밋 229e7c6). 그래서 원본 중 가장 최근
# 달을 읽어 날씨 수집 끝 달과 dbt 달력 끝 날에 같은 값을 넘긴다. 이 값을 넘기지 않으면 dbt 자료
# 테스트 assert_calendar_covers_rentals가 실패한다(커밋 29a4094).
# [하지 않는 일] 모델 재학습은 넣지 않는다. 새 반기를 넣은 뒤 `model.final serving`을 사람이 따로
# 돌린다(README 서비스 DB와 Airflow 절).

from __future__ import annotations

# Airflow 3 DAG와 BashOperator. dbt도 프로젝트 가상환경 안의 실행 파일로 부른다.
import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# dbt 명령은 파이썬 모듈이 아니라 py()를 쓸 수 없어서 PREFIX·PROJECT·PYTHON을 직접 가져와 조립한다.
from bike_common import DEFAULT_ARGS, KST, PREFIX, PROJECT, PYTHON, py

# LATEST_MONTH / LATEST_DAY: 셸 명령 치환 `$(...)` 문자열이다. DAG를 읽을 때가 아니라 작업이 실행될
# 때 컨테이너 안에서 평가되므로, 그 순간 data/raw/trips에 있는 가장 최근 달(YYYY-MM)과 그 달
# 말일(YYYY-MM-DD)이 들어간다.
LATEST_MONTH = f"$({PYTHON} -m bike_demand.pipelines latest-trip-month)"
LATEST_DAY = f"$({PYTHON} -m bike_demand.pipelines latest-trip-month --last-day)"
# DBT: 저장소 루트에서 dbt/ 프로젝트를 빌드하는 명령(모델 + 테스트). 프로필도 dbt/profiles.yml을
# 쓴다.
DBT = f"{PROJECT}/.venv/bin/dbt build --project-dir dbt --profiles-dir dbt"

# schedule=None: 일정 없이 사람이 새 원본 파일을 넣은 뒤 직접 실행한다.
# retries 0: 기본 설정(2번 재시도)을 덮어써 실패하면 바로 멈춘다. 대여이력 전체 dbt 빌드는 수십 분
# 걸리는 수동 작업이라, 실패 원인을 보고 다시 실행하는 쪽을 택한 설정이다.
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
    # 대여이력 ZIP·CSV → 월별 문자열 Parquet(bronze). 달 단위로 파일을 통째로 바꾸므로 다시 돌려도
    # 중복되지 않는다.
    trips = BashOperator(task_id="convert_trips", bash_command=py("bike_demand.ingest.trips"))
    # 대여소 정보 반기 xlsx → bronze Parquet.
    stations = BashOperator(
        task_id="convert_stations", bash_command=py("bike_demand.ingest.stations")
    )
    # 기상청 ASOS 서울(108) 시간자료를 원본 최신 달까지 받는다(--end LATEST_MONTH). 대여 기간과 날씨
    # 기간의 끝을 맞추는 것이 목적이다.
    weather = BashOperator(
        task_id="fetch_asos",
        bash_command=py(f"bike_demand.ingest.weather --end {LATEST_MONTH}"),
    )
    # dbt build에 calendar_end(원본 최신 달 말일)를 var로 넘겨 달력과 대여소×시간 격자를 그날까지
    # 늘린다. f-string 안의 겹친 중괄호는 dbt var 문법의 YAML 사전(calendar_end: 'YYYY-MM-DD')을
    # 만들기 위한 이스케이프다.
    dbt_build = BashOperator(
        task_id="dbt_build",
        bash_command=PREFIX + DBT + f" --vars \"{{calendar_end: '{LATEST_DAY}'}}\"",
    )
    # dbt가 새로 만든 dim_stations를 서비스 DB stations에 upsert한다. 실시간에만 있던 대여소는
    # history 값으로 바뀐다(docs/serving-schema.md).
    load_stations = BashOperator(
        task_id="load_stations", bash_command=py("bike_demand.serving.load stations")
    )
    # 세 입력 작업은 서로 의존하지 않아 동시에 돌고, 셋 다 끝나야 dbt가 돈다.
    [trips, stations, weather] >> dbt_build >> load_stations
