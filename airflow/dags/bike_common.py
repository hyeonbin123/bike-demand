"""DAG들이 함께 쓰는 설정.

프로젝트 명령은 /opt/bike/.venv에서 저장소 루트(/opt/bike) 기준으로 돈다.
"""

# [파일 역할] 세 DAG(collect_realtime, forecast_and_predict, refresh_history)가 함께 쓰는 상수와
# 명령 조립 함수를 모아 둔 모듈이다. DAG 파일이 아니라 공통 모듈이므로 이 파일에는 DAG 객체가 없다.
# [왜 필요한가] Airflow 컨테이너 안에서 프로젝트 명령은 Airflow 자신의 파이썬이 아니라 별도
# 가상환경(/opt/bike/.venv)의 파이썬으로 돌린다(airflow/Dockerfile). 경로·환경 변수·재시도 설정을
# DAG마다 따로 적으면 어긋나기 쉬워 한곳에 둔다.
# [연결] Airflow가 dags 폴더를 import 경로에 넣으므로 DAG 파일이 `from bike_common import ...`로
# 바로 가져온다(pyproject.toml의 isort known-first-party 설명).

from __future__ import annotations

from datetime import timedelta

# pendulum: Airflow가 DAG 날짜에 쓰는 시간대 인식 datetime 라이브러리. start_date에 붙일 KST
# 시간대를 만든다.
import pendulum

# KST: 각 DAG의 start_date에 붙이는 시간대. Airflow는 start_date의 시간대로 cron 일정을 해석하므로,
# `15 2,5,8…` 같은 일정이 UTC가 아니라 한국 시각으로 돈다.
KST = pendulum.timezone("Asia/Seoul")
# PROJECT: 컨테이너 안 저장소 루트. Dockerfile이 프로젝트를 여기에 설치하고, docker-compose.yml이
# data/·src/·dbt/를 이 아래에 마운트한다. 명령의 상대 경로(data/raw/... 등 각 모듈의 기본값)가 이
# 폴더 기준이라 먼저 cd한다.
PROJECT = "/opt/bike"
# PYTHON: 프로젝트 전용 가상환경의 파이썬. Airflow 의존성과 프로젝트 의존성(LightGBM, dbt, DuckDB
# 등)이 섞이지 않도록 Dockerfile이 따로 만든 환경이다.
PYTHON = f"{PROJECT}/.venv/bin/python"

# PYTHONUTF8=1은 dbt 모델 SQL에 한글 주석이 있어 필요하다. PYTHONIOENCODING=utf-8은 한국어로 된
# print·오류 메시지가 Airflow 작업 로그에서 깨지지 않게 한다.
# 모든 명령 앞에 붙인다: 저장소 루트로 이동, 한국어 출력, 오늘 날짜(KST)
PREFIX = f"cd {PROJECT} && export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 && "

# DEFAULT_ARGS: 모든 작업의 기본값. 실패하면 2분 뒤 최대 2번 다시 시도한다. 공공 API의 일시 오류나
# 실시간 적재 잠금 충돌(pipelines.LoadBusy, docstring에 'Airflow 재시도나 다음 실행이 이어받는다'고
# 적혀 있음)을 재시도로 넘기려는 설정이다. refresh_history는 retries를 0으로 덮어쓴다.
DEFAULT_ARGS = {
    "owner": "bike-demand",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


# "모듈 이름 + 인자"를 받아 BashOperator에 넘길 셸 명령 문자열을 만든다. 예:
# py("bike_demand.model.predict --hours 48") → `cd /opt/bike && export ... &&
# /opt/bike/.venv/bin/python -m bike_demand.model.predict --hours 48`.
# 호출 위치: collect_realtime.py, forecast_and_predict.py, refresh_history.py의 모든 파이썬 작업.
def py(module_and_args: str) -> str:
    return PREFIX + f"{PYTHON} -m {module_and_args}"
