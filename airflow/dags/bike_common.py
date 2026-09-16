"""DAG들이 함께 쓰는 설정.

프로젝트 명령은 /opt/bike/.venv에서 저장소 루트(/opt/bike) 기준으로 돈다.
"""

from __future__ import annotations

from datetime import timedelta

import pendulum

KST = pendulum.timezone("Asia/Seoul")
PROJECT = "/opt/bike"
PYTHON = f"{PROJECT}/.venv/bin/python"

# 모든 명령 앞에 붙인다: 저장소 루트로 이동, 한국어 출력, 오늘 날짜(KST)
PREFIX = f"cd {PROJECT} && export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 && "

DEFAULT_ARGS = {
    "owner": "bike-demand",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def py(module_and_args: str) -> str:
    return PREFIX + f"{PYTHON} -m {module_and_args}"
