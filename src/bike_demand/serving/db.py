"""서비스용 PostgreSQL 연결. 접속 문자열은 환경 변수 DATABASE_URL, 없으면 로컬 개발 기본값."""

# [파일 역할] 서비스 DB(PostgreSQL) 연결을 만드는 한 곳. API(api/main.py), 적재(serving/load.py,
# pipelines.py), 예측(model/predict.py), 측정 스크립트, alembic 마이그레이션, 테스트가 모두 이
# 함수로 엔진을 만든다.
# [왜 한곳에 두나] 접속 주소·세션 시간대·연결 시간 제한을 여기서만 정하면, 프로그램마다 시간대가
# 달라 시각이 어긋나거나 DB 장애 때 한쪽만 오래 멈추는 일이 없다.

from __future__ import annotations

import os
from typing import Any

# SQLAlchemy 엔진(연결 풀)을 만든다. 드라이버는 접속 문자열의 postgresql+psycopg(psycopg 3)이다.
from sqlalchemy import Engine, create_engine

# 로컬 개발 기본값. 호스트 포트 55452는 docker-compose.yml의 db 서비스 포트다(Windows가 5432 부근을
# 예약하는 경우가 있고 다른 프로젝트가 55432·55442를 써서 피함). 'localhost' 대신 127.0.0.1을 쓰는
# 것은 Windows에서 localhost가 IPv6를 먼저 시도해 몇 초씩 멈추기 때문이다. Airflow 컨테이너 안에서는
# DATABASE_URL 환경 변수로 db 서비스 주소를 넘긴다.
DEFAULT_DATABASE_URL = "postgresql+psycopg://bike:bike@127.0.0.1:55452/bike_demand"


# 환경 변수 DATABASE_URL이 있으면 그것, 없으면 로컬 기본값. alembic/env.py와 tests/conftest.py도 이
# 함수를 써서 앱·적재·마이그레이션·테스트가 같은 DB를 가리키게 한다.
def database_url() -> str:
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


# 엔진을 만든다. url을 주면 그 주소(테스트의 임시 DB 등), 없으면 database_url(). kwargs는
# create_engine에 그대로 넘긴다(alembic은 poolclass=NullPool을 넘김).
# pool_pre_ping=True: 풀에서 꺼낸 연결이 죽어 있으면(DB 재시작 등) 쓰기 전에 확인하고 새로 연결한다.
# 오래 떠 있는 API·Airflow 작업이 끊긴 연결로 실패하지 않게 한다.
def make_engine(url: str | None = None, **kwargs: Any) -> Engine:
    # 세션 시간대를 KST로 두어 timestamptz를 읽을 때 +09:00으로 돌려받는다.
    return create_engine(
        url or database_url(),
        pool_pre_ping=True,
        # DB가 죽었을 때 요청이 운영체제 기본 제한(수십 초 이상)까지 멈추지 않게 한다
        connect_args={"options": "-c timezone=Asia/Seoul", "connect_timeout": 5},
        **kwargs,
    )
