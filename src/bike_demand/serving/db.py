"""서비스용 PostgreSQL 연결. 접속 문자열은 환경 변수 DATABASE_URL, 없으면 로컬 개발 기본값."""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import Engine, create_engine

DEFAULT_DATABASE_URL = "postgresql+psycopg://bike:bike@127.0.0.1:55452/bike_demand"


def database_url() -> str:
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def make_engine(url: str | None = None, **kwargs: Any) -> Engine:
    # 세션 시간대를 KST로 두어 timestamptz를 읽을 때 +09:00으로 돌려받는다.
    return create_engine(
        url or database_url(),
        pool_pre_ping=True,
        connect_args={"options": "-c timezone=Asia/Seoul"},
        **kwargs,
    )
