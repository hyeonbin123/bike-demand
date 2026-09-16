"""서비스 DB 테스트용 임시 PostgreSQL 데이터베이스.

`docker compose up -d db`로 띄운 서버에 테스트마다 임시 DB를 만들고 마이그레이션을 적용한 뒤 지운다.
개발 DB(bike_demand)의 데이터는 건드리지 않는다. 서버에 접속할 수 없으면 해당 테스트는 건너뛴다.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from bike_demand.serving.db import database_url, make_engine


@pytest.fixture
def pg_engine():
    base = make_url(database_url())
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    name = f"bike_demand_test_{uuid.uuid4().hex[:12]}"
    try:
        with admin.connect() as conn:
            conn.execute(text(f'create database "{name}"'))
    except Exception as exc:  # noqa: BLE001 - 접속 실패 종류와 상관없이 건너뜀
        admin.dispose()
        pytest.skip(f"PostgreSQL에 접속할 수 없음 (docker compose up -d db): {type(exc).__name__}")

    url = base.set(database=name).render_as_string(hide_password=False)
    from alembic import command
    from alembic.config import Config

    config = Config("alembic.ini", attributes={"configure_logger": False})
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(config, "head")
    engine = make_engine(url)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'drop database if exists "{name}" with (force)'))
        admin.dispose()
