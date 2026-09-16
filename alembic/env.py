"""Alembic 환경. 접속 문자열은 bike_demand.serving.db.database_url()에서 가져온다.

다른 접속 문자열(예: 테스트 DB)을 쓰려면 Config에 ``sqlalchemy.url``을 넣어 부른다.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool

from bike_demand.serving.db import database_url, make_engine
from bike_demand.serving.models import Base

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or database_url()


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
