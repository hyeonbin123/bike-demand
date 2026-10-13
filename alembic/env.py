"""Alembic 환경. 접속 문자열은 bike_demand.serving.db.database_url()에서 가져온다.

다른 접속 문자열(예: 테스트 DB)을 쓰려면 Config에 ``sqlalchemy.url``을 넣어 부른다.
"""

# [파일 역할] `uv run alembic upgrade head`가 실행하는 Alembic 환경 스크립트다. 어느 DB에 어떤
# 방식(온라인/오프라인)으로 마이그레이션을 적용할지 정한다.
# [설계 이유] 접속 문자열을 alembic.ini에 따로 적지 않고 애플리케이션과 같은
# 함수(serving.db.database_url)에서 가져와, 앱·적재·마이그레이션이 늘 같은 DB를 보게 한다. 테스트는
# 임시 DB 주소를 Config의 sqlalchemy.url로 넣어 같은 마이그레이션을 그 DB에
# 적용한다(tests/conftest.py, tests/test_serving_load.py).
# [흐름] 로깅 설정 → 대상 메타데이터 지정 → 오프라인이면 SQL만 만들고, 아니면 실제 DB에 접속해 적용.

from logging.config import fileConfig

# alembic.context: 지금 실행 중인 마이그레이션의 설정·모드에 접근하는 객체. sqlalchemy.pool:
# 마이그레이션용 엔진에 연결 풀을 쓰지 않으려고(NullPool) 가져온다.
from alembic import context
from sqlalchemy import pool

# 프로젝트 쪽: 접속 문자열과 엔진 생성 함수(세션 시간대 KST, 접속 제한 시간 5초가 들어 있음), 그리고
# ORM 테이블 정의(Base). 마이그레이션 파일과 models.py의 정의가 같은 스키마를 가리키도록
# 메타데이터를 연결한다.
from bike_demand.serving.db import database_url, make_engine
from bike_demand.serving.models import Base

# alembic.ini의 로깅 설정을 적용한다. 단, 호출하는 쪽이 attributes에 configure_logger=False를 넣으면
# 건너뛴다. 테스트가 이렇게 부르는데, fileConfig는 프로세스 전체 로거 설정을 바꾸므로 pytest의
# 로깅을 건드리지 않으려는 것이다.
config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

# autogenerate(모델과 DB 차이로 마이그레이션 초안 만들기)가 비교할 기준 메타데이터.
target_metadata = Base.metadata


# 접속 문자열 우선순위: Config에 직접 넣은 sqlalchemy.url(테스트의 임시 DB) → 환경 변수 DATABASE_URL
# → 로컬 개발 기본값(127.0.0.1:55452).
def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or database_url()


# 오프라인 모드(`alembic upgrade head --sql`): DB에 접속하지 않고 실행할 SQL 문만 출력한다.
# literal_binds=True는 바인드 변수 자리에 실제 값을 글자로 넣어 그대로 실행 가능한 스크립트를
# 만든다.
def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


# 온라인 모드(기본): 실제 DB에 접속해 마이그레이션을 한 트랜잭션 안에서 적용한다.
# NullPool: 한 번 쓰고 끝나는 작업이라 연결을 풀에 남겨 둘 필요가 없다. 끝나면 dispose로 연결을
# 확실히 닫아, 테스트가 곧바로 임시 DB를 지울 때 남은 연결이 방해하지 않게 한다.
def run_migrations_online() -> None:
    engine = make_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


# alembic 명령이 이 스크립트를 불러올 때 모드에 맞는 함수 하나를 실행한다.
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
