# Airflow 이미지에 이 프로젝트를 별도 가상환경(/opt/bike/.venv)으로 설치한다.
# Airflow 자체 의존성과 섞이지 않게 하려는 것이고, DAG는 BashOperator로 그 가상환경의 명령을 부른다.
#
# 이미지 판은 공개 권고가 고쳐진 판으로 둔다(2026-11-24 갱신). Airflow 3.3.1에는 PYSEC-2026-3988
# (/assets/events가 Dag별 읽기 권한을 거르지 않음), PYSEC-2026-3989(쿠키와 Bearer 토큰이 함께 오면
# 쿠키 사용자로 처리), PYSEC-2026-3990(로그아웃이 Bearer 토큰을 폐기하지 않음)이 있었고 3.3.2에서
# 고쳐졌다. 3.3.1 → 3.3.2 릴리스 노트에는 이 DAG들이 쓰는 airflow.sdk.DAG·BashOperator의
# 인터페이스를 바꾼 항목이 없다. 판을 올릴 때는 태그가 레지스트리에 있는지 먼저 본다(이미지를
# 받지 않고 메타데이터만 읽는다): `docker buildx imagetools inspect apache/airflow:<판>-python3.11`
FROM apache/airflow:3.3.2-python3.11

USER root
# LightGBM이 쓰는 OpenMP 런타임
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# uv 바이너리는 uv 이미지에서 가져온다. 0.12.12에는 PYSEC-2026-4321(uv에 내장된 인증서 목록이 중간
# 인증서 개인키가 유출된 emSign 루트를 신뢰함, 0.13.4에서 고침)이 있어 0.13.6으로 올렸다
# (2026-11-24). 이 태그도 `docker buildx imagetools inspect ghcr.io/astral-sh/uv:<판>`으로 있는지
# 먼저 확인한다.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
RUN mkdir -p /opt/bike && chown airflow:0 /opt/bike
USER airflow

WORKDIR /opt/bike
ENV UV_PROJECT_ENVIRONMENT=/opt/bike/.venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
# Airflow 이미지는 VIRTUAL_ENV를 자기 환경으로 잡아 두므로 uv를 부를 때만 뺀다
COPY --chown=airflow:0 pyproject.toml uv.lock README.md ./
RUN env -u VIRTUAL_ENV uv sync --frozen --no-dev --no-install-project --python python3.11
COPY --chown=airflow:0 src ./src
COPY --chown=airflow:0 dbt ./dbt
RUN env -u VIRTUAL_ENV uv sync --frozen --no-dev --python python3.11

WORKDIR /opt/airflow
