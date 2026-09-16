# Airflow 이미지에 이 프로젝트를 별도 가상환경(/opt/bike/.venv)으로 설치한다.
# Airflow 자체 의존성과 섞이지 않게 하려는 것이고, DAG는 BashOperator로 그 가상환경의 명령을 부른다.
FROM apache/airflow:3.3.1-python3.11

USER root
# LightGBM이 쓰는 OpenMP 런타임
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /usr/local/bin/uv
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
