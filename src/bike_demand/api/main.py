"""대여소·예측·부족 위험 API. 계약은 docs/api.md, 데이터는 서비스 DB(docs/serving-schema.md).

실행: uv run uvicorn bike_demand.api.main:app --port 8000
현재 시각(get_now)과 DB 엔진(get_engine)은 의존성이라 테스트에서 바꿔 끼울 수 있다.
환경 변수 BIKE_DEMAND_DEMO_NOW(시간대 있는 ISO 8601 시각)를 주면 '지금'을 그 시각으로 고정한다(데모
시계). 잘못된 값이면 앱이 시작하지 않는다.
"""

# [파일 역할] 서비스의 HTTP 입구. FastAPI 앱 하나가 다섯 개 API(/health, /stations, /stations/{id},
# /shortage-risk, /predictions/{id})와 대시보드 정적 파일을 제공한다. 데이터는 서비스용
# Postgres에서만 읽고 쓰지 않는다. 1 KB 이상 응답은 받는 쪽이 허용하면 gzip으로 압축해 보낸다.
# [핵심 규칙] (1) 한 응답 안에서는 예측 버전(model_version)을 하나만 쓴다: created_at이 가장 늦은
# 묶음(latest_version). 예보 발표마다 새 묶음이 쌓여 같은 시간을 여러 버전이 예측하기 때문이다. (2)
# 시각은 모두 KST로 돌려준다. (3) DB에 접속하지 못하면 모든 요청이 503 JSON. (4) 경로·쿼리 값의
# 범위는 Path·Query 선언으로 제한해, 형식은 맞지만 처리할 수 없는 값(먼 미래 날짜, NUL이 든
# 문자열)은 DB에 닿기 전에 422다. 자세한 계약은 docs/api.md.
# [설계 이유] 현재 시각과 DB 엔진을 FastAPI 의존성으로 주입해, 테스트가 시각을 고정하고 임시 DB를
# 끼울 수 있게 했다(tests/test_api.py의 dependency_overrides).
# [데모 시계] 서비스가 닫힌 뒤에는 실시간 스냅샷이 없어 /shortage-risk가 503이다. 저장해 둔 한
# 시각의 자료(serving/load.py의 demo 적재)로 그때 화면을 다시 보려면 '지금'도 그 시각이어야 하므로,
# 환경 변수가 있을 때만 get_now가 고정 시각을 돌려준다. 변수가 없으면 전과 똑같이 실제 시각이다.
# [연결] api/shortage.py(부족 계산), serving/db.py(엔진), serving/models.py(테이블).
# 대시보드(static/)는 이 API만 호출한다.

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated

# FastAPI: 경로·쿼리 검증(Path·Query의 범위를 벗어나면 422. fastapi.Path는 pathlib.Path와 이름이
# 겹쳐 PathParam으로 바꿔 부름), 의존성 주입(Depends), 예외 처리기, 정적 파일 제공, 응답
# 압축(GZipMiddleware, Starlette 것). pydantic BaseModel: 응답 형식을 선언해 docs/api.md의 필드와
# 타입을 코드로 고정한다.
# SQLAlchemy Core의 select로 쿼리를 만든다. OperationalError(연결 실패·끊김)와 풀 대기 시간
# 초과(TimeoutError, 내장 TimeoutError와 이름이 겹쳐 PoolTimeout으로 바꿔 부름)는 503 처리기에 쓴다.
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Engine, and_, func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeout

# 프로젝트 모듈: 부족 계산 순수 함수, 엔진 생성(세션 시간대 KST, 접속 제한 5초), ORM 테이블 정의.
from bike_demand.api.shortage import expected_rentals
from bike_demand.serving.db import make_engine
from bike_demand.serving.models import Prediction, RealtimeSnapshot, Station

logger = logging.getLogger(__name__)

# KST: 응답 시각을 +09:00으로 맞추는 고정 오프셋. DETAIL_HOURS: 대여소 상세에서 보여 줄 앞으로의
# 시간 수(docs/api.md '앞으로 6시간'). NOTE: 부족 대수가 반납을 반영하지 않는다는 한계를 응답마다
# 함께 보낸다.
KST = timezone(timedelta(hours=9))
DETAIL_HOURS = 6
NOTE = "반납은 반영하지 않은 값"

# 경로·쿼리 값의 범위. 핸들러 안의 if문이 아니라 Path·Query 선언으로 두어, FastAPI가 DB를 조회하기
# 전에 422로 거르고(docs/api.md 공통) /openapi.json에도 같은 제약이 드러나게 한다.
# - STATION_ID_PATTERN: 대여소ID는 영문자·숫자·하이픈 1~20자. stations.station_id는 varchar(20)에
#   'ST-xxx' 형식이고(docs/serving-schema.md) 저장된 값은 모두 7자 이하다. NUL(0x00)이 든 문자열은
#   PostgreSQL text에 들어갈 수 없어 psycopg가 DataError를 내는데, 이는 503 처리기가 잡는
#   OperationalError가 아니라 500이 됐다. 선언으로 막으면 DB에 닿지 않는다
# - DISTRICT_MAX_LENGTH·DISTRICT_PATTERN: 자치구 필터는 50자 이하(stations.district varchar(50)),
#   제어 문자(U+0000~U+001F) 없음. 저장된 자치구 이름 25개는 모두 4자 이하라 실제 값은 영향이 없고,
#   빈 문자열은 전처럼 통과한다(어느 자치구와도 같지 않아 빈 목록)
# - DATE_MIN·DATE_MAX: /predictions의 date는 2000-01-01~2100-12-31. 자료는 2023년부터고, 범위 없이
#   받으면 date.max(9999-12-31)의 '다음 날 0시'(start + 1일)를 만들 수 없어 OverflowError로 500이
#   났다
STATION_ID_PATTERN = r"^[A-Za-z0-9-]{1,20}$"
DISTRICT_MAX_LENGTH = 50
DISTRICT_PATTERN = r"^[^\x00-\x1f]*$"
DATE_MIN = date(2000, 1, 1)
DATE_MAX = date(2100, 12, 31)

# 위 범위를 붙인 인자 타입 별칭. 같은 범위를 여러 엔드포인트가 똑같이 쓴다(대여소ID는
# /stations/{id}·/predictions/{id}, 자치구는 /stations·/shortage-risk).
StationIdPath = Annotated[str, PathParam(pattern=STATION_ID_PATTERN)]
DistrictQuery = Annotated[
    str | None, Query(max_length=DISTRICT_MAX_LENGTH, pattern=DISTRICT_PATTERN)
]
# 쿼리 이름 date가 datetime.date 타입 이름과 겹치므로 인자 이름은 day로 두고 alias로 받는다.
DateQuery = Annotated[date, Query(alias="date", ge=DATE_MIN, le=DATE_MAX)]


# 기본 엔진은 첫 요청 때 한 번만 만들고 그 뒤로는 같은 연결 풀을 재사용한다. 모듈을 import할 때
# 만들지 않으므로 DB가 없어도 앱을 불러올 수 있다.
@lru_cache
def _default_engine() -> Engine:
    return make_engine()


# 엔드포인트가 쓰는 엔진 의존성. 테스트는 app.dependency_overrides로 이 함수를 임시 DB 엔진이나
# 일부러 접속이 안 되는 엔진으로 바꾼다.
def get_engine() -> Engine:
    return _default_engine()


# 데모 시계를 켜는 환경 변수 이름. 값은 시간대가 있는 ISO 8601 시각(예: 2026-09-19T18:11:00+09:00,
# UTC면 끝에 Z)이다.
DEMO_NOW_ENV = "BIKE_DEMAND_DEMO_NOW"


# 데모 시계 값이 시각으로 읽히지 않거나 시간대가 없을 때의 예외. ValueError의 하위형이라 일반 값
# 오류로도 잡힌다. 앱 시작(lifespan)에서 이 예외가 나면 uvicorn은 'Application startup failed'로
# 끝난다.
class DemoClockError(ValueError):
    pass


# BIKE_DEMAND_DEMO_NOW 값을 KST 시각으로 바꾼다. 없거나 빈 값(공백만 포함)이면 None(데모 시계 꺼짐).
# docker compose의 '${VAR:-}'처럼 변수는 있지만 비어 있는 경우를 꺼짐으로 보려는 것이다.
# 시간대가 없는 값(날짜만 준 값 포함)은 오류다: 서버 시간대로 짐작하면 컨테이너(UTC)와 PC(KST)에서
# 9시간 다른 화면이 나온다. KST 날짜는 DATE_MIN~DATE_MAX 안이어야 한다(/predictions의 date 범위와
# 같음): 그 밖의 극단 값(예: 9999-12-31)은 응답 계산(지금 + 6시간 등)에서 datetime 범위를 넘어 500이
# 되므로 시작 때 막는다. 값은 설정이라 비밀이 아니므로 오류 메시지에 그대로(앞 80자) 넣는다.
# 호출 위치: get_demo_now, scripts/build_demo_bundle.py(--demo-now), tests/test_demo.py.
def parse_demo_now(raw: str | None) -> datetime | None:
    """데모 시계 값(시간대 있는 ISO 8601) → KST datetime.

    없거나 비면 None(꺼짐), 잘못된 값이면 DemoClockError.
    """
    if raw is None or not raw.strip():
        return None
    text_value = raw.strip()
    try:
        value = datetime.fromisoformat(text_value)
        # 시간대가 없으면 astimezone이 PC 시간대로 짐작하므로 바꾸지 않고 오류로 보낸다. 1년 1월 1일
        # 같은 끝 값은 KST로 옮길 때 OverflowError가 난다
        local = value.astimezone(KST) if value.utcoffset() is not None else None
    except (ValueError, OverflowError):
        local = None
    if local is None or not DATE_MIN <= local.date() <= DATE_MAX:
        raise DemoClockError(
            f"{DEMO_NOW_ENV}는 시간대가 있는 ISO 8601 시각({DATE_MIN}~{DATE_MAX} KST)이어야 함"
            f"(예: 2026-09-19T18:11:00+09:00): {text_value[:80]!r}"
        )
    return local


# 데모 시계 의존성: 환경 변수를 요청마다 읽어 고정 시각 또는 None을 돌려준다(읽기는 마이크로초
# 단위라 캐시하지 않는다. 테스트가 monkeypatch로 변수를 바꾸면 바로 반영된다). /health가 이 값을
# 그대로 낸다.
def get_demo_now() -> datetime | None:
    return parse_demo_now(os.environ.get(DEMO_NOW_ENV))


# 현재 시각 의존성. 테스트가 고정 시각을 넣어 '지금부터 N시간', '30분 안 스냅샷' 같은 경계를
# 재현한다. 데모 시계가 켜져 있으면 그 시각이고 흐르지 않는다(모든 요청이 같은 순간을 본다).
# 주의: 데모 시계는 '지금'만 바꾼다. 최근 스냅샷·예측 버전을 고르는 쿼리에는 '지금 이전' 상한이
# 없어, 그 시각 뒤의 자료가 든 DB(예: 수집을 계속한 개발 DB)에서는 그 뒤 자료가 골라진다. 그래서 그
# 시각까지의 자료만 넣은 빈 DB(serving/load.py demo)와 함께 쓴다(docs/api.md '데모 시계').
def get_now() -> datetime:
    demo = get_demo_now()
    return demo if demo is not None else datetime.now(KST)


# 엔드포인트 인자 타입으로 쓰는 별칭. 함수 시그니처에 Depends를 반복해 적지 않으려는 것이다.
EngineDep = Annotated[Engine, Depends(get_engine)]
NowDep = Annotated[datetime, Depends(get_now)]
DemoNowDep = Annotated[datetime | None, Depends(get_demo_now)]


# 대여소 한 곳의 응답 형식(/stations 목록, /stations/{id}의 station). 실시간 스냅샷에만 있는
# 대여소는 번호 외 정보가 비어 있을 수 있어 대부분 필드가 None을 허용한다.
class StationOut(BaseModel):
    station_id: str
    station_no: int | None
    station_name: str | None
    district: str | None
    lat: float | None
    lon: float | None
    docks: int | None


# 가장 최근 실시간 스냅샷(수집 시각, 자전거 수, 거치대 수).
class SnapshotOut(BaseModel):
    fetched_at: datetime
    bike_count: int
    rack_count: int | None


# 한 시간의 예측(정시 시작 시각, 예상 대여 수).
class HourlyPrediction(BaseModel):
    hour_start: datetime
    predicted_rentals: float


# /stations/{id} 응답: 대여소 정보 + 최근 스냅샷 + 앞으로 6시간 예측 + 고른 예측 버전.
class StationDetail(BaseModel):
    station: StationOut
    snapshot: SnapshotOut | None
    predictions: list[HourlyPrediction]
    model_version: str | None


# 부족 위험 목록의 한 줄. as_of는 그 대여소 스냅샷 시각이고, shortfall = expected_rentals -
# bike_count다.
class ShortageStation(BaseModel):
    station_id: str
    station_name: str | None
    district: str | None
    lat: float | None
    lon: float | None
    as_of: datetime
    bike_count: int
    expected_rentals: float
    shortfall: float


# /shortage-risk 응답 전체. 고른 예측 버전, 응답을 만든 시각, 반납 미반영 안내(note)를 목록과 함께
# 보낸다.
class ShortageRisk(BaseModel):
    hours: int
    model_version: str
    generated_at: datetime
    note: str
    stations: list[ShortageStation]


# /predictions/{id}?date= 응답: 하루(KST 0~23시)의 시간별 예측.
class DailyPredictions(BaseModel):
    station_id: str
    date: date
    model_version: str | None
    hours: list[HourlyPrediction]


# /health 응답: DB 상태, 마지막 스냅샷 시각, 고른 예측 버전의 마지막 예측 시간, 데모 시계(켜져
# 있으면 고정 시각, 아니면 None). 대시보드가 demo_now로 데모 안내를 띄우고 스냅샷 나이를 그 시각
# 기준으로 잰다.
class Health(BaseModel):
    status: str
    database: str
    latest_snapshot_at: datetime | None
    latest_prediction_hour: datetime | None
    demo_now: datetime | None


# 앱 시작·종료 처리(lifespan). 시작할 때 데모 시계 값을 한 번 검사해, 잘못된 값이면 요청을 받기 전에
# 서버가 멈추게 한다(요청마다 500이 나는 대신 설정 실수가 바로 드러남). 켜져 있으면 실시간 서비스로
# 착각하지 않게 고정 시각을 WARNING으로 남긴다. 종료 때 할 일은 없다.
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    demo = get_demo_now()
    if demo is not None:
        logger.warning(
            "데모 시계 켜짐: 모든 '지금'을 %s로 고정 (%s)", demo.isoformat(), DEMO_NOW_ENV
        )
    yield


# uvicorn이 불러 가는 ASGI 앱 객체(bike_demand.api.main:app).
app = FastAPI(title="bike-demand", version="0.1.0", lifespan=lifespan)

# 응답 압축. 대시보드가 받는 /stations 전체(대여소 약 3,000곳, JSON 수백 KB)와
# /shortage-risk?limit=500, 정적 JS가 크다. 요청의 Accept-Encoding에 gzip이 있을 때만 1 KB 이상
# 응답을 압축하고(작은 JSON·오류 응답은 그대로) Vary: Accept-Encoding을 붙인다. 본문 내용은 바뀌지
# 않으므로 API 계약(docs/api.md)은 그대로다. 압축 수준은 Starlette 기본값.
app.add_middleware(GZipMiddleware, minimum_size=1024)


# 앱 전체 DB 장애 처리기. 처음에는 /health만 DB 오류를 잡아 나머지 경로는 DB가 멈추면 글자로 된
# 500을 냈고, 대시보드가 장애와 버그를 구분하지 못했다(커밋 112e278). 지금은 모든 경로가 같은 503
# JSON을 낸다. 풀 대기 시간 초과는 커밋 3f2ed20에서 더했다.
@app.exception_handler(OperationalError)
@app.exception_handler(PoolTimeout)
async def database_unavailable(_request, exc: OperationalError | PoolTimeout) -> JSONResponse:
    # DB에 접속하지 못하면(연결 실패, 연결 풀 대기 시간 초과) /health를 포함한 모든 요청이 503
    # (docs/api.md). 쿼리 오류(ProgrammingError 등)는 장애로 가리지 않고 500으로 둔다.
    # 비밀번호·DB 이름 같은 설정 오류도 OperationalError로 오므로 원인은 로그에 남긴다
    logger.error("database unavailable: %s", getattr(exc, "orig", None) or exc)
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


# 대시보드의 HTML·JS·CSS를 API와 같은 서버가 /static 아래로 제공한다. 별도 빌드나 프런트엔드 서버가
# 필요 없고, 같은 출처라 대시보드가 API를 바로 부를 수 있다.
STATIC_DIR = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# 루트 주소는 대시보드 HTML을 돌려준다. API 문서(/docs)에는 드러내지
# 않는다(include_in_schema=False).
@app.get("/", response_class=FileResponse, include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# StationOut 필드와 같은 순서의 컬럼 묶음. 목록과 상세 조회가 같은 컬럼을 고르게 한다.
STATION_COLUMNS = (
    Station.station_id,
    Station.station_no,
    Station.station_name,
    Station.district,
    Station.lat,
    Station.lon,
    Station.docks,
)


# DB에서 읽은 timestamptz나 의존성으로 받은 현재 시각을 KST(+09:00)로 바꾼다. 응답 시각을 모두 KST
# ISO 문자열로 내는 계약(docs/api.md 공통)을 지키는 곳이다.
def _kst(value: datetime) -> datetime:
    return value.astimezone(KST)


# 쓸 예측 버전 하나를 고른다. where를 주면 그 조건(예: 특정 날짜의 시간)에 맞는 행 중에서 고른다.
# 대여소·시간 구간마다 따로 고르지 않고 전체에서 하나만 고르는 것이 계약이다(docs/api.md).
# (created_at, model_version) 인덱스(alembic 0002)가 이 정렬 + LIMIT 1을 빠르게 한다. 호출 위치:
# health, station_detail, shortage_risk, daily_predictions.
def latest_version(conn, where=None) -> str | None:
    """docs/api.md: created_at이 가장 늦은 행의 버전, 같으면 버전 문자열이 큰 것."""
    query = select(Prediction.model_version).order_by(
        Prediction.created_at.desc(), Prediction.model_version.desc()
    )
    if where is not None:
        query = query.where(where)
    return conn.execute(query.limit(1)).scalar_one_or_none()


# 서비스 상태 확인. select 1로 DB 접속을 확인하고, 마지막 스냅샷 시각과 '고른 예측 버전의 마지막
# 시간'을 돌려준다. 대시보드가 마지막 수집 시각을 여기서 가져온다. 스냅샷·예측이 없으면 해당 값만
# null이고 상태는 ok다. demo_now는 데모 시계(꺼져 있으면 null)로, DB와 상관없는 서버 설정 값이다.
@app.get("/health", response_model=Health)
def health(engine: EngineDep, demo_now: DemoNowDep) -> Health:
    # DB 접속 실패는 다른 경로처럼 공통 처리기(database_unavailable)가 503으로 답하고 로그를 남긴다
    with engine.connect() as conn:
        conn.execute(text("select 1"))
        snapshot = conn.execute(select(func.max(RealtimeSnapshot.fetched_at))).scalar_one()
        version = latest_version(conn)
        last_hour = None
        if version is not None:
            last_hour = conn.execute(
                select(func.max(Prediction.hour_start)).where(Prediction.model_version == version)
            ).scalar_one()
    return Health(
        status="ok",
        database="ok",
        latest_snapshot_at=_kst(snapshot) if snapshot else None,
        latest_prediction_hour=_kst(last_hour) if last_hour else None,
        demo_now=demo_now,
    )


# 대여소 목록. district는 자치구 이름과 정확히 같을 때만 거른다(범위는 DistrictQuery). station_id
# 순.
@app.get("/stations", response_model=list[StationOut])
def stations(engine: EngineDep, district: DistrictQuery = None) -> list[StationOut]:
    query = select(*STATION_COLUMNS).order_by(Station.station_id)
    if district is not None:
        query = query.where(Station.district == district)
    with engine.connect() as conn:
        return [StationOut(**row) for row in conn.execute(query).mappings()]


# 대여소 한 곳의 정보 + 가장 최근 스냅샷 + 현재 시각이 든 시간부터 6시간의 예측. 대시보드에서
# 대여소를 고르면 부르는 경로다. 없는 대여소는 404(대여소ID 형식이 StationIdPath 밖이면 그 전에
# 422).
@app.get("/stations/{station_id}", response_model=StationDetail)
def station_detail(station_id: StationIdPath, engine: EngineDep, now: NowDep) -> StationDetail:
    # 현재 시각이 든 시간의 정시. 예: 08:20이면 08시부터 6시간.
    first = _kst(now).replace(minute=0, second=0, microsecond=0)
    with engine.connect() as conn:
        row = (
            conn.execute(select(*STATION_COLUMNS).where(Station.station_id == station_id))
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise HTTPException(404, "station not found")
        # 이 대여소의 가장 최근 스냅샷 한 건(나이 제한 없음). 없으면 None.
        snapshot = (
            conn.execute(
                select(
                    RealtimeSnapshot.fetched_at,
                    RealtimeSnapshot.bike_count,
                    RealtimeSnapshot.rack_count,
                )
                .where(RealtimeSnapshot.station_id == station_id)
                .order_by(RealtimeSnapshot.fetched_at.desc())
                .limit(1)
            )
            .mappings()
            .one_or_none()
        )
        # 버전은 전체 기준으로 고르고, 그 버전에 이 대여소의 시간이 빠져 있으면 그 시간은 목록에서
        # 빠진다(옛 버전으로 채우지 않음).
        version = latest_version(conn)
        hours = []
        if version is not None:
            hours = conn.execute(
                select(Prediction.hour_start, Prediction.predicted_rentals)
                .where(
                    Prediction.model_version == version,
                    Prediction.station_id == station_id,
                    Prediction.hour_start >= first,
                    Prediction.hour_start < first + timedelta(hours=DETAIL_HOURS),
                )
                .order_by(Prediction.hour_start)
            ).all()
    # 예측은 소수 둘째 자리로 반올림해 보낸다(docs/api.md 숫자 반올림).
    return StationDetail(
        station=StationOut(**row),
        snapshot=SnapshotOut(**{**snapshot, "fetched_at": _kst(snapshot["fetched_at"])})
        if snapshot
        else None,
        predictions=[
            HourlyPrediction(hour_start=_kst(h), predicted_rentals=round(v, 2)) for h, v in hours
        ],
        model_version=version,
    )


# 서비스의 핵심 응답: 앞으로 hours시간 예상 대여가 지금 자전거 수보다 많은 대여소를 부족 대수 큰
# 순으로 돌려준다.
# 계산(대여소마다): as_of = 나이 제한 안의 가장 최근 스냅샷 시각, expected = [as_of, as_of + hours)
# 예측 합(일부 걸친 시간은 비율), shortfall = expected - bike_count. 반납은 예측하지 않으므로
# '반납이 없을 때 모자랄 수 있는 대수'다.
# 503은 필터 전 전체 데이터 기준으로만 낸다: 예측 표가 비었을 때(먼저 확인), 모든 대여소를 통틀어
# 최근 스냅샷이 없을 때. 그 밖에 남는 대여소가 없으면 200과 빈 목록.
@app.get("/shortage-risk", response_model=ShortageRisk)
def shortage_risk(
    engine: EngineDep,
    now: NowDep,
    # 쿼리 범위는 Query로 검증하고, 벗어나면 FastAPI가 422를 낸다. hours 1~6, limit 1~500, 스냅샷
    # 나이 1~180분(기본 30분), 자치구는 DistrictQuery(50자 이하, 제어 문자 없음).
    hours: Annotated[int, Query(ge=1, le=6)] = 3,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    district: DistrictQuery = None,
    max_snapshot_age_minutes: Annotated[int, Query(ge=1, le=180)] = 30,
) -> ShortageRisk:
    now = _kst(now)
    # 이보다 오래된 스냅샷만 있는 대여소는 '지금 자전거 수'를 모르는 것으로 보고 뺀다.
    cutoff = now - timedelta(minutes=max_snapshot_age_minutes)
    with engine.connect() as conn:
        version = latest_version(conn)
        if version is None:
            raise HTTPException(503, "no predictions")
        # 대여소마다 cutoff 이후 가장 최근 스냅샷 시각을 구하는 하위 쿼리. 대여소마다 as_of가 다를
        # 수 있다.
        latest = (
            select(
                RealtimeSnapshot.station_id,
                func.max(RealtimeSnapshot.fetched_at).label("fetched_at"),
            )
            .where(RealtimeSnapshot.fetched_at >= cutoff)
            .group_by(RealtimeSnapshot.station_id)
            .subquery()
        )
        # 그 시각의 스냅샷 행(자전거 수)과 대여소 정보(이름·자치구·좌표)를 붙인다. stations와 inner
        # join이라 대여소 표에 없는 스냅샷은 빠진다.
        recent = (
            conn.execute(
                select(
                    Station.station_id,
                    Station.station_name,
                    Station.district,
                    Station.lat,
                    Station.lon,
                    RealtimeSnapshot.fetched_at,
                    RealtimeSnapshot.bike_count,
                )
                .select_from(latest)
                .join(
                    RealtimeSnapshot,
                    and_(
                        RealtimeSnapshot.station_id == latest.c.station_id,
                        RealtimeSnapshot.fetched_at == latest.c.fetched_at,
                    ),
                )
                .join(Station, Station.station_id == latest.c.station_id)
            )
            .mappings()
            .all()
        )
        # 자치구 필터를 적용하기 전에 판단한다. 필터 때문에 비는 것은 장애가 아니므로 200 빈 목록이
        # 된다.
        if not recent:
            raise HTTPException(503, "no recent snapshot")
        candidates = [r for r in recent if district is None or r["district"] == district]
        predictions: dict[str, dict[datetime, float]] = {}
        # 후보 대여소들의 예측을 한 번의 쿼리로 읽는다. 시간 범위는 가장 이른 스냅샷이 든 시간부터
        # 가장 늦은 스냅샷 + hours까지라 모든 대여소의 구간을 덮는다. 고른 버전 하나만 읽어 버전을
        # 섞지 않는다.
        if candidates:
            earliest = min(r["fetched_at"] for r in candidates)
            first_hour = _kst(earliest).replace(minute=0, second=0, microsecond=0)
            latest_end = max(r["fetched_at"] for r in candidates) + timedelta(hours=hours)
            rows = conn.execute(
                select(
                    Prediction.station_id, Prediction.hour_start, Prediction.predicted_rentals
                ).where(
                    Prediction.model_version == version,
                    Prediction.station_id.in_([r["station_id"] for r in candidates]),
                    Prediction.hour_start >= first_hour,
                    Prediction.hour_start < latest_end,
                )
            ).all()
            # 대여소별 {정시 시작(KST): 예측} 사전으로 바꾼다. expected_rentals가 이 사전의 키로
            # 시간을 찾는다.
            for station_id, hour_start, value in rows:
                predictions.setdefault(station_id, {})[_kst(hour_start)] = value

    # 필요한 시간의 예측이 하나라도 없으면(None) 그 대여소는 뺀다. 부족 대수가 0 이하인 곳도 뺀다.
    results = []
    for r in candidates:
        as_of = _kst(r["fetched_at"])
        expected = expected_rentals(as_of, hours, predictions.get(r["station_id"], {}))
        if expected is None:
            continue
        shortfall = expected - r["bike_count"]
        if shortfall <= 0:
            continue
        results.append((shortfall, r, as_of, expected))
    # 부족 대수 큰 순, 같으면 station_id 순. 정렬·비교는 반올림 전 값으로 하고(docs/api.md), limit은
    # 정렬 뒤에 자른다.
    results.sort(key=lambda item: (-item[0], item[1]["station_id"]))

    # 응답 숫자는 예상 대여·부족 대수 모두 소수 첫째 자리로 반올림한다.
    return ShortageRisk(
        hours=hours,
        model_version=version,
        generated_at=now,
        note=NOTE,
        stations=[
            ShortageStation(
                station_id=r["station_id"],
                station_name=r["station_name"],
                district=r["district"],
                lat=r["lat"],
                lon=r["lon"],
                as_of=as_of,
                bike_count=r["bike_count"],
                expected_rentals=round(expected, 1),
                shortfall=round(shortfall, 1),
            )
            for shortfall, r, as_of, expected in results[:limit]
        ],
    )


# 대여소 한 곳의 하루(KST 0~23시) 시간별 예측. 날짜는 DateQuery(쿼리 이름 date, 2000~2100년),
# 대여소ID는 StationIdPath 범위 안이어야 한다. 없는 대여소는 404, 그날 예측이 없으면 model_version
# null과 빈 목록.
@app.get("/predictions/{station_id}", response_model=DailyPredictions)
def daily_predictions(
    station_id: StationIdPath, engine: EngineDep, day: DateQuery
) -> DailyPredictions:
    # 요청 날짜의 KST 0시 ~ 다음 날 0시. day가 DATE_MAX 이하라 다음 날 계산이 넘치지 않는다.
    start = datetime(day.year, day.month, day.day, tzinfo=KST)
    end = start + timedelta(days=1)
    with engine.connect() as conn:
        exists = conn.execute(
            select(Station.station_id).where(Station.station_id == station_id)
        ).scalar_one_or_none()
        if exists is None:
            raise HTTPException(404, "station not found")
        # 버전은 대여소와 상관없이 그날 행이 있는 버전 중에서 고른다(docs/api.md, T34)
        in_day = and_(Prediction.hour_start >= start, Prediction.hour_start < end)
        version = latest_version(conn, in_day)
        rows = []
        if version is not None:
            rows = conn.execute(
                select(Prediction.hour_start, Prediction.predicted_rentals)
                .where(
                    in_day,
                    Prediction.model_version == version,
                    Prediction.station_id == station_id,
                )
                .order_by(Prediction.hour_start)
            ).all()
    return DailyPredictions(
        station_id=station_id,
        date=day,
        model_version=version,
        hours=[
            HourlyPrediction(hour_start=_kst(h), predicted_rentals=round(v, 2)) for h, v in rows
        ],
    )
