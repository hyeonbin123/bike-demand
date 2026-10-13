"""대여소·예측·부족 위험 API. 계약은 docs/api.md, 데이터는 서비스 DB(docs/serving-schema.md).

실행: uv run uvicorn bike_demand.api.main:app --port 8000
현재 시각(get_now)과 DB 엔진(get_engine)은 의존성이라 테스트에서 바꿔 끼울 수 있다.
"""

# [파일 역할] 서비스의 HTTP 입구. FastAPI 앱 하나가 다섯 개 API(/health, /stations, /stations/{id},
# /shortage-risk, /predictions/{id})와 대시보드 정적 파일을 제공한다. 데이터는 서비스용
# Postgres에서만 읽고 쓰지 않는다.
# [핵심 규칙] (1) 한 응답 안에서는 예측 버전(model_version)을 하나만 쓴다: created_at이 가장 늦은
# 묶음(latest_version). 예보 발표마다 새 묶음이 쌓여 같은 시간을 여러 버전이 예측하기 때문이다. (2)
# 시각은 모두 KST로 돌려준다. (3) DB에 접속하지 못하면 모든 요청이 503 JSON. 자세한 계약은
# docs/api.md.
# [설계 이유] 현재 시각과 DB 엔진을 FastAPI 의존성으로 주입해, 테스트가 시각을 고정하고 임시 DB를
# 끼울 수 있게 했다(tests/test_api.py의 dependency_overrides).
# [연결] api/shortage.py(부족 계산), serving/db.py(엔진), serving/models.py(테이블).
# 대시보드(static/)는 이 API만 호출한다.

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated

# FastAPI: 경로·쿼리 검증(Query의 범위를 벗어나면 422), 의존성 주입(Depends), 예외 처리기, 정적 파일
# 제공. pydantic BaseModel: 응답 형식을 선언해 docs/api.md의 필드와 타입을 코드로 고정한다.
# SQLAlchemy Core의 select로 쿼리를 만든다. OperationalError(연결 실패·끊김)와 풀 대기 시간
# 초과(TimeoutError, 내장 TimeoutError와 이름이 겹쳐 PoolTimeout으로 바꿔 부름)는 503 처리기에 쓴다.
from fastapi import Depends, FastAPI, HTTPException, Query
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


# 기본 엔진은 첫 요청 때 한 번만 만들고 그 뒤로는 같은 연결 풀을 재사용한다. 모듈을 import할 때
# 만들지 않으므로 DB가 없어도 앱을 불러올 수 있다.
@lru_cache
def _default_engine() -> Engine:
    return make_engine()


# 엔드포인트가 쓰는 엔진 의존성. 테스트는 app.dependency_overrides로 이 함수를 임시 DB 엔진이나
# 일부러 접속이 안 되는 엔진으로 바꾼다.
def get_engine() -> Engine:
    return _default_engine()


# 현재 시각 의존성. 테스트가 고정 시각을 넣어 '지금부터 N시간', '30분 안 스냅샷' 같은 경계를
# 재현한다.
def get_now() -> datetime:
    return datetime.now(KST)


# 엔드포인트 인자 타입으로 쓰는 별칭. 함수 시그니처에 Depends를 반복해 적지 않으려는 것이다.
EngineDep = Annotated[Engine, Depends(get_engine)]
NowDep = Annotated[datetime, Depends(get_now)]


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


# /health 응답: DB 상태, 마지막 스냅샷 시각, 고른 예측 버전의 마지막 예측 시간.
class Health(BaseModel):
    status: str
    database: str
    latest_snapshot_at: datetime | None
    latest_prediction_hour: datetime | None


# uvicorn이 불러 가는 ASGI 앱 객체(bike_demand.api.main:app).
app = FastAPI(title="bike-demand", version="0.1.0")


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
# null이고 상태는 ok다.
@app.get("/health", response_model=Health)
def health(engine: EngineDep) -> Health:
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
    )


# 대여소 목록. district는 자치구 이름과 정확히 같을 때만 거른다. station_id 순.
@app.get("/stations", response_model=list[StationOut])
def stations(engine: EngineDep, district: str | None = None) -> list[StationOut]:
    query = select(*STATION_COLUMNS).order_by(Station.station_id)
    if district is not None:
        query = query.where(Station.district == district)
    with engine.connect() as conn:
        return [StationOut(**row) for row in conn.execute(query).mappings()]


# 대여소 한 곳의 정보 + 가장 최근 스냅샷 + 현재 시각이 든 시간부터 6시간의 예측. 대시보드에서
# 대여소를 고르면 부르는 경로다. 없는 대여소는 404.
@app.get("/stations/{station_id}", response_model=StationDetail)
def station_detail(station_id: str, engine: EngineDep, now: NowDep) -> StationDetail:
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
    # 나이 1~180분(기본 30분).
    hours: Annotated[int, Query(ge=1, le=6)] = 3,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    district: str | None = None,
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


# 대여소 한 곳의 하루(KST 0~23시) 시간별 예측. 쿼리 이름 date가 datetime.date 타입 이름과 겹치므로
# 인자 이름은 day로 두고 alias로 받는다. 없는 대여소는 404, 그날 예측이 없으면 model_version null과
# 빈 목록.
@app.get("/predictions/{station_id}", response_model=DailyPredictions)
def daily_predictions(
    station_id: str, engine: EngineDep, day: Annotated[date, Query(alias="date")]
) -> DailyPredictions:
    # 요청 날짜의 KST 0시 ~ 다음 날 0시.
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
