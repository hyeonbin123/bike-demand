"""대여소·예측·부족 위험 API. 계약은 docs/api.md, 데이터는 서비스 DB(docs/serving-schema.md).

실행: uv run uvicorn bike_demand.api.main:app --port 8000
현재 시각(get_now)과 DB 엔진(get_engine)은 의존성이라 테스트에서 바꿔 끼울 수 있다.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Engine, and_, func, select, text
from sqlalchemy.exc import SQLAlchemyError

from bike_demand.api.shortage import expected_rentals
from bike_demand.serving.db import make_engine
from bike_demand.serving.models import Prediction, RealtimeSnapshot, Station

KST = timezone(timedelta(hours=9))
DETAIL_HOURS = 6
NOTE = "반납은 반영하지 않은 값"


@lru_cache
def _default_engine() -> Engine:
    return make_engine()


def get_engine() -> Engine:
    return _default_engine()


def get_now() -> datetime:
    return datetime.now(KST)


EngineDep = Annotated[Engine, Depends(get_engine)]
NowDep = Annotated[datetime, Depends(get_now)]


class StationOut(BaseModel):
    station_id: str
    station_no: int | None
    station_name: str | None
    district: str | None
    lat: float | None
    lon: float | None
    docks: int | None


class SnapshotOut(BaseModel):
    fetched_at: datetime
    bike_count: int
    rack_count: int | None


class HourlyPrediction(BaseModel):
    hour_start: datetime
    predicted_rentals: float


class StationDetail(BaseModel):
    station: StationOut
    snapshot: SnapshotOut | None
    predictions: list[HourlyPrediction]
    model_version: str | None


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


class ShortageRisk(BaseModel):
    hours: int
    model_version: str
    generated_at: datetime
    note: str
    stations: list[ShortageStation]


class DailyPredictions(BaseModel):
    station_id: str
    date: date
    model_version: str | None
    hours: list[HourlyPrediction]


class Health(BaseModel):
    status: str
    database: str
    latest_snapshot_at: datetime | None
    latest_prediction_hour: datetime | None


app = FastAPI(title="bike-demand", version="0.1.0")
STATIC_DIR = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse, include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


STATION_COLUMNS = (
    Station.station_id,
    Station.station_no,
    Station.station_name,
    Station.district,
    Station.lat,
    Station.lon,
    Station.docks,
)


def _kst(value: datetime) -> datetime:
    return value.astimezone(KST)


def latest_version(conn, where=None) -> str | None:
    """docs/api.md: created_at이 가장 늦은 행의 버전, 같으면 버전 문자열이 큰 것."""
    query = select(Prediction.model_version).order_by(
        Prediction.created_at.desc(), Prediction.model_version.desc()
    )
    if where is not None:
        query = query.where(where)
    return conn.execute(query.limit(1)).scalar_one_or_none()


@app.get("/health", response_model=Health)
def health(engine: EngineDep) -> Health:
    try:
        with engine.connect() as conn:
            conn.execute(text("select 1"))
            snapshot = conn.execute(select(func.max(RealtimeSnapshot.fetched_at))).scalar_one()
            version = latest_version(conn)
            last_hour = None
            if version is not None:
                last_hour = conn.execute(
                    select(func.max(Prediction.hour_start)).where(
                        Prediction.model_version == version
                    )
                ).scalar_one()
    except SQLAlchemyError:
        raise HTTPException(503, "database unavailable") from None
    return Health(
        status="ok",
        database="ok",
        latest_snapshot_at=_kst(snapshot) if snapshot else None,
        latest_prediction_hour=_kst(last_hour) if last_hour else None,
    )


@app.get("/stations", response_model=list[StationOut])
def stations(engine: EngineDep, district: str | None = None) -> list[StationOut]:
    query = select(*STATION_COLUMNS).order_by(Station.station_id)
    if district is not None:
        query = query.where(Station.district == district)
    with engine.connect() as conn:
        return [StationOut(**row) for row in conn.execute(query).mappings()]


@app.get("/stations/{station_id}", response_model=StationDetail)
def station_detail(station_id: str, engine: EngineDep, now: NowDep) -> StationDetail:
    first = _kst(now).replace(minute=0, second=0, microsecond=0)
    with engine.connect() as conn:
        row = (
            conn.execute(select(*STATION_COLUMNS).where(Station.station_id == station_id))
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise HTTPException(404, "station not found")
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


@app.get("/shortage-risk", response_model=ShortageRisk)
def shortage_risk(
    engine: EngineDep,
    now: NowDep,
    hours: Annotated[int, Query(ge=1, le=6)] = 3,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    district: str | None = None,
    max_snapshot_age_minutes: Annotated[int, Query(ge=1, le=180)] = 30,
) -> ShortageRisk:
    now = _kst(now)
    cutoff = now - timedelta(minutes=max_snapshot_age_minutes)
    with engine.connect() as conn:
        version = latest_version(conn)
        if version is None:
            raise HTTPException(503, "no predictions")
        latest = (
            select(
                RealtimeSnapshot.station_id,
                func.max(RealtimeSnapshot.fetched_at).label("fetched_at"),
            )
            .where(RealtimeSnapshot.fetched_at >= cutoff)
            .group_by(RealtimeSnapshot.station_id)
            .subquery()
        )
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
        if not recent:
            raise HTTPException(503, "no recent snapshot")
        candidates = [r for r in recent if district is None or r["district"] == district]
        predictions: dict[str, dict[datetime, float]] = {}
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
            for station_id, hour_start, value in rows:
                predictions.setdefault(station_id, {})[_kst(hour_start)] = value

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
    results.sort(key=lambda item: (-item[0], item[1]["station_id"]))

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


@app.get("/predictions/{station_id}", response_model=DailyPredictions)
def daily_predictions(
    station_id: str, engine: EngineDep, day: Annotated[date, Query(alias="date")]
) -> DailyPredictions:
    start = datetime(day.year, day.month, day.day, tzinfo=KST)
    end = start + timedelta(days=1)
    with engine.connect() as conn:
        exists = conn.execute(
            select(Station.station_id).where(Station.station_id == station_id)
        ).scalar_one_or_none()
        if exists is None:
            raise HTTPException(404, "station not found")
        in_day = and_(
            Prediction.station_id == station_id,
            Prediction.hour_start >= start,
            Prediction.hour_start < end,
        )
        version = latest_version(conn, in_day)
        rows = []
        if version is not None:
            rows = conn.execute(
                select(Prediction.hour_start, Prediction.predicted_rentals)
                .where(in_day, Prediction.model_version == version)
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
