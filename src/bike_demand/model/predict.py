"""서비스 예측: 최신 스냅샷의 대여소 × 앞으로의 시간 × 최신 단기예보 → predictions 테이블.

- 대여소: 가장 최근 스냅샷(`realtime_snapshots`)에 나온 곳. 학습에 없던 대여소는 새 코드(학습에
  없던 범주)와 서비스 DB의 거치대 수·좌표로 예측한다(docs/experiments.md 측정 전 보충과 같은 처리)
- 시간: 현재 시각이 든 시간부터 `horizon_hours`개 중 최신 예보가 있는 시간
- 날씨: 가장 최근 발표(`weather_forecasts`)를 `forecast_weather.hourly_weather`로 바꾼 값
- model_version: `{모델 이름}-fcst{발표 시각 YYYYMMDDHHMM}`. 같은 버전으로 다시 돌리면 값을 덮어쓴다
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import holidays
import lightgbm as lgb
import numpy as np
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

from bike_demand.model.artifacts import Artifacts
from bike_demand.model.forecast_weather import hourly_weather
from bike_demand.model.frames import FEATURES, published_half
from bike_demand.serving.models import Prediction, RealtimeSnapshot, Station, WeatherForecast

KST = timezone(timedelta(hours=9))


def calendar_features(hour_start: datetime) -> dict[str, float]:
    local = hour_start.astimezone(KST)
    is_holiday = local.date() in holidays.country_holidays("KR", years=local.year)
    day_of_year = local.timetuple().tm_yday
    return {
        "hour_of_day": float(local.hour),
        "day_of_week": float(local.isoweekday()),
        "is_offday": float(local.isoweekday() >= 6 or is_holiday),
        "is_holiday": float(is_holiday),
        "month": float(local.month),
        "doy_sin": math.sin(2 * math.pi * day_of_year / 365.25),
        "doy_cos": math.cos(2 * math.pi * day_of_year / 365.25),
    }


def feature_rows(
    stations: list[dict],
    hours: list[datetime],
    weather: dict[datetime, dict[str, float]],
    artifacts: Artifacts,
    features: list[str] | None = None,
) -> tuple[np.ndarray, list[tuple[str, datetime]]]:
    """(특징 행렬, (station_id, hour_start) 목록). 열 순서는 features(기본 frames.FEATURES)."""
    features = features or FEATURES
    unseen = {}
    rows, keys = [], []
    for station in stations:
        station_id = station["station_id"]
        known = artifacts.stations.get(station_id)
        if known is None:
            code = unseen.setdefault(station_id, artifacts.max_station_code + 1 + len(unseen))
            district = station.get("district")
            static = {
                "station_code": float(code),
                "district_code": _code(artifacts.district_codes.get(district)),
                "docks": _num(station.get("docks")),
                "lat": _num(station.get("lat")),
                "lon": _num(station.get("lon")),
            }
        else:
            static = {
                "station_code": float(known["station_code"]),
                "district_code": _code(known["district_code"]),
                "docks": _num(known["docks"]),
                "lat": _num(known["lat"]),
                "lon": _num(known["lon"]),
            }
        trend = _num(artifacts.trend.get(station_id))
        for hour_start in hours:
            calendar = calendar_features(hour_start)
            profile = artifacts.profile.get(
                (station_id, bool(calendar["is_offday"]), int(calendar["hour_of_day"]))
            )
            values = {
                **static,
                **calendar,
                **weather[hour_start],
                "profile_mean": _num(profile),
                "station_trend": trend,
                "global_trend": _num(artifacts.global_trend),
                **level_features(station_id, hour_start, artifacts),
            }
            rows.append([values[name] for name in features])
            keys.append((station_id, hour_start))
    matrix = np.asarray(rows, dtype=np.float32).reshape(len(rows), len(features))
    return matrix, keys


def level_features(station_id: str, hour_start: datetime, artifacts: Artifacts) -> dict[str, float]:
    """frames.LEVEL_SELECT와 같은 정의: 그 시각에 공개돼 있던 반기와 그 1년 전 반기."""
    local = hour_start.astimezone(KST)
    half = published_half(local.year, local.month)
    mean = artifacts.halves.get((station_id, half))
    prior = artifacts.halves.get((station_id, half - 2))
    total = artifacts.system_halves.get(half)
    prior_total = artifacts.system_halves.get(half - 2)
    return {
        "station_recent_mean": _num(mean),
        "station_recent_ratio": _ratio(mean, prior),
        "system_recent_ratio": _ratio(total, prior_total),
    }


def _ratio(numerator, denominator) -> float:
    if numerator is None or denominator is None or denominator == 0:
        return math.nan
    return float(numerator) / float(denominator)


def _num(value) -> float:
    return math.nan if value is None else float(value)


def _code(value) -> float:
    return math.nan if value is None else float(value)


def latest_forecast(engine: Engine) -> tuple[datetime, dict[datetime, dict[str, float]]] | None:
    with engine.connect() as conn:
        base = conn.execute(select(func.max(WeatherForecast.base_datetime))).scalar_one()
        if base is None:
            return None
        rows = conn.execute(
            select(
                WeatherForecast.fcst_datetime, WeatherForecast.category, WeatherForecast.value
            ).where(WeatherForecast.base_datetime == base)
        ).all()
    return base, hourly_weather((r[0], r[1], r[2]) for r in rows)


def latest_stations(engine: Engine, max_age: timedelta, now: datetime) -> list[dict]:
    with engine.connect() as conn:
        latest = conn.execute(select(func.max(RealtimeSnapshot.fetched_at))).scalar_one()
        if latest is None or now - latest > max_age:
            return []
        rows = conn.execute(
            select(Station.station_id, Station.district, Station.docks, Station.lat, Station.lon)
            .join(RealtimeSnapshot, RealtimeSnapshot.station_id == Station.station_id)
            .where(RealtimeSnapshot.fetched_at == latest)
            .order_by(Station.station_id)
        ).mappings()
        return [dict(row) for row in rows]


def predict_and_store(
    engine: Engine,
    booster: lgb.Booster,
    artifacts: Artifacts,
    model_name: str,
    now: datetime,
    horizon_hours: int = 48,
    max_snapshot_age: timedelta = timedelta(hours=6),
) -> tuple[str, int]:
    """(model_version, 저장한 행 수). 스냅샷이나 예보가 없으면 RuntimeError."""
    forecast = latest_forecast(engine)
    if forecast is None:
        raise RuntimeError("단기예보가 없음")
    base, weather = forecast
    stations = latest_stations(engine, max_snapshot_age, now)
    if not stations:
        raise RuntimeError("최근 스냅샷이 없음")
    first = now.astimezone(KST).replace(minute=0, second=0, microsecond=0)
    candidates = [first + timedelta(hours=i) for i in range(horizon_hours)]
    hours = [h for h in candidates if h in weather]
    if not hours:
        raise RuntimeError("예측할 시간의 예보가 없음")

    matrix, keys = feature_rows(stations, hours, weather, artifacts, booster.feature_name())
    predicted = booster.predict(matrix)
    version = f"{model_name}-fcst{base.astimezone(KST):%Y%m%d%H%M}"
    records = [
        {"station_id": s, "hour_start": h, "model_version": version, "predicted_rentals": float(p)}
        for (s, h), p in zip(keys, predicted, strict=True)
    ]
    table = Prediction.__table__
    with engine.begin() as conn:
        for start in range(0, len(records), 5000):
            statement = insert(table)
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=["station_id", "hour_start", "model_version"],
                    # 같은 버전을 다시 만들면 값과 생성 시각을 새로 한다
                    # (API는 생성 시각으로 쓸 버전을 고름, docs/api.md)
                    set_={
                        "predicted_rentals": statement.excluded.predicted_rentals,
                        "created_at": func.now(),
                    },
                ),
                records[start : start + 5000],
            )
    return version, len(records)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from bike_demand.model import artifacts as artifacts_module
    from bike_demand.serving.db import make_engine

    parser = argparse.ArgumentParser(description="최신 스냅샷·예보로 앞으로의 시간을 예측해 저장")
    parser.add_argument("--models", type=Path, default=Path("data/models/v1"))
    parser.add_argument("--name", help="model_version 앞부분 (기본: serving.json의 측정 버전)")
    parser.add_argument("--hours", type=int, default=48)
    args = parser.parse_args()

    serving = args.models / "serving"
    import json

    serving_info = json.loads((serving / "serving.json").read_text("utf-8"))
    name = args.name or serving_info.get("version", "v1")
    booster = lgb.Booster(model_file=str(serving / "model.txt"))
    loaded = artifacts_module.load(serving / "artifacts")
    now = datetime.now(KST)
    version, count = predict_and_store(make_engine(), booster, loaded, name, now, args.hours)
    print(version, count)
