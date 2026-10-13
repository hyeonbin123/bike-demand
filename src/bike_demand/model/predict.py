"""서비스 예측: 최신 스냅샷의 대여소 × 앞으로의 시간 × 최신 단기예보 → predictions 테이블.

- 대여소: 가장 최근 스냅샷(`realtime_snapshots`)에 나온 곳. 학습에 없던 대여소는 새 코드(학습에
  없던 범주)와 서비스 DB의 거치대 수·좌표로 예측한다(docs/experiments.md 측정 전 보충과 같은 처리)
- 시간: 현재 시각이 든 시간부터 `horizon_hours`개 중 최신 예보가 있는 시간
- 날씨: 서울 격자(60·127)의 가장 최근 발표(`weather_forecasts`)를
  `forecast_weather.hourly_weather`로 바꾼 값
- model_version: `{모델 이름}-fcst{발표 시각 YYYYMMDDHHMM}`. 같은 버전으로 다시 돌리면 값을 덮어쓴다
"""

# [파일 역할] 서비스 예측 작업. forecast_and_predict DAG가 단기예보 발표마다 실행해, 서비스 DB의
# 최신 스냅샷·최신 예보와 CURRENT 서비스 모델로 앞으로 48시간의 대여소별 시간당 예상 대여 수를
# predictions 표에 넣는다. API(/shortage-risk 등)는 이 표만 읽는다.
# [핵심 원칙] 서비스 특징은 학습 특징과 같아야 한다. 그래서 (1) 대여소 번호·패턴·추세·반기 수준은
# 학습 때 고정한 산출물(artifacts.py)에서 읽고, (2) 달력은 학습 시드와 같은 holidays KR 달력으로,
# (3) 날씨는 예보를 학습 정렬 규칙으로 바꿔(forecast_weather.py) 만들며, (4) 열 순서는 모델 자신의
# feature_name()을 따른다. tests/test_model_predict.py가 같은 대여소·시간에서 서비스 특징이 학습
# 특징과 같은지 확인한다(커밋 f20fe80).
# [버전] model_version = 세대 이름 + '-fcst' + 발표 시각. 발표마다 새 묶음이 생기고, 같은 버전을
# 다시 만들면 값과 생성 시각을 새로 써서 API가 그 묶음을 최신으로 고른다(커밋 a3eacdb).

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

# holidays: 학습 달력(dbt 시드)과 같은 KR 공휴일. lightgbm: 저장된 서비스 모델(Booster)로 예측.
# sqlalchemy의 postgresql insert: ON CONFLICT DO UPDATE(upsert)를 쓰려고 PostgreSQL 방언의 insert를
# 쓴다.
import holidays
import lightgbm as lgb
import numpy as np
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

# 프로젝트 모듈: 수집하는 서울 예보 격자, 학습 시점 산출물, 예보 → 날씨 특징 변환, 특징 목록과 공개
# 반기 번호, 서비스 DB 테이블.
from bike_demand.ingest.forecast import SEOUL_NX, SEOUL_NY
from bike_demand.model.artifacts import Artifacts
from bike_demand.model.forecast_weather import hourly_weather
from bike_demand.model.frames import FEATURES, published_half
from bike_demand.serving.models import Prediction, RealtimeSnapshot, Station, WeatherForecast

KST = timezone(timedelta(hours=9))


# 한 시간의 달력 특징을 dim_hours·frames.py와 같은 정의로 만든다(쉬는 날 = 토·일 또는 공휴일, 연중
# 일자 sin·cos는 365.25 주기). 호출 위치: feature_rows, model/shortage_eval.py(대상 시각의 쉬는 날
# 판정).
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


# 대여소 × 시간마다 특징 한 행을 만든다. features를 주면 그 순서(모델의 feature_name())를 따르고,
# 없으면 frames.FEATURES 순서.
# 호출 위치: predict_and_store, model/forecast_gap.py, model/shortage_eval.py,
# tests/test_model_predict.py.
# 성능: 커밋 9bde296에서 달력 계산을 시간마다 한 번으로 바꿨다. 2,700곳 × 48시간에 0.9초이고, 없앤
# 달력 계산만 약 31초였다(커밋 메시지).
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
    # 달력 특징은 시간에만 달려 있다. 대여소마다 다시 계산하면 공휴일 달력을 대여소×시간 번 만든다
    calendars = {hour_start: calendar_features(hour_start) for hour_start in hours}
    rows, keys = [], []
    for station in stations:
        station_id = station["station_id"]
        # 학습 때 없던 대여소: 학습의 가장 큰 번호 뒤로 새 번호를 차례로 매겨 '학습에 없던 범주'로
        # 넣고, 정적 정보는 서비스 DB의 값을 쓴다. 패턴·추세가 없으므로 아래에서 결측(NaN)이
        # 된다(docs/experiments.md v1 측정 전 보충과 같은 처리).
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
            # 학습 때 있던 대여소: 학습 산출물의 번호와 정적 정보를 그대로 쓴다.
            static = {
                "station_code": float(known["station_code"]),
                "district_code": _code(known["district_code"]),
                "docks": _num(known["docks"]),
                "lat": _num(known["lat"]),
                "lon": _num(known["lon"]),
            }
        trend = _num(artifacts.trend.get(station_id))
        for hour_start in hours:
            calendar = calendars[hour_start]
            # profile_mean은 (대여소, 쉬는 날 여부, 시각) 칸의 학습 기간 평균. 칸이 없으면 결측.
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
    # LightGBM 입력용 float32 행렬. reshape는 행이 0개일 때도 (0, 특징 수) 모양을 유지하게 한다.
    matrix = np.asarray(rows, dtype=np.float32).reshape(len(rows), len(features))
    return matrix, keys


# v2 수준 특징 세 개를 학습 SQL(frames.LEVEL_SELECT·LEVEL_JOINS)과 같은 반기 번호 규칙으로 계산한다.
# 공개 반기는 시각마다 다를 수 있다(예: 48시간이 달을 넘는 경우). 호출 위치: feature_rows.
# 반기 완료 검사는 커밋 e5dd59b에서 더했다: 산출물을 갱신하지 않고 다음 반기의 공개 시점을 넘기면,
# 자료 끝에 걸친 반기(0건인 하루만 있을 수 있음)를 최근 수준으로 읽게 되는 것을 막는다.
def level_features(station_id: str, hour_start: datetime, artifacts: Artifacts) -> dict[str, float]:
    """frames.LEVEL_SELECT와 같은 정의: 그 시각에 공개돼 있던 반기와 그 1년 전 반기.

    산출물의 자료가 그 반기를 다 담고 있지 않으면(산출물을 갱신하지 않고 반년 넘게 서비스한 경우)
    0이나 일부 값 대신 결측으로 둔다.
    """
    local = hour_start.astimezone(KST)
    half = published_half(local.year, local.month)
    if not artifacts.half_is_complete(half):
        return dict.fromkeys(
            ("station_recent_mean", "station_recent_ratio", "system_recent_ratio"), math.nan
        )
    mean = artifacts.halves.get((station_id, half))
    prior = artifacts.halves.get((station_id, half - 2))
    total = artifacts.system_halves.get(half)
    prior_total = artifacts.system_halves.get(half - 2)
    return {
        "station_recent_mean": _num(mean),
        "station_recent_ratio": _ratio(mean, prior),
        "system_recent_ratio": _ratio(total, prior_total),
    }


# 분자·분모 중 하나라도 없거나 분모가 0이면 결측(SQL의 nullif와 같은 처리).
def _ratio(numerator, denominator) -> float:
    if numerator is None or denominator is None or denominator == 0:
        return math.nan
    return float(numerator) / float(denominator)


# None을 NaN으로 바꾸는 도우미. LightGBM은 NaN을 결측으로 처리한다.
def _num(value) -> float:
    return math.nan if value is None else float(value)


# 범주 번호용 같은 변환(자치구가 없으면 결측 범주).
def _code(value) -> float:
    return math.nan if value is None else float(value)


# DB에서 서울 격자(60·127)의 가장 최근 발표 예보를 읽어 {hour_start: 날씨 특징}으로 바꾼다. 예보가
# 하나도 없으면 None. 발표 시각과 격자를 함께 걸러, 다른 격자의 예보가 DB에 있어도 그 발표를
# 고르거나 두 격자의 값을 섞지 않는다(forecast_gap·shortage_eval과 같은 격자 조건).
def latest_forecast(engine: Engine) -> tuple[datetime, dict[datetime, dict[str, float]]] | None:
    grid = (WeatherForecast.nx == SEOUL_NX) & (WeatherForecast.ny == SEOUL_NY)
    with engine.connect() as conn:
        latest = select(func.max(WeatherForecast.base_datetime)).where(grid)
        base = conn.execute(latest).scalar_one()
        if base is None:
            return None
        rows = conn.execute(
            select(
                WeatherForecast.fcst_datetime, WeatherForecast.category, WeatherForecast.value
            ).where(grid, WeatherForecast.base_datetime == base)
        ).all()
    return base, hourly_weather((r[0], r[1], r[2]) for r in rows)


# 가장 최근 스냅샷(전체 기준 fetched_at 최대) 한 번에 나온 대여소들을 예측 대상으로 한다. 그
# 스냅샷이 max_age(기본 6시간)보다 오래됐으면 빈 목록이라 예측을 만들지 않는다. 정적 정보는 서비스
# DB stations 표에서 붙인다.
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


# 예보·스냅샷 확인 → 예측할 시간 정하기 → 특징 행렬 → 예측 → upsert. 반환: (model_version, 저장한 행
# 수). 호출 위치: 아래 실행부(Airflow predict 작업).
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
    # 현재 시각이 든 시간부터 horizon_hours개 시간 중 최신 예보가 덮는 시간만 예측한다(예보에 없는
    # 시간은 날씨 특징을 만들 수 없음).
    first = now.astimezone(KST).replace(minute=0, second=0, microsecond=0)
    candidates = [first + timedelta(hours=i) for i in range(horizon_hours)]
    hours = [h for h in candidates if h in weather]
    if not hours:
        raise RuntimeError("예측할 시간의 예보가 없음")

    matrix, keys = feature_rows(stations, hours, weather, artifacts, booster.feature_name())
    predicted = booster.predict(matrix)
    # 버전 이름에 예보 발표 시각을 넣어, 어떤 모델 세대와 어떤 예보로 만든 예측인지 이름만으로 알 수
    # 있게 한다(진단 스크립트·v5 평가가 이 이름의 fcst 뒤 시각으로 발표를 찾는다).
    version = f"{model_name}-fcst{base.astimezone(KST):%Y%m%d%H%M}"
    records = [
        {"station_id": s, "hour_start": h, "model_version": version, "predicted_rentals": float(p)}
        for (s, h), p in zip(keys, predicted, strict=True)
    ]
    table = Prediction.__table__
    # 한 트랜잭션 안에서 5,000행씩 나눠 넣는다(한 문장의 바인드 변수 수가 너무 커지지 않게).
    # 키(대여소, 시간, 버전)가 겹치면 값과 created_at을 새로 쓴다.
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

    from bike_demand.model.final import load_serving_model
    from bike_demand.serving.db import make_engine

    # 실행: python -m bike_demand.model.predict --hours 48(DAG 기본). 기준 시각은 실행 순간의 KST.
    parser = argparse.ArgumentParser(description="최신 스냅샷·예보로 앞으로의 시간을 예측해 저장")
    parser.add_argument("--models", type=Path, default=Path("data/models/v1"))
    parser.add_argument("--hours", type=int, default=48)
    args = parser.parse_args()

    # CURRENT를 한 번 읽어 고른 한 세대의 모델·산출물만 쓴다. 세대 이름이 model_version 앞부분(T29)
    generation, booster, loaded = load_serving_model(args.models)
    now = datetime.now(KST)
    version, count = predict_and_store(make_engine(), booster, loaded, generation, now, args.hours)
    print(version, count)
