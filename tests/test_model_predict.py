from datetime import date, datetime, timedelta, timezone

import lightgbm as lgb
import numpy as np
import pytest
from sqlalchemy import select

from bike_demand.model import artifacts, frames, predict
from bike_demand.serving.models import Prediction, RealtimeSnapshot, Station, WeatherForecast

KST = timezone(timedelta(hours=9))
HISTORY = ("2024-01-01", "2024-01-16")


def test_serving_features_match_training_features(small_warehouse, tmp_path):
    """같은 대여소·시간이면 학습 행(frames)과 서비스 행(predict)의 특징이 같아야 한다."""
    artifacts.export(small_warehouse, HISTORY, tmp_path)
    loaded = artifacts.load(tmp_path)

    window = frames.Window(rows=("2024-01-16", "2024-01-18"), history=HISTORY)
    frame = frames.load_frame(small_warehouse, window, only_active_stations=False)
    training = frames.feature_matrix(frame, frames.FEATURES)

    hours_table = small_warehouse.execute(
        """select hour_start, temp_c, rain_mm, wind_ms, humidity_pct, is_new_snow::int
           from dim_hours where hour_start >= '2024-01-16' and hour_start < '2024-01-18'
           order by hour_start"""
    ).fetchall()
    hours = [h.replace(tzinfo=KST) for h, *_ in hours_table]
    weather = {
        h.replace(tzinfo=KST): dict(zip(frames.WEATHER_FEATURES, map(float, values), strict=True))
        for h, *values in hours_table
    }
    stations = [{"station_id": "ST-1"}, {"station_id": "ST-2"}]
    serving, keys = predict.feature_rows(stations, hours, weather, loaded)

    # 학습 행을 (대여소 코드, 날짜, 시간) 순으로 맞춰 비교
    order = np.lexsort((frame["hour_of_day"], frame["day_index"], frame["station_code"]))
    np.testing.assert_allclose(serving, training[order], rtol=1e-5, equal_nan=True)
    assert keys[0] == ("ST-1", datetime(2024, 1, 16, tzinfo=KST))


def test_serving_level_features_match_training(two_year_warehouse, tmp_path):
    """v2 수준 특징도 학습 행과 서비스 행이 같아야 한다(공개 반기 경계가 있는 달들)."""
    history = ("2023-01-01", "2025-01-01")
    meta = artifacts.export(two_year_warehouse, history, tmp_path, with_levels=True)
    loaded = artifacts.load(tmp_path)
    features = [*frames.FEATURES, *frames.LEVEL_FEATURES]
    assert meta["features"] == features
    for day in ("2023-05-10", "2023-08-01", "2024-07-31", "2024-08-01", "2025-03-15"):
        next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        window = frames.Window(rows=(day, next_day), history=history)
        frame = frames.load_frame(two_year_warehouse, window, False, with_levels=True)
        training = frames.feature_matrix(frame, features)
        hours = sorted(
            datetime(*map(int, day.split("-")), int(h), tzinfo=KST) for h in frame["hour_of_day"]
        )
        weather = {h: {"temp_c": 10.0, "rain_mm": 0.0, "wind_ms": 1.0, "humidity_pct": 50.0,
                       "is_snow": 0.0} for h in hours}  # fmt: skip
        station = [{"station_id": "ST-1"}]
        serving, _ = predict.feature_rows(station, hours, weather, loaded, features)
        order = np.argsort(frame["hour_of_day"])
        np.testing.assert_allclose(serving, training[order], rtol=1e-5, equal_nan=True)


def test_unseen_station_gets_new_code_and_serving_values(small_warehouse, tmp_path):
    artifacts.export(small_warehouse, HISTORY, tmp_path)
    loaded = artifacts.load(tmp_path)
    hour = datetime(2026, 9, 17, 8, tzinfo=KST)
    weather = {hour: dict.fromkeys(frames.WEATHER_FEATURES, 0.0)}
    new = {"station_id": "ST-NEW", "district": "강남구", "docks": 9, "lat": 37.1, "lon": 127.1}
    matrix, _ = predict.feature_rows([new], [hour], weather, loaded)
    row = dict(zip(frames.FEATURES, matrix[0], strict=True))
    assert row["station_code"] == loaded.max_station_code + 1
    assert row["district_code"] == loaded.district_codes["강남구"]
    assert row["docks"] == 9 and np.isnan(row["profile_mean"])


def tiny_booster():
    rng = np.random.default_rng(0)
    x = rng.random((400, len(frames.FEATURES)), dtype=np.float32)
    y = rng.poisson(2, 400).astype(np.float32)
    data = lgb.Dataset(x, y, feature_name=frames.FEATURES)
    return lgb.train({"objective": "poisson", "verbose": -1, "num_leaves": 4}, data, 5)


def test_predict_and_store_writes_and_overwrites(pg_engine, small_warehouse, tmp_path):
    artifacts.export(small_warehouse, HISTORY, tmp_path)
    loaded = artifacts.load(tmp_path)
    now = datetime(2026, 9, 17, 8, 20, tzinfo=KST)
    base = datetime(2026, 9, 17, 5, tzinfo=KST)
    with pg_engine.begin() as conn:
        conn.execute(
            Station.__table__.insert(),
            [
                {"station_id": "ST-1", "source": "history"},
                {"station_id": "ST-9", "source": "realtime"},
            ],
        )
        conn.execute(
            RealtimeSnapshot.__table__.insert(),
            [{"station_id": s, "fetched_at": now - timedelta(minutes=10), "bike_count": 3}
             for s in ("ST-1", "ST-9")],
        )  # fmt: skip
        conn.execute(
            WeatherForecast.__table__.insert(),
            [{"base_datetime": base, "fcst_datetime": base + timedelta(hours=h), "category": c,
              "nx": 60, "ny": 127, "value": v}
             for h in range(4, 8) for c, v in (("TMP", "20"), ("PCP", "강수없음"))],
        )  # fmt: skip

    booster = tiny_booster()
    version, count = predict.predict_and_store(pg_engine, booster, loaded, "v1-test", now)
    # 예보 시각 09~12시 → 시작 시간 08~11시, 대여소 2곳
    assert version == "v1-test-fcst202609170500"
    assert count == 8
    predict.predict_and_store(pg_engine, booster, loaded, "v1-test", now)
    with pg_engine.connect() as conn:
        rows = conn.execute(select(Prediction.__table__)).all()
    assert len(rows) == 8
    assert min(r.hour_start for r in rows) == datetime(2026, 9, 17, 8, tzinfo=KST)


def test_predict_refuses_stale_snapshot(pg_engine, small_warehouse, tmp_path):
    artifacts.export(small_warehouse, HISTORY, tmp_path)
    now = datetime(2026, 9, 17, 8, tzinfo=KST)
    with pg_engine.begin() as conn:
        conn.execute(Station.__table__.insert(), [{"station_id": "ST-1", "source": "history"}])
        conn.execute(
            RealtimeSnapshot.__table__.insert(),
            [{"station_id": "ST-1", "fetched_at": now - timedelta(days=1), "bike_count": 3}],
        )
        conn.execute(
            WeatherForecast.__table__.insert(),
            [{"base_datetime": now, "fcst_datetime": now + timedelta(hours=1),
              "category": "TMP", "nx": 60, "ny": 127, "value": "20"}],
        )  # fmt: skip
    with pytest.raises(RuntimeError, match="스냅샷"):
        predict.predict_and_store(pg_engine, tiny_booster(), artifacts.load(tmp_path), "v", now)


def test_stale_artifacts_give_missing_level_features_not_a_partial_half():
    """자료 끝(2026-07-01) 뒤에 걸친 반기는 0건인 하루만 집계돼 있을 수 있다. 산출물을 갱신하지 않고
    그 반기의 공개 시점(2027-02)을 넘겨도 그 값을 최근 수준으로 읽지 않는다(T53)."""
    h1_2026 = (2026 * 12 + 0) // 6
    stale = artifacts.Artifacts(
        stations={}, district_codes={}, profile={}, trend={}, global_trend=None,
        max_station_code=0,
        halves={("ST-1", h1_2026): 4.0, ("ST-1", h1_2026 - 2): 2.0, ("ST-1", h1_2026 + 1): 0.0},
        system_halves={h1_2026: 400.0, h1_2026 - 2: 200.0, h1_2026 + 1: 0.0},
        history_end=date(2026, 7, 1),
    )  # fmt: skip
    now = predict.level_features("ST-1", datetime(2026, 9, 21, 8, tzinfo=KST), stale)
    assert now == {"station_recent_mean": 4.0, "station_recent_ratio": 2.0,
                   "system_recent_ratio": 2.0}  # fmt: skip
    later = predict.level_features("ST-1", datetime(2027, 2, 1, 8, tzinfo=KST), stale)
    assert all(np.isnan(value) for value in later.values())
    # 자료 끝을 모르는 예전 산출물은 그대로 읽는다
    legacy = artifacts.Artifacts(
        stations={}, district_codes={}, profile={}, trend={}, global_trend=None,
        max_station_code=0, halves={("ST-1", h1_2026 + 1): 0.0},
    )  # fmt: skip
    old_way = predict.level_features("ST-1", datetime(2027, 2, 1, 8, tzinfo=KST), legacy)
    assert old_way["station_recent_mean"] == 0.0


def empty_artifacts():
    return artifacts.Artifacts(
        stations={}, district_codes={}, profile={}, trend={}, global_trend=None,
        max_station_code=0,
    )  # fmt: skip


# 2026-09-25(금)은 추석, 09-29(화)는 평일
CALENDAR_HOURS = [datetime(2026, 9, d, h, tzinfo=KST) for d in (25, 29) for h in (8, 18)]


def test_feature_rows_computes_calendar_once_per_hour(monkeypatch):
    """달력 특징은 시간에만 달려 있다. 대여소×시간마다 공휴일 달력을 다시 만들지 않는다."""
    calls = []
    original = predict.calendar_features

    def counting(hour_start):
        calls.append(hour_start)
        return original(hour_start)

    monkeypatch.setattr(predict, "calendar_features", counting)
    weather = {h: dict.fromkeys(frames.WEATHER_FEATURES, 0.0) for h in CALENDAR_HOURS}
    stations = [{"station_id": f"ST-{i}"} for i in range(3)]
    matrix, keys = predict.feature_rows(stations, CALENDAR_HOURS, weather, empty_artifacts())
    assert sorted(calls) == CALENDAR_HOURS
    assert matrix.shape == (12, len(frames.FEATURES)) and len(keys) == 12


def test_feature_rows_keep_holiday_calendar():
    weather = {h: dict.fromkeys(frames.WEATHER_FEATURES, 0.0) for h in CALENDAR_HOURS}
    stations = [{"station_id": f"ST-{i}"} for i in range(3)]
    matrix, keys = predict.feature_rows(stations, CALENDAR_HOURS, weather, empty_artifacts())
    column = {name: frames.FEATURES.index(name) for name in ("is_holiday", "is_offday")}
    for row, (_, hour) in zip(matrix, keys, strict=True):
        expected = 1.0 if hour.day == 25 else 0.0
        assert row[column["is_holiday"]] == row[column["is_offday"]] == expected
        assert row[column["is_holiday"]] == predict.calendar_features(hour)["is_holiday"]
