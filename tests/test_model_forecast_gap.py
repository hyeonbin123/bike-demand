"""v4(T12) 측정 계산. 실제 API·DB·모델 없이 합성 자료로 확인한다."""

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pytest

from bike_demand.model import artifacts, forecast_gap, frames

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]


def asos_item(tm, ta="20.0", rn="", ws="1.0", hm="50", snow=""):
    return {"tm": tm, "ta": ta, "rn": rn, "ws": ws, "hm": hm, "hr3Fhsc": snow}


def staging_sql_rows(items):
    """dbt stg_weather_hourly SQL을 같은 항목(bronze 열 이름)에 그대로 돌린 결과."""
    db = duckdb.connect()
    db.execute("""create table source_weather (observed_at varchar, temp_c varchar,
        wind_ms varchar, humidity_pct varchar, rain_mm varchar, new_snow_3h_cm varchar,
        snow_cm varchar)""")
    db.executemany(
        "insert into source_weather values (?, ?, ?, ?, ?, ?, ?)",
        [(i["tm"], i["ta"] or None, i["ws"] or None, i["hm"] or None, i["rn"] or None,
          i["hr3Fhsc"] or None, None) for i in items],
    )  # fmt: skip
    sql = (ROOT / "dbt/models/staging/stg_weather_hourly.sql").read_text(encoding="utf-8")
    sql = sql.replace("{{ source('bronze', 'asos_hourly') }}", "source_weather")
    cursor = db.execute(f"select hour_start, temp_c, rain_mm, wind_ms, humidity_pct, "
                        f"is_new_snow from ({sql}) order by hour_start")  # fmt: skip
    return {
        h.replace(tzinfo=KST): {"temp_c": t, "rain_mm": r, "wind_ms": w, "humidity_pct": hm,
                                "is_snow": float(s)}
        for h, t, r, w, hm, s in cursor.fetchall()
    }  # fmt: skip


def test_observed_weather_matches_the_training_staging_sql():
    items = [
        # 9월: 1시간 강수가 매시 보고, 결측 기온 하나
        asos_item("2026-09-17 01:00", rn="0.5"),
        asos_item("2026-09-17 02:00", ta=""),
        asos_item("2026-09-17 03:00", rn="2.0"),
        # 1월: 3시간 누적 강수와 3시간 신적설이 3의 배수 시각에만
        asos_item("2026-01-09 01:00"),
        asos_item("2026-01-09 02:00"),
        asos_item("2026-01-09 03:00", rn="0.9", snow="0.3"),
        asos_item("2026-01-09 04:00"),
    ]
    expected = staging_sql_rows(items)
    actual = forecast_gap.observed_hourly(items)
    assert actual.keys() == expected.keys()
    for hour, values in expected.items():
        for name, value in values.items():
            if value is None:
                assert math.isnan(actual[hour][name]), (hour, name)
            else:
                assert actual[hour][name] == pytest.approx(value), (hour, name)
    assert actual[datetime(2026, 1, 9, 0, tzinfo=KST)]["rain_mm"] == pytest.approx(0.3)
    assert actual[datetime(2026, 1, 9, 3, tzinfo=KST)]["is_snow"] == 0.0


def test_lead_buckets_count_from_the_issue_to_the_forecast_time():
    base = datetime(2026, 9, 17, 5, tzinfo=KST)
    # hour_start 05시의 예보 시각은 06시 → 간격 1시간
    assert forecast_gap.lead_bucket(base, base) == "1-6h"
    assert forecast_gap.lead_bucket(base - timedelta(hours=1), base) is None  # 간격 0
    assert forecast_gap.lead_bucket(base + timedelta(hours=5), base) == "1-6h"
    assert forecast_gap.lead_bucket(base + timedelta(hours=6), base) == "7-24h"
    assert forecast_gap.lead_bucket(base + timedelta(hours=47), base) == "25-48h"
    assert forecast_gap.lead_bucket(base + timedelta(hours=48), base) is None


def weather(temp, rain, snow=0.0, humidity=50.0, wind=1.0):
    return {"temp_c": temp, "rain_mm": rain, "wind_ms": wind, "humidity_pct": humidity,
            "is_snow": snow}  # fmt: skip


def test_weather_errors_by_lead_bucket():
    base = datetime(2026, 9, 17, 5, tzinfo=KST)
    hours = [base + timedelta(hours=i) for i in range(4)]
    forecast = {
        hours[0]: weather(21.0, 0.0),
        hours[1]: weather(19.0, 1.0),  # 비 예보, 실제 비
        hours[2]: weather(math.nan, 1.0),  # 기온 결측, 비 예보인데 안 옴
        hours[3]: weather(20.0, 0.0),  # 비 못 맞힘
    }
    observed = {
        hours[0]: weather(20.0, 0.0),
        hours[1]: weather(20.0, 3.0),
        hours[2]: weather(20.0, 0.0),
        hours[3]: weather(20.0, 0.5),
    }
    report = forecast_gap.weather_errors([(base, forecast)], observed)
    short = report["1-6h"]
    assert short["pairs"] == 4 and report["7-24h"]["pairs"] == 0
    assert short["temp_c"] == {"n": 3, "mae": pytest.approx(2 / 3), "bias": pytest.approx(0.0)}
    rain = short["rain"]
    assert (rain["observed_hours"], rain["forecast_hours"]) == (2, 2)
    assert rain["agreement"] == 0.5 and rain["recall"] == 0.5 and rain["precision"] == 0.5
    assert rain["amount_mae_when_observed"] == pytest.approx((2.0 + 0.5) / 2)
    assert rain["reference_only"] is True
    assert short["snow"]["measurable"] is False and short["snow"]["recall"] is None


def test_predictions_differ_only_through_weather(small_warehouse, tmp_path):
    artifacts.export(small_warehouse, ("2024-01-01", "2024-01-16"), tmp_path)
    loaded = artifacts.load(tmp_path)
    hours = [datetime(2024, 1, 16, h, tzinfo=KST) for h in range(3)]
    forecast = {h: weather(10.0, 0.0) for h in hours}
    observed = {h: weather(12.0, 1.0) for h in hours[:2]}  # 마지막 시간은 관측 없음
    features = frames.FEATURES
    temp = features.index("temp_c")

    def predict(matrix):
        return matrix[:, temp]

    keys, by_f, by_o = forecast_gap.predict_both(
        predict, [{"station_id": "ST-1"}, {"station_id": "ST-2"}], hours, forecast, observed,
        loaded, features,
    )  # fmt: skip
    assert len(keys) == 4 and {h for _, h in keys} == set(hours[:2])
    np.testing.assert_array_equal(by_f, 10.0)
    np.testing.assert_array_equal(by_o, 12.0)


def test_shift_totals_and_top_shortage_follow_the_api_order():
    as_of = datetime(2026, 9, 17, 8, 30, tzinfo=KST)
    hours = [datetime(2026, 9, 17, 8 + i, tzinfo=KST) for i in range(4)]
    keys = [(s, h) for s in ("A", "B", "C") for h in hours]
    by_f = np.array([2.0] * 4 + [1.0] * 4 + [3.0] * 4)
    by_o = np.array([1.0] * 4 + [1.0] * 4 + [3.0] * 4)
    observed = {h: weather(20.0, 1.0 if h.hour == 8 else 0.0) for h in hours}
    bikes = {"A": 0, "B": 0, "C": 9}
    totals = forecast_gap.ShiftTotals()
    totals.add(keys, by_f, by_o, observed, as_of, bikes, top_n=1)
    report = totals.report()
    assert report["rows"] == {"all": 12, "wet": 3, "dry": 9}
    assert report["mean_abs_diff"]["all"] == pytest.approx(4 / 12)
    assert report["mean_abs_diff"]["wet"] == pytest.approx(1 / 3)
    assert report["mean_ratio_forecast_over_observed"] == pytest.approx(24 / 20)
    # 3시간 합: 8시 절반 + 9·10시 + 11시 절반 = 3시간치. A만 차이(6 - 3)
    assert report["mean_abs_diff_3h_sum"] == pytest.approx(3 / 3)
    # 예보: A 6-0=6이 1위. 관측: A 3과 B 3이 같아 대여소ID 순으로 A. C는 9-9=0이라 빠짐
    assert report["top50_overlap"] == {"issues": 1, "mean": 1.0, "min": 1.0}
    assert forecast_gap.top_shortage(bikes, {"A": 3.0, "B": 3.0, "C": 9.0}, 5) == ["A", "B"]
