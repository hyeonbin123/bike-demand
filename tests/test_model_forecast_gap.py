"""v4(T12) 측정 계산. 실제 API·DB·모델 없이 합성 자료로 확인한다."""

# [무엇을 검증하나] v4 측정(관측 날씨 대신 단기예보를 쓸 때의 차이, model/forecast_gap.py)의 계산
# 규칙을 합성 자료로 확인한다.
# - 관측 날씨 변환이 학습 때의 dbt stg_weather_hourly SQL과 똑같은 결과를 내는지(그 SQL을 직접 돌려
#   비교)
# - 예보 간격 구간(1-6h, 7-24h, 25-48h) 나누기, 구간별 날씨 오차·강수 일치율, 비 온 시간 가드(고유
#   시각 기준)
# - 관측 받을 날짜와 '전날까지만 제공' 규칙
# - 예보/관측 날씨로 각각 예측할 때 날씨 외 특징은 같고 관측 없는 시각은 결측으로 남기기
# - 흔들림 합계를 비 옴·비 안 옴·알 수 없음으로 나누고, 부족 상위 목록 겹침을 API와 같은 순서로 세기
# [왜 중요한가] docs/experiments.md v4 '측정 전 보충'에 적은 세부 규칙이 코드와 어긋나면 v4
# 결과(README·docs/experiments.md 표의 수치)가 계획과 다른 것을 잰 셈이 된다.

import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# duckdb: dbt staging SQL을 메모리 DB에서 그대로 실행한다. artifacts·frames: 예측 비교에 쓰는 학습
# 산출물과 특징 목록.
import duckdb
import numpy as np
import pytest

from bike_demand.model import artifacts, forecast_gap, frames

KST = timezone(timedelta(hours=9))
# 저장소 루트. dbt 모델 SQL 파일을 읽는 데 쓴다.
ROOT = Path(__file__).resolve().parents[1]


# ASOS API 항목 하나(관측 시각, 기온, 강수, 풍속, 습도, 3시간 신적설). 빈 문자열은 원본처럼 '값
# 없음'.
def asos_item(tm, ta="20.0", rn="", ws="1.0", hm="50", snow=""):
    return {"tm": tm, "ta": ta, "rn": rn, "ws": ws, "hm": hm, "hr3Fhsc": snow}


# 같은 항목을 bronze 열 이름으로 메모리 DuckDB 표에 넣고, 저장소의 stg_weather_hourly.sql에서 dbt
# source 참조만 그 표 이름으로 바꿔 실행한다. 파이썬 구현을 손으로 쓴 기대값이 아니라 학습이 실제로
# 쓴 SQL과 비교하려는 것이다.
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


# 9월(매시 1시간 강수, 결측 기온 하나)과 1월(3의 배수 시각에만 3시간 누적 강수·신적설) 관측으로
# 파이썬 변환과 SQL 결과가 모든 시각·특징에서 같은지 본다. SQL의 NULL은 파이썬의 NaN에 대응한다.
# 겨울 3시간 강수 0.9mm는 앞선 세 시간에 0.3씩 나뉘고, 03시 관측의 신적설은 그 앞 세 시간만 켜므로
# hour_start 03시(04시 관측)는 눈 0이다.
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


# 간격 = (hour_start + 1시간) − 발표 시각(예보 시각 기준). 05시 발표에서 hour_start 05시는 06시
# 예보라 1시간 → 1-6h, 간격 0은 구간 밖, 6시간 → 1-6h, 7시간 → 7-24h, 48시간 → 25-48h, 49시간은 구간
# 밖.
def test_lead_buckets_count_from_the_issue_to_the_forecast_time():
    base = datetime(2026, 9, 17, 5, tzinfo=KST)
    # hour_start 05시의 예보 시각은 06시 → 간격 1시간
    assert forecast_gap.lead_bucket(base, base) == "1-6h"
    assert forecast_gap.lead_bucket(base - timedelta(hours=1), base) is None  # 간격 0
    assert forecast_gap.lead_bucket(base + timedelta(hours=5), base) == "1-6h"
    assert forecast_gap.lead_bucket(base + timedelta(hours=6), base) == "7-24h"
    assert forecast_gap.lead_bucket(base + timedelta(hours=47), base) == "25-48h"
    assert forecast_gap.lead_bucket(base + timedelta(hours=48), base) is None


# 날씨 특징 사전 하나를 만드는 도우미.
def weather(temp, rain, snow=0.0, humidity=50.0, wind=1.0):
    return {"temp_c": temp, "rain_mm": rain, "wind_ms": wind, "humidity_pct": humidity,
            "is_snow": snow}  # fmt: skip


# 1-6h 구간에 네 쌍: 기온은 결측 하나를 빼고 3쌍 MAE 2/3·편향 0. 강수는 예보·관측 비 옴이 각각
# 2쌍이고 맞힌 것은 1쌍이라 일치율·재현율·정밀도 모두 0.5, 비 온 시간 강수량 MAE는 (2.0 + 0.5)/2. 비
# 온 시간이 가드(10시간)보다 적어 참고용이고, 눈이 없으면 '잴 수 없음'(재현율 None).
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
    assert rain["observed_pairs"] == rain["observed_unique_hours"] == rain["forecast_pairs"] == 2
    assert rain["agreement"] == 0.5 and rain["recall"] == 0.5 and rain["precision"] == 0.5
    assert rain["amount_mae_when_observed"] == pytest.approx((2.0 + 0.5) / 2)
    assert rain["reference_only"] is True
    assert short["snow"]["measurable"] is False and short["snow"]["recall"] is None


# 같은 비 온 두 시간을 발표 8개가 담으면 쌍은 16개로 가드(10)를 넘지만 고유 시각은 2개다. 가드는
# 고유 시각으로 판단해야 한 번의 비를 여러 발표가 부풀리지 못한다.
def test_rain_guard_counts_distinct_hours_not_issue_pairs():
    """같은 비 온 두 시간을 발표 8개가 담아도 고유 시각은 2라 참고용으로 남는다(T47)."""
    target = [datetime(2026, 9, 20, 12, tzinfo=KST) + timedelta(hours=i) for i in range(2)]
    observed = {h: weather(20.0, 2.0) for h in target}
    issues = []
    for n in range(8):
        base = target[0] - timedelta(hours=26 + 2 * n)  # 간격 27~42시간
        issues.append((base, {h: weather(20.0, 1.0) for h in target}))
    rain = forecast_gap.weather_errors(issues, observed)["25-48h"]["rain"]
    assert rain["observed_pairs"] == 16 >= forecast_gap.MIN_RAIN_HOURS
    assert rain["observed_unique_hours"] == 2 and rain["reference_only"] is True


# 마지막 발표 09-23 23시의 48시간 앞은 09-25이므로 관측은 09-25까지 받는다. API가 전날까지만 주므로
# 09-26에는 받을 수 있고 09-25 당일에는 SystemExit.
def test_observed_download_ends_with_the_longest_lead_and_waits_a_day():
    """기본 마지막 발표 09-23 23시 → 관측 09-25까지. 09-26에는 받고 09-25에는 거부(T48)."""
    end = forecast_gap.observed_end(datetime(2026, 9, 23, 23, tzinfo=KST))
    assert end == date(2026, 9, 25)
    forecast_gap.check_fetchable(end, date(2026, 9, 26))
    with pytest.raises(SystemExit):
        forecast_gap.check_fetchable(end, date(2026, 9, 25))


# conftest의 small_warehouse로 실제 산출물을 만들고, 예측 함수 대신 '기온 열을 그대로 돌려주는
# 함수'를 넣어 어떤 날씨가 들어갔는지 직접 본다. 대여소 2곳 × 3시간 = 6행 모두 남고, 예보 쪽은 모두
# 10도, 관측 쪽은 관측이 있는 시각 12도·없는 마지막 시각은 NaN이다. 관측 행이 없는 시각을 빼지 않고
# 결측으로 두는 규칙.
def test_predictions_keep_hours_without_observations(small_warehouse, tmp_path):
    """관측 행이 없는 시각도 빼지 않고 날씨만 결측으로 둔다. 날씨 외 특징은 같다(T46)."""
    artifacts.export(small_warehouse, ("2024-01-01", "2024-01-16"), tmp_path)
    loaded = artifacts.load(tmp_path)
    hours = [datetime(2024, 1, 16, h, tzinfo=KST) for h in range(3)]
    forecast = {h: weather(10.0, 0.0) for h in hours}
    observed = {h: weather(12.0, 1.0) for h in hours[:2]}  # 마지막 시간은 관측 행 없음
    features = frames.FEATURES
    temp = features.index("temp_c")

    def predict(matrix):
        return matrix[:, temp]

    keys, by_f, by_o = forecast_gap.predict_both(
        predict, [{"station_id": "ST-1"}, {"station_id": "ST-2"}], hours, forecast, observed,
        loaded, features,
    )  # fmt: skip
    assert len(keys) == 6 and {h for _, h in keys} == set(hours)
    np.testing.assert_array_equal(by_f, 10.0)
    missing = np.array([h == hours[2] for _, h in keys])
    np.testing.assert_array_equal(by_o[~missing], 12.0)
    assert np.isnan(by_o[missing]).all()


# 대여소 A·B·C × 08~11시. 예보/관측 예측이 A만 다르다(2 대 1).
# - 시간 단위: 전체 12행의 평균 절대 차이 4/12, 비 온 08시(3행)는 1/3, 예측 평균 비율 24/20
# - 3시간 합: as_of 08:30부터 08시 절반 + 09·10시 + 11시 절반이라 A만 6 대 3, 평균 3/3. 구간에 비 온
#   08시가 걸려 이 발표는 wet
# - 상위 1곳 목록: 예보 쪽은 A(부족 6), 관측 쪽은 A·B가 3으로 같아 대여소ID 순으로 A, 자전거 9대인
#   C는 부족 0이라 빠짐 → 겹침 100%
# top_shortage가 /shortage-risk와 같은 순서(부족 큰 순, 같으면 ID 순, 부족 > 0만)인지도 따로 본다.
def test_shift_totals_split_by_rain_and_follow_the_api_order():
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
    assert report["all"]["rows"] == 12
    assert (report["wet"]["rows"], report["dry"]["rows"], report["unknown"]["rows"]) == (3, 9, 0)
    assert report["all"]["mean_abs_diff"] == pytest.approx(4 / 12)
    assert report["wet"]["mean_abs_diff"] == pytest.approx(1 / 3)
    assert report["all"]["mean_ratio_forecast_over_observed"] == pytest.approx(24 / 20)
    assert report["wet"]["mean_ratio_forecast_over_observed"] == pytest.approx(6 / 5)
    # 3시간 합: 8시 절반 + 9·10시 + 11시 절반. A만 차이(6 - 3). 8시에 비가 와 이 발표는 wet
    assert report["all"]["mean_abs_diff_3h_sum"] == pytest.approx(3 / 3)
    assert report["wet"]["issues"] == 1 and report["wet"]["stations_3h"] == 3
    assert report["dry"]["issues"] == 0 and report["dry"]["mean_abs_diff_3h_sum"] is None
    # 예보: A 6-0=6이 1위. 관측: A 3과 B 3이 같아 대여소ID 순으로 A. C는 9-9=0이라 빠짐
    assert report["all"]["top50_overlap"] == {"issues": 1, "mean": 1.0, "min": 1.0}
    assert report["wet"]["top50_overlap"]["issues"] == 1
    assert forecast_gap.top_shortage(bikes, {"A": 3.0, "B": 3.0, "C": 9.0}, 5) == ["A", "B"]


# 30분 안 스냅샷이 없으면(bike_counts=None) 예측 흔들림은 세되 목록 비교만 빼고
# 'no_recent_snapshot'으로 센다. 관측 행이 없는 10·11시는 unknown 무리로, 그 시간이 3시간 구간에
# 걸린 발표도 unknown이다.
# 예보 쪽 목록이 비면(자전거 99대라 부족 없음) 겹침을 0으로 세지 않고 'empty_forecast_list'로 따로
# 센다(겹침 = 교집합 ÷ 예보 목록 길이라 0으로 나눌 수 없음).
def test_shift_totals_without_recent_snapshot_or_observations():
    """30분 안 스냅샷이 없으면 목록 비교만 빼고, 관측 없는 시간은 unknown으로 센다(T46, T49)."""
    as_of = datetime(2026, 9, 17, 8, 15, tzinfo=KST)
    hours = [datetime(2026, 9, 17, 8 + i, tzinfo=KST) for i in range(4)]
    keys = [("A", h) for h in hours]
    observed = {h: weather(20.0, 0.0) for h in hours[:2]}  # 10·11시 관측 행 없음
    totals = forecast_gap.ShiftTotals()
    totals.add(keys, np.full(4, 2.0), np.full(4, 1.0), observed, as_of, None)
    report = totals.report()
    assert (report["dry"]["rows"], report["unknown"]["rows"]) == (2, 2)
    assert report["unknown"]["issues"] == 1 and report["unknown"]["stations_3h"] == 1
    assert report["all"]["mean_abs_diff_3h_sum"] == pytest.approx(3.0)
    assert report["all"]["top50_overlap"]["issues"] == 0
    assert report["lists_skipped"] == {"no_recent_snapshot": 1, "empty_forecast_list": 0}

    empty = forecast_gap.ShiftTotals()
    empty.add(keys, np.full(4, 2.0), np.full(4, 1.0), observed, as_of, {"A": 99})
    assert empty.report()["lists_skipped"] == {"no_recent_snapshot": 0, "empty_forecast_list": 1}
