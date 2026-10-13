"""측정용 데이터(대여소×시간 행)와 특징을 DuckDB warehouse에서 만든다. docs/experiments.md v1

기간은 모두 [시작, 끝) 반열린 구간의 날짜 문자열("YYYY-MM-DD")이다.
대여소 과거 패턴·추세는 `Window.history`(train 기간)에서만 계산해 평가 기간 값이 섞이지 않게 한다.
"""

# [파일 역할] 모든 측정(v1~v3)과 서비스 모델 학습이 쓰는 '대여소×시간 행 + 특징' 생성기. dbt가 만든
# warehouse(DuckDB)의 격자·달력·대여소 표를 SQL 한 번으로 묶어 numpy 열 사전으로 돌려준다.
# [핵심 원칙] 예측 시점에 알 수 있는 것만 특징으로 쓴다(docs/experiments.md v1). 최근 몇 시간 대여
# 수 같은 lag 특징은 서비스 때 없으므로(대여이력은 반기 공개) 쓰지 않고, 대여소 패턴·추세는
# Window.history(학습 기간)에서만 계산한다.
# [구성] 특징 목록 상수 → 기간(Window) → 학습 기간 CTE(history_ctes) → v2 수준 특징 SQL 조각 → v3
# 시점별 대여소 정보 SQL 조각 → 전체 SQL 조립(_sql) → 실행(load_frame) → 행렬 변환(feature_matrix).
# [연결] model/validate.py·validate_v2.py·validate_v3.py(측정), model/final.py(test·서비스 학습),
# model/artifacts.py(서비스용 산출물이 같은 CTE 정의를 씀), model/predict.py(서비스 예측이 같은 특징
# 순서·반기 번호를 씀).

from __future__ import annotations

from dataclasses import dataclass

# duckdb: warehouse 파일에 SQL을 실행하고 결과를 numpy로 바로 받는다(fetchnumpy). numpy: LightGBM에
# 넘길 float32 배열.
import duckdb
import numpy as np

# 날씨 특징. M2는 이것만 뺀다.
WEATHER_FEATURES = ["temp_c", "rain_mm", "wind_ms", "humidity_pct", "is_snow"]
# M1(v1)의 특징 순서. 이 순서가 곧 LightGBM 입력 열 순서라 학습과 서비스 예측(predict.py)이 같은
# 목록을 쓴다. 대여소 정적 정보(코드, 자치구, 거치대, 좌표) → 달력 → 날씨 → 대여소 과거
# 패턴(profile_mean) → 추세 순.
FEATURES = [
    "station_code",
    "district_code",
    "docks",
    "lat",
    "lon",
    "hour_of_day",
    "day_of_week",
    "is_offday",
    "is_holiday",
    "month",
    "doy_sin",
    "doy_cos",
    *WEATHER_FEATURES,
    "profile_mean",
    "station_trend",
    "global_trend",
]
# LightGBM에 범주형으로 알려 줄 특징(대여소·자치구 번호). 숫자 크기에 뜻이 없으므로 순서형으로
# 다루지 않게 한다(validate.py가 학습 데이터셋을 만들 때 씀).
CATEGORICAL = ["station_code", "district_code"]
# v2: 행의 시각에 이미 공개돼 있던 가장 최근 반기로 계산한 수준 (docs/experiments.md v2)
LEVEL_FEATURES = ["station_recent_mean", "station_recent_ratio", "system_recent_ratio"]


# 한 번의 측정·학습에 쓰는 두 기간. rows와 history를 나눠 두어 '평가 행의 특징은 학습 기간 값으로만
# 계산'을 코드 구조로 강제한다. 예: v1 validation은 rows 2025-01-01~2025-07-01, history
# 2023-01-01~2025-01-01.
@dataclass(frozen=True)
class Window:
    """rows: 행을 뽑을 기간. history: 대여소 패턴·추세를 계산할 기간(학습 기간)."""

    rows: tuple[str, str]
    history: tuple[str, str]


# 'YYYY-MM-DD'를 months개월 옮긴 달의 1일로 돌려준다(일은 버림). 추세 기간 계산용.
def shift_months(day: str, months: int) -> str:
    year, month, _ = map(int, day.split("-"))
    total = year * 12 + (month - 1) + months
    return f"{total // 12:04d}-{total % 12 + 1:02d}-01"


# 대여소 추세 특징(station_trend)의 두 기간. 예: history 끝이 2025-01-01이면 마지막 6개월
# 2024-07~12, 그 1년 전 2023-07~12. 호출 위치: history_ctes, tests/test_model_frames.py.
def trend_windows(history: tuple[str, str]) -> tuple[tuple[str, str], tuple[str, str]]:
    """(마지막 6개월, 그 1년 전 같은 6개월). history 끝은 달의 첫날이어야 한다."""
    end = history[1]
    last = (shift_months(end, -6), end)
    prior = (shift_months(end, -18), shift_months(end, -12))
    return last, prior


# 학습 기간으로만 계산하는 CTE 묶음을 SQL 문자열로 만든다. 날짜는 코드가 만든 'YYYY-MM-DD' 문자열만
# 들어가므로 f-string으로 끼워 넣는다(사용자 입력이 아님).
def history_ctes(history: tuple[str, str]) -> str:
    """history 기간으로 계산하는 CTE들(codes, district_codes, profile, trend 등).

    학습·평가 행(_sql)과 서비스용 산출물(model/artifacts.py)이 같은 정의를 쓴다.
    """
    # CTE 내용:
    # - codes: 대여소ID를 정렬 순서로 0부터 번호 매긴 station_code(범주형 특징). dim_stations 전체
    #   기준이라 기간과 상관없이 같은 번호다
    # - district_codes: 자치구 이름 → 번호
    # - history: 학습 기간의 격자 행(0건 포함)과 쉬는 날 여부·시각
    # - profile: 대여소×쉬는 날 여부×시각 평균 = profile_mean(가장 큰 특징, gain 약 74~78%)이자
    #   기준선 B0
    # - hour_profile, global_profile, global_hour: 기준선 B1과 학습 기간에 없던 대여소의 대체값
    # - trend, global_trend: 마지막 6개월 / 1년 전 6개월 대여 비율(분모 0이면 결측). v1에서 기여가
    #   거의 0이었다
    hist_start, hist_end = history
    (last_start, last_end), (prior_start, prior_end) = trend_windows(history)
    return f"""
codes as (
    select station_id, row_number() over (order by station_id) - 1 as station_code,
           station_no, district, docks, lat, lon
    from dim_stations
),
district_codes as (
    select district, row_number() over (order by district) - 1 as district_code
    from (select distinct district from dim_stations where district is not null)
),
history as (
    select g.station_id, g.hour_start, g.rentals, h.is_offday, h.hour_of_day
    from int_station_hour_grid as g
    join dim_hours as h using (hour_start)
    where g.hour_start >= '{hist_start}' and g.hour_start < '{hist_end}'
),
profile as (
    select station_id, is_offday, hour_of_day, avg(rentals) as profile_mean
    from history group by all
),
hour_profile as (  -- B1: 요일 구분 없는 대여소×시간 평균
    select station_id, hour_of_day, avg(rentals) as hour_mean
    from history group by all
),
global_profile as (  -- 학습 기간에 없던 대여소를 기준선에서 채울 값
    select is_offday, hour_of_day, avg(rentals) as global_profile_mean
    from history group by all
),
global_hour as (
    select hour_of_day, avg(rentals) as global_hour_mean
    from history group by all
),
trend as (
    select
        station_id,
        sum(rentals) filter (where hour_start >= '{last_start}' and hour_start < '{last_end}')
            / nullif(sum(rentals) filter (
                where hour_start >= '{prior_start}' and hour_start < '{prior_end}'), 0)
            as station_trend
    from history group by 1
),
global_trend as (
    select
        sum(rentals) filter (where hour_start >= '{last_start}' and hour_start < '{last_end}')
            / nullif(sum(rentals) filter (
                where hour_start >= '{prior_start}' and hour_start < '{prior_end}'), 0)
            as global_trend
    from history
)
"""


# 반기 번호 = (연*12 + 월-1) // 6. 행이 속한 달 번호 n에서 1개월 이상 전에 끝난 가장 최근 반기는
# (n - 7) // 6 이다 (반기 h는 달 [6h, 6h+6)을 덮고, 6h+6 <= n-1 이어야 함).
# 반기 번호 계산을 파이썬으로 옮긴 것. 지금은 다른 코드에서 부르지 않고 아래 published_half와 SQL
# 식의 뜻을 설명하는 짝으로 남아 있다.
def half_of_month(year: int, month: int) -> int:
    """반기 번호. SQL의 (연*12 + 월-1) // 6과 같다."""
    return (year * 12 + month - 1) // 6


# 예: 2025년 1월 행 → 2024년 상반기, 2025년 2·3월 행 → 2024년 하반기(반기가 끝나고 1개월 뒤
# 공개된다고 가정, docs/experiments.md v2). 호출 위치: model/predict.py(서비스 예측의 수준 특징).
def published_half(year: int, month: int) -> int:
    """그 달에 이미 공개돼 있던(1개월 이상 전에 끝난) 가장 최근 반기 번호."""
    return (year * 12 + month - 1 - 7) // 6


# v2 수준 특징용 CTE: 대여소×반기의 시간당 평균(0건 포함)·합, 그리고 반기별 전체 대여 합. 격자
# 전체에서 계산하지만 아래 LEVEL_JOINS가 행보다 앞서 공개된 반기만 붙이므로 미래 값은 들어가지
# 않는다.
LEVEL_CTES = """
halves as (
    select station_id, (year(hour_start) * 12 + month(hour_start) - 1) // 6 as half,
           avg(rentals) as half_mean, sum(rentals) as half_sum
    from int_station_hour_grid group by all
),
system_halves as (
    select half, sum(half_sum) as total from halves group by 1
),
"""

# v3(T24): 행이 속한 달보다 앞선 달을 기준으로 한 가장 최근 대여소 정보 스냅샷의 정적 정보.
# 대여소ID·자치구 코드 번호(codes, district_codes)는 그대로 쓴다.
ASOF_STATION_CTES = """
station_months as (
    select c.station_id, m.month_key, s.district, s.docks, s.lat, s.lon
    from codes as c
    cross join (
        select distinct year(hour_start) * 12 + month(hour_start) - 1 as month_key
        from dim_hours
    ) as m
    asof left join (
        select station_no, district, docks, lat, lon,
               cast(left(snapshot, 4) as integer) * 12
                   + cast(right(snapshot, 2) as integer) - 1 as snapshot_key
        from stg_stations
    ) as s
        on s.station_no = c.station_no and m.month_key > s.snapshot_key
),
"""

# 행의 달(month_key)로 위 station_months를 붙이고, 그 시점의 자치구 이름으로 자치구 번호를 다시
# 찾는다.
ASOF_STATION_JOINS = """
left join station_months as sm
    on sm.station_id = g.station_id
    and sm.month_key = year(g.hour_start) * 12 + month(g.hour_start) - 1
left join district_codes as sdc on sdc.district = sm.district"""

# v2 수준 특징 세 개: 공개 반기의 대여소 평균, 그 평균 ÷ 1년 전 같은 반기 평균(반기 번호 - 2), 전체
# 대여 합의 같은 비율. 분모가 0이면 nullif로 결측.
LEVEL_SELECT = """,
    lh.half_mean::float as station_recent_mean,
    (lh.half_mean / nullif(lp.half_mean, 0))::float as station_recent_ratio,
    (sh.total / nullif(sp.total, 0))::float as system_recent_ratio"""

# 행의 달 번호에서 7을 빼고 6으로 나눈 반기 번호(= published_half)로 공개 반기를 고른다. 공개 반기
# 자료가 없으면(2023년 이전) left join이라 결측으로 남는다.
LEVEL_JOINS = """
left join halves as lh
    on lh.station_id = g.station_id
    and lh.half = (year(g.hour_start) * 12 + month(g.hour_start) - 1 - 7) // 6
left join halves as lp on lp.station_id = g.station_id and lp.half = lh.half - 2
left join system_halves as sh
    on sh.half = (year(g.hour_start) * 12 + month(g.hour_start) - 1 - 7) // 6
left join system_halves as sp on sp.half = sh.half - 2"""


# 측정 한 번의 SELECT 문 전체를 조립한다. 선택에 따라 v2 수준 특징(with_levels)과 v3 시점별 대여소
# 정보(asof_stations)를 붙인다. 호출 위치: load_frame.
def _sql(
    window: Window,
    only_active_stations: bool,
    with_levels: bool = False,
    asof_stations: bool = False,
) -> str:
    rows_start, rows_end = window.rows
    # 정적 특징(거치대·좌표·자치구)을 어느 별칭에서 읽을지 고른다. v3는 시점별(sm, sdc), v1·v2는
    # 최신 dim_stations(c, dc). 대여소 번호(station_code)는 어느 쪽이든 codes에서 읽는다.
    static = ("sm", "sdc") if asof_stations else ("c", "dc")
    extra_ctes = ("," + ASOF_STATION_CTES.rstrip().rstrip(",")) if asof_stations else ""
    extra_joins = (LEVEL_JOINS if with_levels else "") + (
        ASOF_STATION_JOINS if asof_stations else ""
    )
    # only_active_stations: 행 기간에 대여가 한 건 이상 있는 대여소만 남긴다. 평가 대상 규칙('해당
    # 기간에 대여가 한 건도 없는 대여소는 뺀다', docs/experiments.md v1)을 위한 것이다.
    active_filter = (
        f"""and g.station_id in (
            select station_id from int_station_hour_grid
            where hour_start >= '{rows_start}' and hour_start < '{rows_end}'
            group by 1 having sum(rentals) > 0)"""
        if only_active_stations
        else ""
    )
    # SELECT 내용: 목표 rentals, 조기 종료·날짜 구분에 쓰는 day_index(1970-01-01부터의 일 수),
    # FEATURES 순서의 특징, 기준선 값 b0·b1, (선택) 수준 특징. 연중 일자는 sin·cos로 바꿔 12월
    # 31일과 1월 1일이 가깝게 한다. 격자와 달력은 inner join이라 달력 밖 시간은 빠지고, 패턴·추세는
    # left join이라 학습 기간에 없던 대여소는 결측으로 남는다.
    return f"""
with {LEVEL_CTES if with_levels else ""}{history_ctes(window.history)}{extra_ctes}
select
    g.rentals::float as rentals,
    epoch(g.hour_start)::bigint // 86400 as day_index,
    c.station_code::float as station_code,
    {static[1]}.district_code::float as district_code,
    {static[0]}.docks::float as docks,
    {static[0]}.lat::float as lat,
    {static[0]}.lon::float as lon,
    h.hour_of_day::float as hour_of_day,
    h.day_of_week::float as day_of_week,
    h.is_offday::int::float as is_offday,
    h.is_holiday::int::float as is_holiday,
    h.month::float as month,
    sin(2 * pi() * h.day_of_year / 365.25)::float as doy_sin,
    cos(2 * pi() * h.day_of_year / 365.25)::float as doy_cos,
    h.temp_c::float as temp_c,
    h.rain_mm::float as rain_mm,
    h.wind_ms::float as wind_ms,
    h.humidity_pct::float as humidity_pct,
    h.is_new_snow::int::float as is_snow,
    p.profile_mean::float as profile_mean,
    t.station_trend::float as station_trend,
    (select global_trend from global_trend)::float as global_trend,
    coalesce(p.profile_mean, gp.global_profile_mean)::float as b0,
    coalesce(hp.hour_mean, gh.global_hour_mean)::float as b1{LEVEL_SELECT if with_levels else ""}
from int_station_hour_grid as g
join dim_hours as h using (hour_start)
join codes as c using (station_id)
left join district_codes as dc using (district)
left join profile as p
    on p.station_id = g.station_id and p.is_offday = h.is_offday
    and p.hour_of_day = h.hour_of_day
left join trend as t on t.station_id = g.station_id
left join hour_profile as hp
    on hp.station_id = g.station_id and hp.hour_of_day = h.hour_of_day
left join global_profile as gp on gp.is_offday = h.is_offday and gp.hour_of_day = h.hour_of_day
left join global_hour as gh on gh.hour_of_day = h.hour_of_day{extra_joins}
where g.hour_start >= '{rows_start}' and g.hour_start < '{rows_end}'
{active_filter}
"""


# 호출 위치: validate.py, validate_v2.py, validate_v3.py, final.py, tests(test_model_frames.py,
# test_model_levels.py, test_model_predict.py).
def load_frame(
    con: duckdb.DuckDBPyConnection,
    window: Window,
    only_active_stations: bool,
    with_levels: bool = False,
    asof_stations: bool = False,
) -> dict[str, np.ndarray]:
    """열 이름 → numpy 배열. float32(특징·목표), day_index는 int64.

    with_levels: LEVEL_FEATURES(v2)도 만든다. 격자 전체에서 행보다 앞선 반기만 읽는다.
    asof_stations: 거치대 수·좌표·자치구를 행보다 앞선 대여소 정보 스냅샷에서 가져온다(v3, T24).
    """
    # fetchnumpy는 열마다 numpy 배열을 준다. 결측이 있는 열은 masked array로 오므로 NaN으로 채운
    # float32로 바꾼다(LightGBM은 NaN을 결측으로 처리). 메모리를 아끼려고 특징은 float32로 두고,
    # day_index만 정수로 남긴다.
    sql = _sql(window, only_active_stations, with_levels, asof_stations)
    arrays = con.execute(sql).fetchnumpy()
    out: dict[str, np.ndarray] = {}
    for name in arrays:
        column = arrays[name]
        if hasattr(column, "filled"):  # 결측이 있는 열은 masked array로 온다
            column = column.astype(np.float32).filled(np.nan)
        out[name] = column if name == "day_index" else column.astype(np.float32, copy=False)
    return out


# 열 사전에서 features 순서대로 (행 수 × 특징 수) float32 행렬을 만든다. 행렬을 한 번에 할당하고
# 열을 채워 중간 복사를 줄인다. 호출 위치: validate·validate_v2·validate_v3, final.py,
# tests/test_model_predict.py.
def feature_matrix(frame: dict[str, np.ndarray], features: list[str]) -> np.ndarray:
    matrix = np.empty((len(frame["rentals"]), len(features)), dtype=np.float32)
    for i, name in enumerate(features):
        matrix[:, i] = frame[name]
    return matrix
