"""측정용 데이터(대여소×시간 행)와 특징을 DuckDB warehouse에서 만든다. docs/experiments.md v1

기간은 모두 [시작, 끝) 반열린 구간의 날짜 문자열("YYYY-MM-DD")이다.
대여소 과거 패턴·추세는 `Window.history`(train 기간)에서만 계산해 평가 기간 값이 섞이지 않게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np

# 날씨 특징. M2는 이것만 뺀다.
WEATHER_FEATURES = ["temp_c", "rain_mm", "wind_ms", "humidity_pct", "is_snow"]
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
CATEGORICAL = ["station_code", "district_code"]
# v2: 행의 시각에 이미 공개돼 있던 가장 최근 반기로 계산한 수준 (docs/experiments.md v2)
LEVEL_FEATURES = ["station_recent_mean", "station_recent_ratio", "system_recent_ratio"]


@dataclass(frozen=True)
class Window:
    """rows: 행을 뽑을 기간. history: 대여소 패턴·추세를 계산할 기간(학습 기간)."""

    rows: tuple[str, str]
    history: tuple[str, str]


def shift_months(day: str, months: int) -> str:
    year, month, _ = map(int, day.split("-"))
    total = year * 12 + (month - 1) + months
    return f"{total // 12:04d}-{total % 12 + 1:02d}-01"


def trend_windows(history: tuple[str, str]) -> tuple[tuple[str, str], tuple[str, str]]:
    """(마지막 6개월, 그 1년 전 같은 6개월). history 끝은 달의 첫날이어야 한다."""
    end = history[1]
    last = (shift_months(end, -6), end)
    prior = (shift_months(end, -18), shift_months(end, -12))
    return last, prior


def history_ctes(history: tuple[str, str]) -> str:
    """history 기간으로 계산하는 CTE들(codes, district_codes, profile, trend 등).

    학습·평가 행(_sql)과 서비스용 산출물(model/artifacts.py)이 같은 정의를 쓴다.
    """
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
def half_of_month(year: int, month: int) -> int:
    """반기 번호. SQL의 (연*12 + 월-1) // 6과 같다."""
    return (year * 12 + month - 1) // 6


def published_half(year: int, month: int) -> int:
    """그 달에 이미 공개돼 있던(1개월 이상 전에 끝난) 가장 최근 반기 번호."""
    return (year * 12 + month - 1 - 7) // 6


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

ASOF_STATION_JOINS = """
left join station_months as sm
    on sm.station_id = g.station_id
    and sm.month_key = year(g.hour_start) * 12 + month(g.hour_start) - 1
left join district_codes as sdc on sdc.district = sm.district"""

LEVEL_SELECT = """,
    lh.half_mean::float as station_recent_mean,
    (lh.half_mean / nullif(lp.half_mean, 0))::float as station_recent_ratio,
    (sh.total / nullif(sp.total, 0))::float as system_recent_ratio"""

LEVEL_JOINS = """
left join halves as lh
    on lh.station_id = g.station_id
    and lh.half = (year(g.hour_start) * 12 + month(g.hour_start) - 1 - 7) // 6
left join halves as lp on lp.station_id = g.station_id and lp.half = lh.half - 2
left join system_halves as sh
    on sh.half = (year(g.hour_start) * 12 + month(g.hour_start) - 1 - 7) // 6
left join system_halves as sp on sp.half = sh.half - 2"""


def _sql(
    window: Window,
    only_active_stations: bool,
    with_levels: bool = False,
    asof_stations: bool = False,
) -> str:
    rows_start, rows_end = window.rows
    static = ("sm", "sdc") if asof_stations else ("c", "dc")
    extra_ctes = ("," + ASOF_STATION_CTES.rstrip().rstrip(",")) if asof_stations else ""
    extra_joins = (LEVEL_JOINS if with_levels else "") + (
        ASOF_STATION_JOINS if asof_stations else ""
    )
    active_filter = (
        f"""and g.station_id in (
            select station_id from int_station_hour_grid
            where hour_start >= '{rows_start}' and hour_start < '{rows_end}'
            group by 1 having sum(rentals) > 0)"""
        if only_active_stations
        else ""
    )
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
    (h.snow_cm > 0)::int::float as is_snow,
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
    sql = _sql(window, only_active_stations, with_levels, asof_stations)
    arrays = con.execute(sql).fetchnumpy()
    out: dict[str, np.ndarray] = {}
    for name in arrays:
        column = arrays[name]
        if hasattr(column, "filled"):  # 결측이 있는 열은 masked array로 온다
            column = column.astype(np.float32).filled(np.nan)
        out[name] = column if name == "day_index" else column.astype(np.float32, copy=False)
    return out


def feature_matrix(frame: dict[str, np.ndarray], features: list[str]) -> np.ndarray:
    matrix = np.empty((len(frame["rentals"]), len(features)), dtype=np.float32)
    for i, name in enumerate(features):
        matrix[:, i] = frame[name]
    return matrix
