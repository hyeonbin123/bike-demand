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
           district, docks, lat, lon
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


def _sql(window: Window, only_active_stations: bool) -> str:
    rows_start, rows_end = window.rows
    active_filter = (
        f"""and g.station_id in (
            select station_id from int_station_hour_grid
            where hour_start >= '{rows_start}' and hour_start < '{rows_end}'
            group by 1 having sum(rentals) > 0)"""
        if only_active_stations
        else ""
    )
    return f"""
with {history_ctes(window.history)}
select
    g.rentals::float as rentals,
    epoch(g.hour_start)::bigint // 86400 as day_index,
    c.station_code::float as station_code,
    dc.district_code::float as district_code,
    c.docks::float as docks,
    c.lat::float as lat,
    c.lon::float as lon,
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
    coalesce(hp.hour_mean, gh.global_hour_mean)::float as b1
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
left join global_hour as gh on gh.hour_of_day = h.hour_of_day
where g.hour_start >= '{rows_start}' and g.hour_start < '{rows_end}'
{active_filter}
"""


def load_frame(
    con: duckdb.DuckDBPyConnection, window: Window, only_active_stations: bool
) -> dict[str, np.ndarray]:
    """열 이름 → numpy 배열. float32(특징·목표), day_index는 int64."""
    arrays = con.execute(_sql(window, only_active_stations)).fetchnumpy()
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
