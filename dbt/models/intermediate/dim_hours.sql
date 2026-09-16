-- 시간(정시 시작) = 한 행. 달력 정보와 그 시간의 관측 날씨.
-- 범위는 대여 기록이 있는 기간 전체(2023-01-01 00시 ~ 2026-06-30 23시, var로 바꿀 수 있음).
-- 날씨가 빠진 시간은 행을 남기고 날씨 값만 비운다.

{{ config(materialized='table') }}

with hours as (

    select unnest(generate_series(
        timestamp '{{ var("calendar_start", "2023-01-01") }}',
        timestamp '{{ var("calendar_end", "2026-06-30") }}' + interval 23 hour,
        interval 1 hour
    )) as hour_start

),

holidays as (

    select cast(holiday_date as date) as holiday_date, holiday_name
    from {{ ref('kr_holidays') }}

)

select
    h.hour_start,
    cast(h.hour_start as date) as day,
    hour(h.hour_start) as hour_of_day,
    isodow(h.hour_start) as day_of_week,  -- 1 월요일 ~ 7 일요일
    month(h.hour_start) as month,
    dayofyear(h.hour_start) as day_of_year,
    hol.holiday_name is not null as is_holiday,
    isodow(h.hour_start) >= 6 or hol.holiday_name is not null as is_offday,
    hol.holiday_name,
    w.temp_c,
    w.rain_mm,
    w.wind_ms,
    w.humidity_pct,
    w.is_new_snow,
    w.snow_depth_cm
from hours as h
left join holidays as hol on cast(h.hour_start as date) = hol.holiday_date
left join {{ ref('stg_weather_hourly') }} as w using (hour_start)
