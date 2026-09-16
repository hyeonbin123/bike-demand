-- 대여소 × 시간 격자 (대여 0건인 시간 포함). 학습·평가 대상 행을 만든다. docs/experiments.md v1
-- 대여소가 운영 중이었던 기간만: 대여나 반납이 처음 있었던 날 0시부터 마지막으로 있었던 날 23시까지.
-- 반납만 있는 대여소(대여 0건)는 뺀다.

{{ config(materialized='table') }}

with hourly as (

    select * from {{ ref('fct_station_hourly') }}

),

active as (

    select
        h.station_id,
        date_trunc('day', min(h.hour_start)) as first_day,
        date_trunc('day', max(h.hour_start)) as last_day
    from hourly as h
    inner join {{ ref('dim_stations') }} as d using (station_id)
    where d.total_rentals > 0
    group by 1

),

grid as (

    select
        a.station_id,
        unnest(generate_series(a.first_day, a.last_day + interval 23 hour, interval 1 hour)) as hour_start
    from active as a

)

select
    g.station_id,
    g.hour_start,
    coalesce(h.rentals, 0)::integer as rentals
from grid as g
left join hourly as h using (station_id, hour_start)
