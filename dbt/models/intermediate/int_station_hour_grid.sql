-- 대여소 × 시간 격자 (대여 0건인 시간 포함). 학습·평가 대상 행을 만든다. docs/experiments.md v1
-- 대여소가 운영 중이었던 기간만: 대여나 반납이 처음 있었던 날 0시부터 마지막으로 있었던 날 23시까지.
-- 반납만 있는 대여소(대여 0건)는 뺀다.
-- 자료의 마지막 날 밤에 빌린 자전거는 다음 날 반납돼 fct에 달력 밖(자료 끝 다음 날)의 반납 행이 생긴다.
-- 그 하루는 대여 자료가 없는 날이므로 격자에 넣지 않는다: 달력 끝(calendar_end) 다음 날부터의 행은 보지 않는다.
-- (넘으면 대여 0건인 하루가 다음 반기의 평균 0으로 집계돼 수준 특징을 망가뜨린다, T53)

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
        and h.hour_start < timestamp '{{ var("calendar_end", "2026-06-30") }}' + interval 1 day
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
