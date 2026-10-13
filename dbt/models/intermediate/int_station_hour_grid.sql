-- 대여소 × 시간 격자 (대여 0건인 시간 포함). 학습·평가 대상 행을 만든다. docs/experiments.md v1
-- 대여소가 운영 중이었던 기간만: 대여나 반납이 처음 있었던 날 0시부터 마지막으로 있었던 날 23시까지.
-- 반납만 있는 대여소(대여 0건)는 뺀다.
-- 자료의 마지막 날 밤에 빌린 자전거는 다음 날 반납돼 fct에 달력 밖(자료 끝 다음 날)의 반납 행이 생긴다.
-- 그 하루는 대여 자료가 없는 날이므로 격자에 넣지 않는다: 달력 끝(calendar_end) 다음 날부터의 행은 보지 않는다.
-- (넘으면 대여 0건인 하루가 다음 반기의 평균 0으로 집계돼 수준 특징을 망가뜨린다, T53)

{{ config(materialized='table') }}

-- [모델 역할] 학습·평가의 기본 행 집합. 한 행 = (대여소, 정시 시작 1시간), 대여가 0건인 시간도 rentals =
-- 0으로 들어 있다.
-- [왜 0건을 채우나] fct_station_hourly는 대여나 반납이 있었던 시간만 담는다. 평가 지표(대여소×시간 MAE)는
-- 0건인 시간도 포함해야 하고(docs/experiments.md v1 평가 대상 행), 대여소 패턴(profile_mean)과 반기 수준
-- 평균도 0건 시간을 분모에 넣어야 실제 시간당 평균이 된다.
-- [누가 쓰나] model/frames.py(학습·평가 행, 패턴·수준 특징), model/final.py(서비스 학습 기간의 끝을 이 표의
-- 마지막 시간에서 계산), model/shortage_eval.py, scripts/check_prediction_level.py.

with hourly as (

    select * from {{ ref('fct_station_hourly') }}

),

active as (

    -- 대여소마다 운영 기간을 정한다: 대여나 반납이 처음 있었던 날 0시부터 마지막으로 있었던 날의 0시까지(아래
    -- grid에서 23시까지 늘림). 평가 대상 행은 대여소가 운영 중이었던 시간만이라서(docs/experiments.md v1 평가
    -- 대상 행) 이 기간 밖은 만들지 않는다.
    select
        h.station_id,
        date_trunc('day', min(h.hour_start)) as first_day,
        date_trunc('day', max(h.hour_start)) as last_day
    from hourly as h
    inner join {{ ref('dim_stations') }} as d using (station_id)
    -- 반납만 있는 대여소(13곳, 정비·수거 거점으로 보임)는 대여 수요 예측 대상이 아니어서 뺀다.
    where d.total_rentals > 0
        -- 달력 끝 다음 날의 반납 행이 운영 기간의 마지막 날을 하루 늘리지 않게 거른다(위 머리 설명, 커밋
        -- e5dd59b).
        and h.hour_start < timestamp '{{ var("calendar_end") }}' + interval 1 day
    group by 1

),

grid as (

    -- 운영 기간의 모든 시간을 한 시간 간격으로 펼친다(대여소마다 generate_series 목록을 unnest).
    select
        a.station_id,
        unnest(generate_series(a.first_day, a.last_day + interval 23 hour, interval 1 hour)) as hour_start
    from active as a

)

-- 펼친 격자에 실제 시간별 대여 수를 left join으로 붙이고, 기록이 없는 시간은 0으로 채운다. 반납 수는 여기
-- 담지 않는다(필요한 곳은 fct_station_hourly를 따로 붙임).
select
    g.station_id,
    g.hour_start,
    coalesce(h.rentals, 0)::integer as rentals
from grid as g
left join hourly as h using (station_id, hour_start)
