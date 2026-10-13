-- 대여소 × 시간(정시 시작) = 한 행. 대여나 반납이 한 건 이상 있었던 시간만 담는다.
-- 대여 취소 표시가 된 기록은 빼고, 반납 대여소가 없는 기록은 반납 집계에서만 뺀다.
-- 기간 끝의 반납(2026-07-01 새벽 등)과 기간 앞 전날 대여의 반납은 한쪽만 잡힌다.

-- [모델 역할] marts 층의 사실 표(table). 예측 목표인 대여소별 시간당 대여 수(rentals)와 반납 수(returns)를
-- 담는다. 시간은 date_trunc로 내린 정시 시작 [h, h+1) 구간이다.
-- [누가 쓰나] int_station_hour_grid(0건 채운 격자의 원천), model/shortage_eval.py(작년 같은 계절의 반납
-- 프로파일), dbt 자료 테스트(합계 검사, 달력 범위 검사).
-- [검증] assert_hourly_rentals_match_trips가 이 표의 합계를 대여 기록에서 따로 센 값과 비교한다.

with trips as (

    select * from {{ ref('stg_trips') }}
    -- 0분 + 같은 대여소 반납(대여 취소·고장 교환으로 본 기록)은 대여소의 자전거를 실제로 줄이지 않으므로 수요
    -- 집계에서 뺀다(docs/data.md).
    where not is_cancelled

),

rentals as (

    select
        -- 대여: 빌린 대여소 × 빌린 시각이 든 시간.
        rent_station_id as station_id,
        date_trunc('hour', rented_at) as hour_start,
        count(*) as rentals
    from trips
    group by all

),

returns as (

    select
        -- 반납: 반납 대여소 × 반납 시각이 든 시간. 반납 대여소가 빈 기록은 대여에는 세고 여기서만 뺀다.
        return_station_id as station_id,
        date_trunc('hour', returned_at) as hour_start,
        count(*) as returns
    from trips
    where return_station_id is not null
    group by all

)

-- full outer join으로 대여만 있는 시간과 반납만 있는 시간을 모두 남기고, 없는 쪽은 0으로 채운다.
select
    station_id,
    hour_start,
    coalesce(rentals.rentals, 0) as rentals,
    coalesce(returns.returns, 0) as returns
from rentals
full outer join returns using (station_id, hour_start)
