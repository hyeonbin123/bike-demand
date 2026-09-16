-- 대여소 × 시간(정시 시작) = 한 행. 대여나 반납이 한 건 이상 있었던 시간만 담는다.
-- 대여 취소 표시가 된 기록은 빼고, 반납 대여소가 없는 기록은 반납 집계에서만 뺀다.
-- 기간 끝의 반납(2026-07-01 새벽 등)과 기간 앞 전날 대여의 반납은 한쪽만 잡힌다.

with trips as (

    select * from {{ ref('stg_trips') }}
    where not is_cancelled

),

rentals as (

    select
        rent_station_id as station_id,
        date_trunc('hour', rented_at) as hour_start,
        count(*) as rentals
    from trips
    group by all

),

returns as (

    select
        return_station_id as station_id,
        date_trunc('hour', returned_at) as hour_start,
        count(*) as returns
    from trips
    where return_station_id is not null
    group by all

)

select
    station_id,
    hour_start,
    coalesce(rentals.rentals, 0) as rentals,
    coalesce(returns.returns, 0) as returns
from rentals
full outer join returns using (station_id, hour_start)
