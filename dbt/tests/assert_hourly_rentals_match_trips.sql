-- 시간별 집계의 대여 합계가 취소를 뺀 대여 건수와 같아야 한다. 다르면 행을 돌려준다.
with facts as (
    select sum(rentals) as rentals from {{ ref('fct_station_hourly') }}
),

trips as (
    select count(*) as rentals from {{ ref('stg_trips') }} where not is_cancelled
)

select facts.rentals as fact_rentals, trips.rentals as trip_rentals
from facts, trips
where facts.rentals <> trips.rentals
