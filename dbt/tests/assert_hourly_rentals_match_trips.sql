-- 시간별 집계의 대여·반납 합계를 대여 기록에서 따로 센 값과 비교한다. 다르면 행을 돌려준다.
-- 취소 조건을 is_cancelled 컬럼 대신 직접 다시 써서, 그 컬럼의 결함이 양쪽에 같이 숨지 않게 한다.
-- 집계가 비어 있어도(합계 NULL) 걸리도록 coalesce 후 비교한다.

with facts as (
    select
        coalesce(sum(rentals), 0) as rentals,
        coalesce(sum(returns), 0) as returns
    from {{ ref('fct_station_hourly') }}
),

cancelled as (
    select count(*) as n
    from {{ ref('stg_trips') }}
    where duration_min = 0
        and return_station_id is not null
        and rent_station_id = return_station_id
),

trips as (
    select
        count(*) - (select n from cancelled) as rentals,
        count(return_station_id) - (select n from cancelled) as returns
    from {{ ref('stg_trips') }}
)

select facts.rentals as fact_rentals, trips.rentals as trip_rentals,
       facts.returns as fact_returns, trips.returns as trip_returns
from facts, trips
where facts.rentals is distinct from trips.rentals
    or facts.returns is distinct from trips.returns
