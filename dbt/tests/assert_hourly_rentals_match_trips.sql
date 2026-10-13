-- 시간별 집계의 대여·반납 합계를 대여 기록에서 따로 센 값과 비교한다. 다르면 행을 돌려준다.
-- 취소 조건을 is_cancelled 컬럼 대신 직접 다시 써서, 그 컬럼의 결함이 양쪽에 같이 숨지 않게 한다.
-- 집계가 비어 있어도(합계 NULL) 걸리도록 coalesce 후 비교한다.

-- 배경: 처음 테스트는 양쪽에 같은 is_cancelled를 쓰고 합계를 건수와 비교해, 사실 표가 비어 있어도
-- 통과했다(Codex 검토 라운드 1, 커밋 4cb3b33). 지금은 취소 규칙을 다시 쓰고, 반납도 비교하고, NULL도 불일치로
-- 본다.
with facts as (
    -- 집계 쪽: fct_station_hourly 전체의 대여·반납 합.
    select
        coalesce(sum(rentals), 0) as rentals,
        coalesce(sum(returns), 0) as returns
    from {{ ref('fct_station_hourly') }}
),

cancelled as (
    -- 취소 건수를 docs/data.md 규칙으로 직접 센다. 반납 대여소가 빈 기록은 취소가 아니므로 not null 조건을
    -- 명시한다.
    select count(*) as n
    from {{ ref('stg_trips') }}
    where duration_min = 0
        and return_station_id is not null
        and rent_station_id = return_station_id
),

trips as (
    -- 기록 쪽: 대여 = 전체 건수 - 취소. 반납 = 반납 대여소가 있는 건수(count는 NULL을 세지 않음) - 취소. 취소
    -- 기록은 모두 반납 대여소가 있으므로 반납에서도 그대로 뺀다.
    select
        count(*) - (select n from cancelled) as rentals,
        count(return_station_id) - (select n from cancelled) as returns
    from {{ ref('stg_trips') }}
)

-- is distinct from은 한쪽이 NULL이어도 참이 되므로 NULL을 불일치로 잡는다. 행이 나오면 테스트 실패.
select facts.rentals as fact_rentals, trips.rentals as trip_rentals,
       facts.returns as fact_returns, trips.returns as trip_returns
from facts, trips
where facts.rentals is distinct from trips.rentals
    or facts.returns is distinct from trips.returns
