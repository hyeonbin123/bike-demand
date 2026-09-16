{{ config(materialized="table") }}

-- 대여 한 건 = 한 행. 타입을 바꾸고, 완전히 같은 기록은 하나만 남기고,
-- 빌린 자리에서 0분 만에 돌려준 기록(대여 취소·고장 교환으로 봄)에 표시를 단다. docs/data.md 참고

with source as (

    select * from {{ source('bronze', 'trips') }}

),

typed as (

    select
        bike_no,
        strptime(rented_at, '%Y-%m-%d %H:%M:%S') as rented_at,
        rent_station_id,
        cast(rent_station_no as integer) as rent_station_no,
        strptime(returned_at, '%Y-%m-%d %H:%M:%S') as returned_at,
        return_station_id,
        cast(return_station_no as integer) as return_station_no,
        cast(duration_min as integer) as duration_min,
        cast(distance_m as double) as distance_m,
        user_type,
        bike_type,
        month as source_month
    from source

),

deduplicated as (

    select distinct on (bike_no, rented_at, rent_station_id, returned_at, return_station_id) *
    from typed

)

select
    *,
    -- 반납 대여소가 빈 행은 비교 결과가 NULL이 되어 `not is_cancelled` 필터에서 빠지므로 false로 둔다.
    coalesce(duration_min = 0 and rent_station_id = return_station_id, false) as is_cancelled
from deduplicated
