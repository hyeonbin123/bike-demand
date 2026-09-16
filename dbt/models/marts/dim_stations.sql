-- 대여소 한 곳 = 한 행. 키는 대여이력의 대여소ID(ST-xxx).
-- 이름·위치는 그 대여소번호가 나온 가장 최근 스냅샷에서 가져온다(대여소 정보에 없는 곳은 빈 값).

with trip_stations as (

    select
        rent_station_id as station_id,
        any_value(rent_station_no) as station_no,
        min(rented_at) as first_rented_at,
        max(rented_at) as last_rented_at,
        count(*) filter (where not is_cancelled) as total_rentals
    from {{ ref('stg_trips') }}
    group by 1

),

latest_snapshot as (

    select *
    from {{ ref('stg_stations') }}
    qualify row_number() over (partition by station_no order by snapshot desc) = 1

)

select
    t.station_id,
    t.station_no,
    s.station_name,
    s.district,
    s.lat,
    s.lon,
    s.docks,
    s.snapshot as info_snapshot,
    t.first_rented_at,
    t.last_rented_at,
    t.total_rentals
from trip_stations as t
left join latest_snapshot as s using (station_no)
