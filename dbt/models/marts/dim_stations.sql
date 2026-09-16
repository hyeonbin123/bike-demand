-- 대여소 한 곳 = 한 행. 키는 대여이력의 대여소ID(ST-xxx).
-- 대여나 반납에 한 번이라도 나온 곳을 모두 담는다. 반납만 있는 곳(13곳, 정비·수거 거점으로 보임)은
-- 대여 수가 0이다. 이름·위치는 그 대여소번호가 나온 가장 최근 스냅샷에서 가져온다(없으면 빈 값).

with trips as (

    select * from {{ ref('stg_trips') }}

),

rent_side as (

    select
        rent_station_id as station_id,
        any_value(rent_station_no) as station_no,
        min(rented_at) as first_rented_at,
        max(rented_at) as last_rented_at,
        count(*) filter (where not is_cancelled) as total_rentals
    from trips
    group by 1

),

return_side as (

    select
        return_station_id as station_id,
        any_value(return_station_no) as station_no,
        count(*) filter (where not is_cancelled) as total_returns
    from trips
    where return_station_id is not null
    group by 1

),

trip_stations as (

    select
        station_id,
        coalesce(r.station_no, t.station_no) as station_no,
        r.first_rented_at,
        r.last_rented_at,
        coalesce(r.total_rentals, 0) as total_rentals,
        coalesce(t.total_returns, 0) as total_returns
    from rent_side as r
    full outer join return_side as t using (station_id)

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
    t.total_rentals,
    t.total_returns
from trip_stations as t
left join latest_snapshot as s using (station_no)
