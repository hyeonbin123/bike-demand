-- 대여소 한 곳 = 한 행. 키는 대여이력의 대여소ID(ST-xxx).
-- 대여나 반납에 한 번이라도 나온 곳을 모두 담는다. 반납만 있는 곳(13곳, 정비·수거 거점으로 보임)은
-- 대여 수가 0이다. 이름·위치는 그 대여소번호가 나온 가장 최근 스냅샷에서 가져온다(없으면 빈 값).

-- [모델 역할] marts 층의 대여소 차원 표(table). 한 행 = 대여이력에 나온 대여소 하나(station_id).
-- [왜 대여·반납 양쪽에서 모으나] 처음에는 대여 쪽만 모아서 반납에만 나오는 13곳(예: ST-48, 대여 0·반납
-- 6,094)이 fct_station_hourly에는 있는데 이 표에는 없었다(커밋 92affac). 그래서 두 쪽을 합치고, _marts.yml의
-- relationships 테스트로 fct의 모든 station_id가 여기 있는지 확인한다.
-- [누가 쓰나] int_station_hour_grid(total_rentals로 반납 전용 대여소 제외), model/frames.py(대여소ID·자치구
-- 코드 번호 매기기), serving/load.py(서비스 DB stations upsert).
-- [주의] 이름·좌표·거치대 수는 가장 최근 스냅샷 값이라 과거 학습 행에 그대로 붙이면 미래 정보가 섞인다.
-- 그래서 v3부터 학습 행의 정적 정보는 그 달보다 앞선 스냅샷(stg_stations)에서 따로 붙인다(docs/experiments.md
-- v3).

with trips as (

    select * from {{ ref('stg_trips') }}

),

rent_side as (

    select
        -- 대여 쪽: 대여소별 첫·마지막 대여 시각과 취소를 뺀 대여 수. 대여소ID와 대여소번호는 1:1로
        -- 확인됐으므로(docs/data.md) 번호는 any_value로 하나 가져온다.
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
        -- 반납 쪽: 반납 대여소가 비어 있는 기록(전체의 0.46%)은 어느 대여소인지 모르므로 뺀다.
        return_station_id as station_id,
        any_value(return_station_no) as station_no,
        count(*) filter (where not is_cancelled) as total_returns
    from trips
    where return_station_id is not null
    group by 1

),

trip_stations as (

    select
        -- 두 쪽을 full outer join으로 합친다. 한쪽에만 있는 대여소도 남고, 없는 쪽의 건수는 0으로 둔다.
        -- 번호는 대여 쪽을 먼저 쓴다.
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

    -- 대여소 정보 xlsx에는 대여소ID가 없고 번호만 있어 번호로 잇는다. 번호마다 가장 최근 반기 스냅샷 한 행만
    -- 남긴다(qualify로 창 함수 결과를 바로 거름).
    select *
    from {{ ref('stg_stations') }}
    qualify row_number() over (partition by station_no order by snapshot desc) = 1

)

-- 어느 스냅샷에도 없는 대여소는 left join이라 행은 남고 이름·자치구·좌표·거치대가 빈다(lat의 not_null
-- 테스트는 경고 수준). info_snapshot은 정보를 가져온 스냅샷 시점(YYYY-MM).
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
