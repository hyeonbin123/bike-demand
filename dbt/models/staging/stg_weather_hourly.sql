-- 기상청 ASOS 서울(108) 시간 관측. 관측 시각 = 한 행.
-- 1시간 강수량(rn)은 관측 시각 직전 1시간 동안 내린 양이다(예: 13:00 행 = 12:00~13:00).
-- 대여 시간(hour_start 12:00, 12:00~13:00)과 맞추기 위해 hour_start = 관측 시각 - 1시간으로 둔다.
-- 기온·풍속·습도는 그 시각의 순간값이라 구간 끝 값을 쓰게 되지만 한 시간 차이라 그대로 둔다.
-- rn이 비어 있으면 비가 오지 않은 것(기상청 제공 방식)으로 보고 0으로 채운다.

with source as (

    select * from {{ source('bronze', 'asos_hourly') }}

)

select
    strptime(observed_at, '%Y-%m-%d %H:%M') - interval 1 hour as hour_start,
    strptime(observed_at, '%Y-%m-%d %H:%M') as observed_at,
    cast(temp_c as double) as temp_c,
    coalesce(cast(rain_mm as double), 0) as rain_mm,
    cast(wind_ms as double) as wind_ms,
    cast(humidity_pct as double) as humidity_pct,
    coalesce(cast(snow_cm as double), 0) as snow_cm
from source
