-- 기상청 ASOS 서울(108) 시간 관측을 대여 시간(hour_start, 정시 시작 1시간)에 맞춘다.
-- 기온·풍속·습도: 관측 시각 t의 순간값을 hour_start = t - 1시간에 붙인다(구간 끝 값).
-- 강수(rn): 4~10월은 직전 1시간 누적이 매시 보고되므로 t - 1시간에 붙인다.
--   11~3월은 3시간 간격(0·3·6…시)으로 직전 3시간 누적만 보고된다(실제 자료에서 확인, 그 밖의 시각은
--   항상 빈 값). 그래서 그 3시간 누적을 앞선 세 시간에 1/3씩 나눈다. 빈 값은 비가 오지 않은 것으로 본다.
-- 눈: 적설(dsnw)은 쌓인 깊이라 눈이 그친 뒤에도 남는다. 서비스 예보의 SNO(1시간 신적설)와 맞추려고
--   3시간 신적설(hr3Fhsc, 0·3·6…시 보고)이 0보다 크면 그 앞 세 시간을 모두 "새로 눈이 옴"으로 둔다.
-- 겨울/여름 경계(3월 31일 밤·10월 31일 밤)의 몇 시간은 규칙이 바뀌는 시각이라 조금 어긋날 수 있다.

with source as (

    select * from {{ source('bronze', 'asos_hourly') }}

),

observations as (

    select
        strptime(observed_at, '%Y-%m-%d %H:%M') as observed_at,
        cast(temp_c as double) as temp_c,
        cast(wind_ms as double) as wind_ms,
        cast(humidity_pct as double) as humidity_pct,
        coalesce(cast(rain_mm as double), 0) as rain_mm,
        coalesce(cast(new_snow_3h_cm as double), 0) as new_snow_3h_cm,
        coalesce(cast(snow_cm as double), 0) as snow_depth_cm
    from source

),

hours as (

    select
        *,
        observed_at - interval 1 hour as hour_start,
        -- 이 시간을 덮는 3시간 보고 시각: t 이후 가장 가까운 3의 배수 시
        observed_at + to_hours(cast((3 - hour(observed_at) % 3) % 3 as integer)) as three_hour_at,
        month(observed_at) in (11, 12, 1, 2, 3) as rain_is_three_hourly
    from observations

)

select
    h.hour_start,
    h.observed_at,
    h.temp_c,
    h.wind_ms,
    h.humidity_pct,
    case
        when h.rain_is_three_hourly then coalesce(c.rain_mm, 0) / 3
        else h.rain_mm
    end as rain_mm,
    h.rain_mm as rain_mm_reported,
    coalesce(c.new_snow_3h_cm, 0) > 0 as is_new_snow,
    h.snow_depth_cm
from hours as h
left join observations as c on c.observed_at = h.three_hour_at
