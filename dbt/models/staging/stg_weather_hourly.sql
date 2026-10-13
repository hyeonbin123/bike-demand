-- 기상청 ASOS 서울(108) 시간 관측을 대여 시간(hour_start, 정시 시작 1시간)에 맞춘다.
-- 기온·풍속·습도: 관측 시각 t의 순간값을 hour_start = t - 1시간에 붙인다(구간 끝 값).
-- 강수(rn): 4~10월은 직전 1시간 누적이 매시 보고되므로 t - 1시간에 붙인다.
--   11~3월은 3시간 간격(0·3·6…시)으로 직전 3시간 누적만 보고된다(실제 자료에서 확인, 그 밖의 시각은
--   항상 빈 값). 그래서 그 3시간 누적을 앞선 세 시간에 1/3씩 나눈다. 빈 값은 비가 오지 않은 것으로 본다.
-- 눈: 적설(dsnw)은 쌓인 깊이라 눈이 그친 뒤에도 남는다. 서비스 예보의 SNO(1시간 신적설)와 맞추려고
--   3시간 신적설(hr3Fhsc, 0·3·6…시 보고)이 0보다 크면 그 앞 세 시간을 모두 "새로 눈이 옴"으로 둔다.
-- 겨울/여름 경계(3월 31일 밤·10월 31일 밤)의 몇 시간은 규칙이 바뀌는 시각이라 조금 어긋날 수 있다.

-- [모델 역할] staging 층(view). 입력은 ingest/weather.py가 받은 ASOS 시간자료 bronze Parquet, 출력은
-- hour_start 하나에 한 행(_weather.yml의 unique 테스트). dim_hours가 이 표를 붙인다.
-- [이력] 겨울 강수 나누기와 신적설 기준 눈은 v3 측정 전에 바꾼 정의다(docs/experiments.md v3 측정 전 보충,
-- 커밋 a909857). 그 전에는 겨울 3시간 누적을 한 시간에만 붙였고(비 온 시간 320 → 바꾼 뒤 957), 눈은 쌓인
-- 깊이로 켜서 그친 뒤에도 켜져 있었다(1,188시간 → 318시간).
-- [검증] _weather.yml 단위 테스트가 1월의 3시간 누적 0.9mm가 세 시간에 0.3씩 나뉘고 신적설이 세 시간 모두에
-- 켜지는지, 7월 값은 그대로인지 확인한다.

with source as (

    select * from {{ source('bronze', 'asos_hourly') }}

),

observations as (

    select
        -- 관측 시각 문자열(분까지)을 timestamp로 바꾸고, 수치 컬럼을 double로 바꾼다.
        strptime(observed_at, '%Y-%m-%d %H:%M') as observed_at,
        cast(temp_c as double) as temp_c,
        cast(wind_ms as double) as wind_ms,
        cast(humidity_pct as double) as humidity_pct,
        -- ASOS는 비가 오지 않은 시간의 강수를 빈 값으로 준다(docs/data.md). 그래서 빈 강수·신적설·적설은
        -- 0으로 본다. 기온·풍속·습도의 빈 값은 그대로 둔다(진짜 결측).
        coalesce(cast(rain_mm as double), 0) as rain_mm,
        coalesce(cast(new_snow_3h_cm as double), 0) as new_snow_3h_cm,
        coalesce(cast(snow_cm as double), 0) as snow_depth_cm
    from source

),

hours as (

    select
        *,
        -- 시각 t에 보고된 강수는 직전 1시간(t-1 ~ t)의 누적이므로 대여 시간 hour_start = t - 1시간에 붙인다.
        -- 기온·풍속·습도도 같은 행을 따라가 그 시간의 끝 시각 순간값이 된다(위 머리 설명).
        observed_at - interval 1 hour as hour_start,
        -- 이 시간을 덮는 3시간 보고 시각: t 이후 가장 가까운 3의 배수 시
        observed_at + to_hours(cast((3 - hour(observed_at) % 3) % 3 as integer)) as three_hour_at,
        -- 11~3월은 강수가 3시간 간격으로만 보고되는 달이다(실제 자료에서 확인한 규칙).
        month(observed_at) in (11, 12, 1, 2, 3) as rain_is_three_hourly
    from observations

)

select
    h.hour_start,
    h.observed_at,
    h.temp_c,
    h.wind_ms,
    h.humidity_pct,
    -- 겨울: 이 시간을 덮는 3시간 보고(c)의 누적값을 3으로 나눠 세 시간에 똑같이 나눈다. 세 시간의 합이 원래
    -- 누적과 같아 겨울 강수 총량은 바뀌지 않는다. 여름: 그 시각의 1시간 값을 그대로 쓴다.
    case
        when h.rain_is_three_hourly then coalesce(c.rain_mm, 0) / 3
        else h.rain_mm
    end as rain_mm,
    -- 그 시각에 실제로 보고된 원래 값도 남겨 두어 나눈 값과 비교·검증할 수 있게 한다.
    h.rain_mm as rain_mm_reported,
    -- 덮는 3시간 보고의 신적설이 0보다 크면 그 세 시간 모두 새로 눈이 온 시간으로 본다. 서비스 단기예보의
    -- SNO(1시간 신적설)와 뜻을 맞추려는 정의다.
    coalesce(c.new_snow_3h_cm, 0) > 0 as is_new_snow,
    h.snow_depth_cm
from hours as h
-- 같은 관측 표를 다시 붙여 이 시간을 덮는 3시간 보고 행을 찾는다. 보고 행이 없으면(자료 끝 등) left join이라
-- 값이 비고, 위 coalesce가 0으로 바꾼다.
left join observations as c on c.observed_at = h.three_hour_at
