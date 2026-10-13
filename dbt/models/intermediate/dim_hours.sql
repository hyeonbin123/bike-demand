-- 시간(정시 시작) = 한 행. 달력 정보와 그 시간의 관측 날씨.
-- 범위는 대여 기록이 있는 기간 전체(2023-01-01 00시 ~ 2026-06-30 23시, var로 바꿀 수 있음).
-- 날씨가 빠진 시간은 행을 남기고 날씨 값만 비운다.

{{ config(materialized='table') }}

-- [모델 역할] intermediate 층의 시간 차원 표. 대여소×시간 격자(int_station_hour_grid)에 hour_start로 붙여
-- 학습·평가 특징의 달력·날씨 부분을 공급한다.
-- [누가 쓰나] model/frames.py(학습·평가 행과 대여소 패턴 계산), model/shortage_eval.py(작년 같은 계절
-- 프로파일), scripts/check_prediction_level.py(수준 진단). frames.py는 이 표와 격자를 inner join하므로 이
-- 표에 없는 시간은 학습 행이 되지 못한다.
-- [materialized] dbt_project.yml에 intermediate 층 기본값이 없어(dbt 기본은 view) 위 config로 table을
-- 지정한다.
-- [기간] calendar_start·calendar_end 기본값은 dbt_project.yml vars 한 곳에만 둔다(커밋 29a4094). 새 반기를
-- 넣을 때 refresh_history DAG가 원본 최신 달 말일을 calendar_end로 넘긴다(커밋 229e7c6).

with hours as (

    -- DuckDB generate_series는 끝 값을 포함한 목록을 돌려주고 unnest가 그것을 행으로 편다. 끝을 calendar_end
    -- 0시 + 23시간으로 잡아 마지막 날의 23시까지 한 시간 간격으로 만든다.
    select unnest(generate_series(
        timestamp '{{ var("calendar_start") }}',
        timestamp '{{ var("calendar_end") }}' + interval 23 hour,
        interval 1 hour
    )) as hour_start

),

-- 한국 공휴일 시드(dbt/seeds/kr_holidays.csv). holidays 패키지(KR)로 만든 것이라 대체·임시공휴일과 선거일이
-- 들어 있다(src/bike_demand/holidays_seed.py가 생성).
holidays as (

    select cast(holiday_date as date) as holiday_date, holiday_name
    from {{ ref('kr_holidays') }}

)

select
    -- 달력 특징: 시각(0~23), ISO 요일, 달, 연중 일자. 모델은 연중 일자를 sin·cos로 바꿔 쓴다(frames.py). 모두
    -- 예측 시점에 미리 알 수 있는 값이라 서비스에서도 그대로 만들 수 있다(docs/experiments.md v1 특징 규칙).
    h.hour_start,
    cast(h.hour_start as date) as day,
    hour(h.hour_start) as hour_of_day,
    isodow(h.hour_start) as day_of_week,  -- 1 월요일 ~ 7 일요일
    month(h.hour_start) as month,
    dayofyear(h.hour_start) as day_of_year,
    -- is_offday(쉬는 날) = 토·일(ISO 요일 6·7) 또는 공휴일. 기준선 B0의 요일구분과 핵심 특징
    -- profile_mean(대여소×쉬는 날 여부×시각 평균)이 이 값으로 나뉜다.
    hol.holiday_name is not null as is_holiday,
    isodow(h.hour_start) >= 6 or hol.holiday_name is not null as is_offday,
    hol.holiday_name,
    -- 관측 날씨(ASOS 서울 108): 기온, 1시간 강수, 풍속, 습도, 새로 눈이 옴 여부. 학습·validation·test는 이
    -- 관측값을 쓰고, 서비스 예측은 같은 자리에 단기예보 값을 넣는다(관측과 예보의 차이는 v4에서 측정).
    -- snow_depth_cm(쌓인 눈 깊이)은 참고로 남겨 둔 값이고 모델 특징에는 is_new_snow를 쓴다(v3 날씨 정의
    -- 수정).
    w.temp_c,
    w.rain_mm,
    w.wind_ms,
    w.humidity_pct,
    w.is_new_snow,
    w.snow_depth_cm
-- 두 조인 모두 left join: 공휴일이 아닌 날, 관측이 빠진 시간도 행은 남긴다. 날씨가 없는 시간은 값이 비고
-- LightGBM이 결측으로 처리한다.
from hours as h
left join holidays as hol on cast(h.hour_start as date) = hol.holiday_date
left join {{ ref('stg_weather_hourly') }} as w using (hour_start)
