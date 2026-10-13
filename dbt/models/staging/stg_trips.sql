{{ config(materialized="table") }}

-- 대여 한 건 = 한 행. 타입을 바꾸고, 완전히 같은 기록은 하나만 남기고,
-- 빌린 자리에서 0분 만에 돌려준 기록(대여 취소·고장 교환으로 봄)에 표시를 단다. docs/data.md 참고

-- [모델 역할] staging 층의 대여 기록 정제. 입력은 ingest/trips.py가 컬럼 이름만 바꿔 저장한 월별 문자열
-- Parquet(약 1억 4천만 행)이다.
-- [materialized] staging 기본(view)과 달리 위 config로 table을 만든다. fct_station_hourly, dim_stations, 합계
-- 검사 테스트가 모두 이 결과를 읽으므로, table로 두면 이들이 타입 변환·중복 제거를 각자 다시 하지 않는다.
-- [검증] _staging.yml의 단위 테스트가 중복 제거와 0분 기록 세 경우(같은 대여소, 다른 대여소, 반납 대여소
-- 없음)를 확인한다.

with source as (

    select * from {{ source('bronze', 'trips') }}

),

typed as (

    select
        bike_no,
        -- bronze의 모든 값은 문자열이다. 시각은 원본 형식(초 단위)으로 해석한다(형식 오류 0건, docs/data.md).
        strptime(rented_at, '%Y-%m-%d %H:%M:%S') as rented_at,
        rent_station_id,
        -- 대여소번호의 앞자리 0을 떼고 정수로 바꿔 대여소 정보(stg_stations)의 번호와 맞춘다.
        cast(rent_station_no as integer) as rent_station_no,
        strptime(returned_at, '%Y-%m-%d %H:%M:%S') as returned_at,
        return_station_id,
        cast(return_station_no as integer) as return_station_no,
        cast(duration_min as integer) as duration_min,
        cast(distance_m as double) as distance_m,
        user_type,
        bike_type,
        -- month는 bronze 폴더 이름(month=YYYY-MM)에서 hive 분할로 읽힌 값이다. 어느 월 파일에서 온 행인지
        -- 남긴다.
        month as source_month
    from source

),

deduplicated as (

    -- 자전거·대여 시각·대여소·반납 시각·반납 대여소가 모두 같은 행은 한 번만 남긴다(원본에 10,252건,
    -- docs/data.md).
    select distinct on (bike_no, rented_at, rent_station_id, returned_at, return_station_id) *
    from typed

)

select
    *,
    -- 취소 규칙: 이용시간이 정확히 0분이고 빌린 대여소에 반납한 기록. 다른 대여소에 0분으로 반납한 기록은
    -- 이동이 있었던 것으로 보고 남긴다(docs/data.md).
    -- 반납 대여소가 빈 행은 비교 결과가 NULL이 되어 `not is_cancelled` 필터에서 빠지므로 false로 둔다.
    coalesce(duration_min = 0 and rent_station_id = return_station_id, false) as is_cancelled
from deduplicated
