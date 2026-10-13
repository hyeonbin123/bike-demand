-- 대여소 정보 반기 스냅샷. 시점 × 대여소번호 = 한 행

-- [모델 역할] staging 층(view). ingest/stations.py가 반기 xlsx 8개를 문자열 그대로 모은 bronze Parquet에
-- 타입을 입힌다. (snapshot, station_no) 조합이 유일한지는 _staging.yml 테스트가 확인한다.
-- [누가 쓰나] dim_stations(번호마다 최신 스냅샷), model/frames.py(v3부터 행이 속한 달보다 앞선 스냅샷의
-- 거치대·좌표·자치구를 asof join으로 붙임, docs/experiments.md v3).

select
    -- xlsx의 대여소번호는 숫자(301), 대여이력은 앞자리 0이 붙은 문자열(00301)이다. 양쪽 모두 정수로 바꿔 같은
    -- 값으로 조인되게 한다(stg_trips도 정수로 바꿈).
    cast(station_no as integer) as station_no,
    station_name,
    district,
    address,
    cast(lat as double) as lat,
    cast(lon as double) as lon,
    cast(installed_at as timestamp) as installed_at,
    -- 거치대 수 = LCD 거치대 수 + QR 거치대 수. xlsx가 두 방식의 거치대 수를 다른 칸에 적으므로 더해서 하나로
    -- 만들고, 빈 칸은 0으로 본다.
    coalesce(cast(docks_lcd as integer), 0) + coalesce(cast(docks_qr as integer), 0) as docks,
    operation_type,
    -- 파일 이름의 기준 시점(YY.M월 기준)을 YYYY-MM으로 바꾼 값. 어느 반기 정보인지 구분하는 키다.
    snapshot
from {{ source('bronze', 'stations') }}
