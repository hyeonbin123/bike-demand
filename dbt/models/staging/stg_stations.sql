-- 대여소 정보 반기 스냅샷. 시점 × 대여소번호 = 한 행

select
    cast(station_no as integer) as station_no,
    station_name,
    district,
    address,
    cast(lat as double) as lat,
    cast(lon as double) as lon,
    cast(installed_at as timestamp) as installed_at,
    coalesce(cast(docks_lcd as integer), 0) + coalesce(cast(docks_qr as integer), 0) as docks,
    operation_type,
    snapshot
from {{ source('bronze', 'stations') }}
