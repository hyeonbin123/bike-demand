-- 격자의 모든 시간은 달력(dim_hours)에 있어야 한다. 달력 밖 시간이 있으면 그 행을 돌려준다(T53).
-- 달력 밖 시간은 날씨·요일 정보가 없어 학습 행이 되지 못하면서 반기 평균에는 0으로 섞인다.

select g.station_id, g.hour_start
from {{ ref('int_station_hour_grid') }} as g
left join {{ ref('dim_hours') }} as d using (hour_start)
where d.hour_start is null
