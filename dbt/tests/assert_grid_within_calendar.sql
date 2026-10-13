-- 격자의 모든 시간은 달력(dim_hours)에 있어야 한다. 달력 밖 시간이 있으면 그 행을 돌려준다(T53).
-- 달력 밖 시간은 날씨·요일 정보가 없어 학습 행이 되지 못하면서 반기 평균에는 0으로 섞인다.

-- dbt 단일(singular) 테스트: 돌려준 행이 있으면 실패. left join 뒤 달력 쪽 hour_start가 비는 격자 행, 즉
-- 달력에 없는 시간을 찾는다(anti join). 학습 코드(model/frames.py)는 격자와 달력을 inner join하므로 이런 행은
-- 학습에서는 빠지지만 반기 평균 계산에는 들어간다.
select g.station_id, g.hour_start
from {{ ref('int_station_hour_grid') }} as g
left join {{ ref('dim_hours') }} as d using (hour_start)
where d.hour_start is null
