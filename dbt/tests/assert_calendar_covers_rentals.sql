-- 대여가 있는 모든 시간은 달력(dim_hours) 안에 있어야 한다. 달력 밖이면 그 행을 돌려준다.
-- 새 반기를 변환하고 calendar_end를 올리지 않은 채 빌드하면 격자도 달력 끝에서 잘려(T53) 새 반기가
-- 조용히 빠진다. 달력과 격자가 함께 잘리므로 assert_grid_within_calendar로는 잡히지 않는다.
-- 자료 마지막 날 밤 대여의 다음 날 반납 행(rentals = 0)은 설계대로 달력 밖이라 뺀다.

-- dbt 단일(singular) 테스트: 이 쿼리가 행을 하나라도 돌려주면 실패한다. 커밋 29a4094에서 추가했다. 이전에는
-- calendar_end를 넘기지 않은 빌드가 테스트를 모두 통과하면서 새 반기를 조용히 잘랐다. 반납 전용 행을 빼려고
-- rentals > 0인 시간만 보고, left join 뒤 달력 쪽이 비면 달력 밖이다.
select f.station_id, f.hour_start, f.rentals
from {{ ref('fct_station_hourly') }} as f
left join {{ ref('dim_hours') }} as d using (hour_start)
where f.rentals > 0 and d.hour_start is null
