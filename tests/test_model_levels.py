import duckdb
import numpy as np
import pytest

from bike_demand.model import frames


@pytest.fixture
def two_year_warehouse():
    """대여소 하나가 반기마다 시간당 1, 2, 3, 4, 5대를 빌리는 2023-01 ~ 2025-06 격자."""
    db = duckdb.connect()
    db.execute("""
        create table dim_stations as
        select 'ST-1' as station_id, '강남구' as district, 10 as docks, 37.5 as lat, 127.0 as lon
    """)
    db.execute("""
        create table dim_hours as
        select h as hour_start, hour(h) as hour_of_day, isodow(h) as day_of_week,
               month(h) as month, dayofyear(h) as day_of_year,
               false as is_holiday, isodow(h) >= 6 as is_offday,
               10.0 as temp_c, 0.0 as rain_mm, 1.0 as wind_ms, 50.0 as humidity_pct,
               0.0 as snow_cm
        from unnest(generate_series(timestamp '2023-01-01', timestamp '2025-06-30 23:00:00',
                                    interval 1 hour)) t(h)
    """)
    db.execute("""
        create table int_station_hour_grid as
        select 'ST-1' as station_id, hour_start,
               ((year(hour_start) - 2023) * 2 + (month(hour_start) > 6)::int + 1) as rentals
        from dim_hours
    """)
    return db


def levels_at(frame, day: str) -> tuple[float, ...]:
    index = int(np.flatnonzero(frame["day_index"] == np.datetime64(day, "D").astype(int))[0])
    return tuple(float(frame[name][index]) for name in frames.LEVEL_FEATURES)


def test_levels_use_only_half_years_published_before_the_row(two_year_warehouse):
    window = frames.Window(rows=("2023-01-01", "2025-07-01"), history=("2023-01-01", "2025-01-01"))
    frame = frames.load_frame(two_year_warehouse, window, False, with_levels=True)

    # 2023-05: 공개된 반기가 없음(2022년 하반기)
    assert all(np.isnan(v) for v in levels_at(frame, "2023-05-10"))
    # 2023-07: 1개월 안 됐으므로 아직 2023 상반기(1)가 아니라 2022 하반기 → 없음
    assert np.isnan(levels_at(frame, "2023-07-10")[0])
    # 2023-08: 2023 상반기 평균 1, 1년 전 반기 없음 → 비율 빈 값
    mean, ratio, system = levels_at(frame, "2023-08-10")
    assert mean == 1 and np.isnan(ratio) and np.isnan(system)
    # 2024-07: 아직 2023 하반기(2). 그 1년 전(2022 하반기)은 없음
    mean, ratio, _ = levels_at(frame, "2024-07-31")
    assert mean == 2 and np.isnan(ratio)
    # 2024-08: 2024 상반기(3) / 2023 상반기(1)
    mean, ratio, system = levels_at(frame, "2024-08-01")
    assert (mean, ratio) == (3, 3)
    assert system == pytest.approx(3 * 182 / 181)  # 합계 비율: 2024 상반기는 윤년이라 하루 더 김
    # 2025-03: 2024 하반기(4) / 2023 하반기(2), 두 반기 날수가 같음
    assert levels_at(frame, "2025-03-15") == pytest.approx((4, 2, 2))
    # 평가 기간 자신의 값(5)은 어떤 행에도 들어가지 않음
    assert np.nanmax(frame["station_recent_mean"]) == 4


def test_levels_are_off_by_default(two_year_warehouse):
    window = frames.Window(rows=("2025-01-01", "2025-02-01"), history=("2023-01-01", "2025-01-01"))
    frame = frames.load_frame(two_year_warehouse, window, False)
    assert not set(frames.LEVEL_FEATURES) & set(frame)
