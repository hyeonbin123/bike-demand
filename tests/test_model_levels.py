import numpy as np
import pytest

from bike_demand.model import frames


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


def test_asof_station_info_uses_only_earlier_snapshots(two_year_warehouse):
    """대여소 정보 스냅샷: 22.12월 기준 거치대 10, 24.6월 기준 20(자치구 바뀜)."""
    two_year_warehouse.execute("""
        create table stg_stations as
        select * from (values (1, '강남구', 10, 37.5, 127.0, '2022-12'),
                              (1, '서초구', 20, 37.4, 127.1, '2024-06'))
            t(station_no, district, docks, lat, lon, snapshot)
    """)
    window = frames.Window(rows=("2024-06-01", "2024-08-01"), history=("2023-01-01", "2024-06-01"))
    frame = frames.load_frame(two_year_warehouse, window, False, asof_stations=True)
    june = frame["day_index"] < np.datetime64("2024-07-01", "D").astype(int)

    # 24.6월 기준 스냅샷은 6월 행에는 아직 쓰지 않고 7월 행부터 쓴다
    assert set(frame["docks"][june]) == {10.0}
    assert set(frame["docks"][~june]) == {20.0}
    assert set(frame["lat"][~june]) == {np.float32(37.4)}
    # 서초구는 코드 매기기(dim_stations 기준)에 없으므로 자치구 코드는 빈 값
    assert np.isnan(frame["district_code"][~june]).all()
    assert set(frame["district_code"][june]) == {0.0}

    default = frames.load_frame(two_year_warehouse, window, False)
    assert set(default["docks"]) == {10.0}  # 기본값은 dim_stations(여기서는 10) 그대로


def test_asof_station_info_is_empty_before_any_snapshot(two_year_warehouse):
    two_year_warehouse.execute("""
        create table stg_stations as
        select 1 as station_no, '강남구' as district, 10 as docks, 37.5 as lat, 127.0 as lon,
               '2024-06' as snapshot
    """)
    window = frames.Window(rows=("2024-06-01", "2024-06-02"), history=("2023-01-01", "2024-06-01"))
    frame = frames.load_frame(two_year_warehouse, window, False, asof_stations=True)
    assert np.isnan(frame["docks"]).all() and np.isnan(frame["lat"]).all()
