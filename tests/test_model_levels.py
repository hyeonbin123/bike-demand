# [무엇을 검증하나] v2 수준 특징(공개 반기 평균·비율)과 v3의 '행보다 앞선 대여소 정보 스냅샷'이 각
# 행의 시각에 이미 공개돼 있던 자료만 쓰는지 확인한다.
# - 공개 반기: 행이 속한 달의 첫날보다 1개월 이상 전에 끝난 가장 최근 반기(docs/experiments.md v2)
# - 비율은 1년 전 같은 반기가 없으면 결측, 전체 비율은 대여 합의 비율이라 반기 길이(윤년)가 반영된다
# - 대여소 정보는 행이 속한 달보다 앞선 기준 시점의 스냅샷만 쓰고, 없으면 결측
# [왜 중요한가] 과거 행이 미래에 공개될 자료를 보면 측정이 낙관적으로 나온다. v3는 바로 이 누수를
# 고치려고 다시 잰 측정이다.
# [방식] conftest의 two_year_warehouse는 반기마다 시간당 대여가 1, 2, 3, 4, 5대라 어느 반기를
# 골랐는지가 값으로 바로 보인다.

import numpy as np
import pytest

from bike_demand.model import frames


# 그 날짜의 첫 행에서 수준 특징 3개(station_recent_mean, station_recent_ratio,
# system_recent_ratio)를 꺼낸다.
def levels_at(frame, day: str) -> tuple[float, ...]:
    index = int(np.flatnonzero(frame["day_index"] == np.datetime64(day, "D").astype(int))[0])
    return tuple(float(frame[name][index]) for name in frames.LEVEL_FEATURES)


# 날짜별로 고른 반기와 값: 2023-05·07은 공개된 반기가 없어 결측(07월은 2023 상반기가 끝난 지 1개월이
# 안 됨), 2023-08은 2023 상반기 평균 1이고 1년 전 반기가 없어 비율 결측, 2024-07은 아직 2023
# 하반기(2), 2024-08은 2024 상반기 3 ÷ 2023 상반기 1 = 3.
# 전체 비율은 시간당 평균이 아니라 대여 합의 비율이라 2024 상반기(윤년, 182일) ÷ 2023
# 상반기(181일)만큼 3보다 조금 크다. 2025-03은 2024 하반기 4 ÷ 2023 하반기 2. 평가 기간 자신(2025
# 상반기, 5)은 어떤 행에도 들어가지 않는다.
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


# with_levels를 주지 않으면 수준 특징 열 자체가 없다. v1 측정처럼 수준 특징을 요청하지 않은 호출은
# 예전과 같은 열만 받는다.
def test_levels_are_off_by_default(two_year_warehouse):
    window = frames.Window(rows=("2025-01-01", "2025-02-01"), history=("2023-01-01", "2025-01-01"))
    frame = frames.load_frame(two_year_warehouse, window, False)
    assert not set(frames.LEVEL_FEATURES) & set(frame)


# 22.12월 기준(거치대 10, 강남구)과 24.6월 기준(거치대 20, 서초구, 좌표 바뀜) 스냅샷이 있을 때,
# 24.6월 스냅샷은 6월 행에는 아직 쓰지 않고 7월 행부터 쓴다(행이 속한 달보다 앞선 기준 시점만).
# 자치구 코드는 dim_stations 기준으로 번호를 매기므로 거기에 없는 서초구는 결측이다. asof_stations를
# 끄면 v1·v2 방식대로 dim_stations 값(10)을 모든 행에 쓴다.
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


# 행보다 앞선 스냅샷이 하나도 없으면(24.6월 기준 스냅샷만 있고 행은 6월) 거치대 수·좌표를 최신
# 값으로 채우지 않고 결측으로 둔다(docs/experiments.md v3 고치는 방법).
def test_asof_station_info_is_empty_before_any_snapshot(two_year_warehouse):
    two_year_warehouse.execute("""
        create table stg_stations as
        select 1 as station_no, '강남구' as district, 10 as docks, 37.5 as lat, 127.0 as lon,
               '2024-06' as snapshot
    """)
    window = frames.Window(rows=("2024-06-01", "2024-06-02"), history=("2023-01-01", "2024-06-01"))
    frame = frames.load_frame(two_year_warehouse, window, False, asof_stations=True)
    assert np.isnan(frame["docks"]).all() and np.isnan(frame["lat"]).all()
