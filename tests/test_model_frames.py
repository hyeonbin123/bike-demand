import numpy as np
import pytest

from bike_demand.model import frames
from bike_demand.model.metrics import evaluate


def test_trend_windows():
    last, prior = frames.trend_windows(("2023-01-01", "2025-01-01"))
    assert last == ("2024-07-01", "2025-01-01")
    assert prior == ("2023-07-01", "2024-01-01")


def test_profile_uses_history_only_and_baselines_fall_back(small_warehouse):
    window = frames.Window(rows=("2024-01-16", "2024-02-01"), history=("2024-01-01", "2024-01-16"))
    frame = frames.load_frame(small_warehouse, window, only_active_stations=True)
    st1 = frame["station_code"] == 0
    weekday_8 = (frame["hour_of_day"] == 8) & (frame["is_offday"] == 0)

    # 평가 기간의 100대가 섞이지 않고 학습 기간 평균(2대)
    assert np.allclose(frame["profile_mean"][st1 & weekday_8], 2.0)
    assert np.allclose(frame["b0"][st1 & weekday_8], 2.0)
    # 학습 기간에 없던 ST-2: 패턴 특징은 결측, 기준선은 전체 대여소 평균(여기서는 ST-1뿐이라 2.0)
    st2 = frame["station_code"] == 1
    assert np.isnan(frame["profile_mean"][st2]).all()
    assert np.allclose(frame["b0"][st2 & weekday_8], 2.0)
    assert np.isnan(frame["station_trend"][st2]).all()
    assert frame["rentals"][st1 & weekday_8].min() == 100


def test_inactive_stations_are_dropped_from_evaluation(small_warehouse):
    window = frames.Window(rows=("2024-01-01", "2024-01-16"), history=("2024-01-01", "2024-01-16"))
    frame = frames.load_frame(small_warehouse, window, only_active_stations=True)
    assert set(np.unique(frame["station_code"])) == {0.0}


def test_evaluate_three_hour_blocks():
    actual = np.array([1, 2, 3, 4], dtype=np.float32)
    predicted = np.array([2, 2, 2, 0], dtype=np.float32)
    station = np.zeros(4, dtype=np.float32)
    day = np.zeros(4, dtype=np.int64)
    hour = np.array([0, 1, 2, 7], dtype=np.float32)  # 0~2시 한 구간, 7시는 6~8시 구간
    result = evaluate(actual, predicted, station, day, hour)
    assert result["mae"] == pytest.approx((1 + 0 + 1 + 4) / 4)
    assert result["peak_mae"] == pytest.approx(4)
    assert result["mae_3h"] == pytest.approx((abs(6 - 6) + abs(4 - 0)) / 2)
    assert result["wape"] == pytest.approx(6 / 10)
