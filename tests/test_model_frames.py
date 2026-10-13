# [무엇을 검증하나] 측정용 행·특징을 만드는 model/frames.py와 지표 계산 model/metrics.py의 기본
# 규칙을 확인한다: 추세 기간 계산, 대여소 패턴을 학습 기간(history)으로만 계산하기(평가 기간 값 누수
# 방지), 학습에 없던 새 대여소의 처리, 평가 기간에 대여가 없는 대여소 제외, 3시간 합 MAE 등 지표.
# [왜 중요한가] 이 프로젝트의 핵심 원칙이 '예측할 때 알 수 있는 것만 특징으로'다(README 설계 판단).
# 평가 기간의 대여가 대여소 패턴에 섞이면 validation 점수가 실제보다 좋게 나온다.

import numpy as np
import pytest

# small_warehouse fixture(conftest)는 학습 기간 ST-1의 평일 8시 2대, 평가 기간 100대로 일부러 크게
# 다르게 만든 warehouse다.
from bike_demand.model import frames
from bike_demand.model.metrics import evaluate


# 학습 기간이 2025-01-01에 끝나면 '마지막 6개월'은 2024-07~12, '그 1년 전 같은 6개월'은
# 2023-07~12다(docs/experiments.md v1의 대여소 추세 정의).
def test_trend_windows():
    last, prior = frames.trend_windows(("2023-01-01", "2025-01-01"))
    assert last == ("2024-07-01", "2025-01-01")
    assert prior == ("2023-07-01", "2024-01-01")


# 평가 행(1/16~)의 profile_mean과 기준선 B0이 학습 기간 평균 2대인지 본다(평가 기간 100대가 섞이면
# 값이 커진다). 평가 행의 실제 대여는 100대 그대로다.
# 학습 기간에 없던 ST-2: 패턴·추세 특징은 결측(LightGBM이 결측으로 처리), 기준선 B0은 전체 대여소의
# 같은 요일구분×시간 평균으로 채운다(측정 전 보충 규칙).
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


# only_active_stations=True면 그 기간에 대여가 있는 대여소만 평가 행에 넣는다(계획의 '평가 대상
# 행'). 1/1~1/15에는 ST-1만 남는다.
def test_inactive_stations_are_dropped_from_evaluation(small_warehouse):
    window = frames.Window(rows=("2024-01-01", "2024-01-16"), history=("2024-01-01", "2024-01-16"))
    frame = frames.load_frame(small_warehouse, window, only_active_stations=True)
    assert set(np.unique(frame["station_code"])) == {0.0}


# 지표 계산을 손으로 셀 수 있는 4행으로 확인한다. MAE (1+0+1+4)/4, 출퇴근(7시) MAE 4, WAPE = 오차 합
# 6 / 실제 합 10. 3시간 합 MAE는 대여소×날짜×(시간 // 3) 구간마다 실제·예측을 더한 뒤 비교하므로
# 0~2시 구간(6 대 6)과 6~8시 구간(4 대 0)의 평균 2다.
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
