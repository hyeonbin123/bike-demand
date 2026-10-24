# [무엇을 검증하나] 지표 계산 model/metrics.py의 evaluate가 docs/experiments.md의 정의대로 값을
# 내는지 확인한다. (1) v1부터 쓰던 7개 키(rows, mae, peak_mae, wape, mae_3h, actual_mean,
# predicted_mean)의 값이 손으로 센 값과 같다 — 보조 지표를 더한 뒤에도 바뀌지 않아야 기록
# 수치(v1~v5)와 이어진다. (2) '학습 행 순서 고정과 보조 지표' 절에서 더한 poisson_deviance·
# poisson_clipped_rows·mean_ratio가 적어 둔 식과 같다.
# [왜 중요한가] 주 지표 MAE는 조건부 중앙값이 최소로 만드는 점수라 예측을 낮추기만 한 후보도 이득을
# 본다. 모델(objective poisson)과 서비스(시간별 예측을 더한 3시간 기대 합)는 평균을 쓰므로, 평균에
# 맞는 점수(Poisson deviance)와 수준 비율(예측 평균 ÷ 실제 평균)을 같은 함수가 함께 남겨야 뒤의
# 판정 규칙이 MAE만으로 고르는 일을 막을 수 있다.

import math

import numpy as np
import pytest

from bike_demand.model import metrics
from bike_demand.model.metrics import evaluate

# 손으로 셀 수 있는 4행. 대여소·날짜는 모두 같고 시각만 0, 1, 2, 7시(0~2시 구간 셋 + 출퇴근 7시
# 하나). 실제 y = [1, 0, 3, 2], 예측 μ = [2, 0.5, 3, 0]. 네 번째 행의 예측 0은 Poisson deviance에서
# 1e-6으로 잘린다.
ACTUAL = np.array([1, 0, 3, 2], dtype=np.float32)
PREDICTED = np.array([2, 0.5, 3, 0], dtype=np.float32)
STATION = np.zeros(4, dtype=np.float32)
DAY = np.zeros(4, dtype=np.int64)
HOUR = np.array([0, 1, 2, 7], dtype=np.float32)


# 결과 키는 v1 7개 + 보조 3개로 고정한다. 키가 늘거나 이름이 바뀌면 결과 JSON을 읽는 쪽(판정 규칙,
# 문서 표)이 함께 바뀌어야 하므로 집합으로 못 박는다.
def test_evaluate_keys_are_the_seven_original_plus_three_auxiliary():
    result = evaluate(ACTUAL, PREDICTED, STATION, DAY, HOUR)
    assert set(result) == {
        "rows",
        "mae",
        "peak_mae",
        "wape",
        "mae_3h",
        "actual_mean",
        "predicted_mean",
        "poisson_deviance",
        "poisson_clipped_rows",
        "mean_ratio",
    }


# 기존 7개 키: MAE (1 + 0.5 + 0 + 2) / 4, 출퇴근 MAE는 7시 행 하나 |2 − 0| = 2, WAPE = 오차 합 3.5 /
# 실제 합 6, 3시간 합 MAE는 0~2시 구간(실제 4, 예측 5.5)과 6~8시 구간(2 대 0)의 평균 (1.5 + 2) / 2,
# 평균은 실제 6/4, 예측 5.5/4. 보조 지표를 더해도 이 값들은 그대로다.
def test_original_metrics_are_unchanged():
    result = evaluate(ACTUAL, PREDICTED, STATION, DAY, HOUR)
    assert result["rows"] == 4
    assert result["mae"] == pytest.approx(3.5 / 4)
    assert result["peak_mae"] == pytest.approx(2)
    assert result["wape"] == pytest.approx(3.5 / 6)
    assert result["mae_3h"] == pytest.approx((1.5 + 2) / 2)
    assert result["actual_mean"] == pytest.approx(1.5)
    assert result["predicted_mean"] == pytest.approx(1.375)


# Poisson deviance = 평균 2·[y·ln(y/μ) − (y − μ)], y = 0이면 2μ, μ는 1e-6 아래로 자른다(문서의 식).
# 행별: y=1, μ=2 → 2·(ln 0.5 + 1); y=0, μ=0.5 → 1; y=3, μ=3 → 0; y=2, μ=0 → μ를 1e-6으로 잘라
# 2·(2·ln(2·10⁶) − (2 − 10⁻⁶)). 잘린 행은 하나다.
def test_poisson_deviance_matches_the_written_formula_and_counts_clipped_rows():
    result = evaluate(ACTUAL, PREDICTED, STATION, DAY, HOUR)
    clipped = metrics.MIN_POISSON_MEAN
    assert clipped == 1e-6
    rows = [
        2 * (1 * math.log(1 / 2) - (1 - 2)),
        2 * 0.5,
        0.0,
        2 * (2 * math.log(2 / clipped) - (2 - clipped)),
    ]
    assert result["poisson_deviance"] == pytest.approx(sum(rows) / 4, rel=1e-9)
    assert result["poisson_clipped_rows"] == 1
    assert isinstance(result["poisson_clipped_rows"], int)


# 예측이 모두 양수면 잘린 행이 0이고, 예측이 실제와 같으면 deviance가 0이다. 같은 예측에 0.9나 1.1을
# 곱하면(수준만 바꿈) deviance는 커진다 — MAE와 달리 평균이 맞는 예측을 가장 좋게 보는 점수라는 뜻.
def test_poisson_deviance_is_zero_at_the_truth_and_grows_when_the_level_is_scaled():
    actual = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    hour = np.array([0, 1, 2, 7, 8, 9], dtype=np.float32)  # 출퇴근 시각도 섞어 peak_mae가 비지 않게
    keys = (np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.int64), hour)
    exact = evaluate(actual, actual, *keys)
    assert exact["poisson_deviance"] == pytest.approx(0.0, abs=1e-12)
    assert exact["poisson_clipped_rows"] == 0
    assert exact["mean_ratio"] == pytest.approx(1.0)
    for scale in (0.9, 1.1):
        scaled = evaluate(actual, actual * scale, *keys)
        assert scaled["poisson_deviance"] > 0
        assert scaled["mean_ratio"] == pytest.approx(scale, rel=1e-6)


# 예측 평균 비율 = predicted_mean ÷ actual_mean. 수준 편향(v3 M3' +5.7%, 서비스 9월 1.15배)을 한
# 값으로 읽는 보조 지표다. 위 4행에서 5.5/4 ÷ 6/4 = 5.5/6.
def test_mean_ratio_is_predicted_mean_over_actual_mean():
    result = evaluate(ACTUAL, PREDICTED, STATION, DAY, HOUR)
    assert result["mean_ratio"] == pytest.approx(5.5 / 6)
    assert result["mean_ratio"] == pytest.approx(result["predicted_mean"] / result["actual_mean"])


# 실제가 모두 0인 입력(평가 대상 규칙상 실제로는 없음)에서도 예외 없이 끝나야 한다. WAPE처럼
# mean_ratio도 0으로 나눈 값(inf)이 되고, deviance는 2μ의 평균이다.
def test_all_zero_actuals_do_not_raise():
    actual = np.zeros(3, dtype=np.float32)
    predicted = np.array([1, 2, 3], dtype=np.float32)
    hour = np.array([0, 1, 7], dtype=np.float32)
    keys = (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.int64), hour)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = evaluate(actual, predicted, *keys)
    assert result["poisson_deviance"] == pytest.approx(2 * 2.0)  # 평균 2μ, μ 평균 2
    assert result["poisson_clipped_rows"] == 0
    assert math.isinf(result["mean_ratio"]) and math.isinf(result["wape"])
