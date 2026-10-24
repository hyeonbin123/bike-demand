"""docs/experiments.md v1의 지표와 2026-10-24에 더한 보조 지표(Poisson deviance, 예측 평균 비율)."""

# [파일 역할] 측정 계획(docs/experiments.md v1 지표 절)에 적은 지표를 한 함수로 계산한다. 모든
# 후보(B0, B1, M1, M3, M1', M3')와 test가 같은 함수로 재므로 지표 정의가 측정마다 달라지지 않는다.
# 2026-10-24('학습 행 순서 고정과 보조 지표' 절)에 poisson_deviance·poisson_clipped_rows·
# mean_ratio를 더했다. 기존 7개 키의 정의와 값은 그대로다(tests/test_model_metrics.py가 손으로 센
# 값과 대조).
# [왜 보조 지표인가] 주 지표 MAE는 조건부 중앙값이 최소로 만드는 점수라, 정보를 더하지 않고 예측
# 수준을 낮추기만 한 후보도 MAE가 줄어든다(저장한 v3 부스터 예측에 0.85를 곱하기만 해도 한 주 MAE
# −2.9%). 모델(objective poisson)과 서비스(시간별 예측을 더한 3시간 기대 합)는 평균을 쓰므로, 평균에
# 맞는 점수(Poisson deviance)와 수준 비율(예측 평균 ÷ 실제 평균)을 같은 함수가 함께 남긴다. 지난
# 판정(v1~v3)에는 쓰지 않고, 앞으로의 판정 규칙이 MAE와 함께 보도록 둔 값이다.
# [연결] model/validate.py, validate_v2.py, validate_v3.py, final.py가 evaluate를 부른다.

from __future__ import annotations

# numpy: 수천만 행의 오차를 벡터 연산으로 한 번에 계산한다.
import numpy as np

# 출퇴근 시간: 7~9시, 17~19시(계획의 보조 지표 '출퇴근 시간 MAE'). 판정 규칙에서 MAE와 함께 이 값도
# 기준보다 낮아야 한다.
PEAK_HOURS = (7, 8, 9, 17, 18, 19)
# Poisson deviance에서 예측 평균 μ의 하한. 예측이 0이면 ln(y/μ)가 발산하므로 이 값 아래는 여기로
# 자른다(docs/experiments.md의 식 'μ는 1e-6 아래로 자름'). 잘린 행 수는 poisson_clipped_rows로 함께
# 보고해 자르기가 값에 얼마나 끼어들었는지 보인다. LightGBM poisson 예측(exp)과 기준선 평균은 보통
# 0보다 커서 잘리는 행은 드물다.
MIN_POISSON_MEAN = 1e-6


# 입력: 실제·예측 대여 수와 각 행의 대여소 번호·날짜 번호(day_index)·시각. 출력: 행 수, MAE(주
# 지표), 출퇴근 MAE, WAPE(오차 합 ÷ 실제 합), 3시간 합 MAE, 실제·예측 평균(수준 편향을 보는 값),
# Poisson deviance와 그 자르기 행 수, 예측 평균 ÷ 실제 평균(mean_ratio).
# 입력은 float32로 오지만 합계가 크므로 float64로 바꿔 누적 오차를 줄인다.
# 주의: 실제 합이 0이면 wape와 mean_ratio는 0으로 나눈 값(inf, numpy 경고)이 된다. 평가 대상 규칙(그
# 기간에 대여가 있는 대여소만)에서는 생기지 않는다.
def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    station_code: np.ndarray,
    day_index: np.ndarray,
    hour_of_day: np.ndarray,
) -> dict[str, float]:
    actual = actual.astype(np.float64)
    predicted = predicted.astype(np.float64)
    error = np.abs(actual - predicted)
    peak = np.isin(hour_of_day, PEAK_HOURS)

    # 3시간 합: 대여소×날짜×(시간 // 3) 구간마다 실제·예측을 더한 뒤 MAE
    # 하루를 0~2시, 3~5시, …, 21~23시의 8구간으로 나눈다(서비스가 보는 '앞으로 3시간' 단위에 가까운
    # 보조 지표).
    # (대여소, 날짜, 구간)을 정수 키 하나로 합친다: 날짜 번호가 1천만보다 작고 구간이 0~7이라 키가
    # 겹치지 않는다. np.unique의 역색인으로 같은 키 행을 묶고 bincount로 구간별 합을 낸다(파이썬
    # 반복 없이 수천만 행 처리).
    block = (hour_of_day // 3).astype(np.int64)
    keys = (station_code.astype(np.int64) * 10_000_000 + day_index.astype(np.int64)) * 8 + block
    _, inverse = np.unique(keys, return_inverse=True)
    actual_3h = np.bincount(inverse, weights=actual)
    predicted_3h = np.bincount(inverse, weights=predicted)

    # Poisson deviance: 행마다 2·[y·ln(y/μ) − (y − μ)], y = 0이면 2μ, 그 평균. 예측이 실제와 같을 때
    # 0으로 가장 작고, 수준을 올리거나 내리면 커진다(MAE는 수준을 내려도 줄어들 수 있음).
    # y·ln(y/μ)는 y > 0인 행만 계산해 0·ln 0의 NaN을 피한다(y = 0이면 그 항이 0이라 2μ만 남음). μ는
    # MIN_POISSON_MEAN 아래로 자르고 잘린 행 수를 센다.
    mean = np.maximum(predicted, MIN_POISSON_MEAN)
    positive = actual > 0
    log_term = np.zeros_like(actual)
    log_term[positive] = actual[positive] * np.log(actual[positive] / mean[positive])
    deviance = 2.0 * (log_term - (actual - mean))

    return {
        "rows": int(len(actual)),
        "mae": float(error.mean()),
        "peak_mae": float(error[peak].mean()),
        "wape": float(error.sum() / actual.sum()),
        "mae_3h": float(np.abs(actual_3h - predicted_3h).mean()),
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(predicted.mean()),
        "poisson_deviance": float(deviance.mean()),
        "poisson_clipped_rows": int(np.count_nonzero(predicted < MIN_POISSON_MEAN)),
        "mean_ratio": float(predicted.mean() / actual.mean()),
    }
