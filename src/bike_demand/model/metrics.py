"""docs/experiments.md v1의 지표."""

# [파일 역할] 측정 계획(docs/experiments.md v1 지표 절)에 적은 지표를 한 함수로 계산한다. 모든
# 후보(B0, B1, M1, M3, M1', M3')와 test가 같은 함수로 재므로 지표 정의가 측정마다 달라지지 않는다.
# [연결] model/validate.py, validate_v2.py, validate_v3.py, final.py가 evaluate를 부른다.

from __future__ import annotations

# numpy: 수천만 행의 오차를 벡터 연산으로 한 번에 계산한다.
import numpy as np

# 출퇴근 시간: 7~9시, 17~19시(계획의 보조 지표 '출퇴근 시간 MAE'). 판정 규칙에서 MAE와 함께 이 값도
# 기준보다 낮아야 한다.
PEAK_HOURS = (7, 8, 9, 17, 18, 19)


# 입력: 실제·예측 대여 수와 각 행의 대여소 번호·날짜 번호(day_index)·시각. 출력: 행 수, MAE(주
# 지표), 출퇴근 MAE, WAPE(오차 합 ÷ 실제 합), 3시간 합 MAE, 실제·예측 평균(수준 편향을 보는 값).
# 입력은 float32로 오지만 합계가 크므로 float64로 바꿔 누적 오차를 줄인다.
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

    return {
        "rows": int(len(actual)),
        "mae": float(error.mean()),
        "peak_mae": float(error[peak].mean()),
        "wape": float(error.sum() / actual.sum()),
        "mae_3h": float(np.abs(actual_3h - predicted_3h).mean()),
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(predicted.mean()),
    }
