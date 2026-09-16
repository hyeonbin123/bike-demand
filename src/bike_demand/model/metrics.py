"""docs/experiments.md v1의 지표."""

from __future__ import annotations

import numpy as np

PEAK_HOURS = (7, 8, 9, 17, 18, 19)


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
