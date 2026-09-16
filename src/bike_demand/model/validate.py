"""docs/experiments.md v1 validation: B0, B1, M1, M2를 재고 판정 규칙대로 고른다.

- train: 2023-01-01 ~ 2024-12-31, validation: 2025-01-01 ~ 2025-06-30
- LightGBM 조기 종료: train 안의 2024-11-01 ~ 2024-12-31을 떼어 l1으로 판단
  (50라운드 동안 나아지지 않으면 멈춤).
  정해진 라운드 수로 train 전체에서 다시 학습한 모델을 validation에 쓴다
- 결과: data/models/v1/validation.json (측정값, 설정, 선택)

실행: uv run python -m bike_demand.model.validate  (저장소 루트, 수십 분 걸림)
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np

from bike_demand.model.frames import (
    CATEGORICAL,
    FEATURES,
    WEATHER_FEATURES,
    Window,
    feature_matrix,
    load_frame,
)
from bike_demand.model.metrics import evaluate

TRAIN = ("2023-01-01", "2025-01-01")
VALIDATION = ("2025-01-01", "2025-07-01")
EARLY_STOP_FROM = "2024-11-01"

PARAMS = {
    "objective": "poisson",
    "metric": "l1",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_threads": 10,
    "seed": 42,
    "verbose": -1,
}
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50


def _day_index(day: str) -> int:
    return int(np.datetime64(day, "D").astype(np.int64))


def fit_lightgbm(
    train: dict[str, np.ndarray],
    features: list[str],
    log: dict,
    early_stop_from: str = EARLY_STOP_FROM,
    consume: bool = False,
) -> tuple[lgb.Booster, int]:
    """early_stop_from 이후(학습 기간의 마지막 두 달)로 라운드 수를 정한다.

    그 라운드 수로 학습 기간 전체에서 다시 학습한 모델을 돌려준다.
    """
    categorical = [f for f in features if f in CATEGORICAL]
    split = train["day_index"] < _day_index(early_stop_from)
    matrix = feature_matrix(train, features)
    if consume:  # 행렬을 만든 뒤 특징 열은 더 쓰지 않으므로 메모리에서 뺀다(큰 기간 학습용)
        for name in features:
            train.pop(name, None)
        gc.collect()

    started = time.perf_counter()
    fit_set = lgb.Dataset(
        matrix[split], train["rentals"][split], feature_name=features,
        categorical_feature=categorical, free_raw_data=True,
    )  # fmt: skip
    stop_set = lgb.Dataset(
        matrix[~split], train["rentals"][~split], reference=fit_set, free_raw_data=True
    )
    probe = lgb.train(
        PARAMS,
        fit_set,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[stop_set],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    best = probe.best_iteration
    log["early_stopping_seconds"] = round(time.perf_counter() - started, 1)
    log["best_iteration"] = best
    log["early_stopping_l1"] = probe.best_score["valid_0"]["l1"]
    del probe, fit_set, stop_set
    gc.collect()

    started = time.perf_counter()
    full_set = lgb.Dataset(
        matrix, train["rentals"], feature_name=features, categorical_feature=categorical,
        free_raw_data=True,
    )  # fmt: skip
    booster = lgb.train(PARAMS, full_set, num_boost_round=best)
    log["refit_seconds"] = round(time.perf_counter() - started, 1)
    del full_set, matrix
    gc.collect()
    return booster, best


def choose(results: dict[str, dict]) -> tuple[str, str]:
    """판정 규칙 (docs/experiments.md v1). (선택 ID, 이유)"""
    b0, m1 = results["B0"], results["M1"]
    if m1["mae"] <= b0["mae"] * 0.97 and m1["peak_mae"] < b0["peak_mae"]:
        return "M1", "M1 MAE가 B0보다 3% 이상 낮고 출퇴근 MAE도 낮음"
    best = min(("B0", "B1"), key=lambda k: results[k]["mae"])
    return best, "M1이 규칙 1을 통과하지 못해 B0·B1 중 MAE가 낮은 것"


def main(warehouse: Path, out_dir: Path) -> dict:
    con = duckdb.connect(str(warehouse), read_only=True)
    con.execute("SET threads = 10")
    report: dict = {"train": TRAIN, "validation": VALIDATION, "params": PARAMS, "models": {}}

    started = time.perf_counter()
    valid = load_frame(con, Window(rows=VALIDATION, history=TRAIN), only_active_stations=True)
    report["validation_load_seconds"] = round(time.perf_counter() - started, 1)
    keys = (valid["station_code"], valid["day_index"], valid["hour_of_day"])
    for baseline in ("B0", "B1"):
        report["models"][baseline] = evaluate(valid["rentals"], valid[baseline.lower()], *keys)
        print(baseline, report["models"][baseline], flush=True)

    started = time.perf_counter()
    train = load_frame(con, Window(rows=TRAIN, history=TRAIN), only_active_stations=False)
    con.close()
    report["train_rows"] = int(len(train["rentals"]))
    report["train_load_seconds"] = round(time.perf_counter() - started, 1)
    print("train rows", report["train_rows"], flush=True)

    candidates = {
        "M1": FEATURES,
        "M2": [f for f in FEATURES if f not in WEATHER_FEATURES],
    }
    for model_id, features in candidates.items():
        log: dict = {"features": features}
        booster, _ = fit_lightgbm(train, features, log)
        started = time.perf_counter()
        predicted = booster.predict(feature_matrix(valid, features), num_threads=10)
        log["predict_seconds"] = round(time.perf_counter() - started, 1)
        importance = booster.feature_importance(importance_type="gain")
        log["gain_share"] = {
            name: round(float(value / importance.sum()), 4)
            for name, value in sorted(zip(features, importance, strict=True), key=lambda x: -x[1])
        }
        report["models"][model_id] = {**evaluate(valid["rentals"], predicted, *keys), **log}
        print(model_id, {k: v for k, v in report["models"][model_id].items() if k != "features"})
        out_dir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(out_dir / f"validation_{model_id}.txt"))
        del booster
        gc.collect()

    report["selected"], report["reason"] = choose(report["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("selected", report["selected"], report["reason"], flush=True)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v1 validation 측정")
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("data/models/v1"))
    args = parser.parse_args()
    main(args.warehouse, args.out)
