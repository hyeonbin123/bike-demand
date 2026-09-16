"""v1 선택 뒤의 두 단계 (docs/experiments.md v1).

1. test: validation에서 고른 후보 하나를 train+validation(2023-01~2025-06)으로 다시 만들어
   test(2025-07~2026-06)에서 **한 번만** 잰다. 결과 파일이 이미 있으면 다시 재지 않는다.
2. serving: 같은 후보를 전체 기간(2023-01~2026-06)으로 학습해 서비스 예측에 쓸 모델과
   산출물(model/artifacts.py)을 저장한다.

LightGBM 후보의 라운드 수는 validation 때와 같은 방식(학습 기간의 마지막 두 달로 조기 종료한 뒤
전체 기간으로 다시 학습)으로 정한다.

실행: uv run python -m bike_demand.model.final test | serving
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import duckdb
import lightgbm as lgb

from bike_demand.model import artifacts
from bike_demand.model.frames import (
    FEATURES,
    WEATHER_FEATURES,
    Window,
    feature_matrix,
    load_frame,
    shift_months,
)
from bike_demand.model.metrics import evaluate
from bike_demand.model.validate import fit_lightgbm

RETRAIN = ("2023-01-01", "2025-07-01")
TEST = ("2025-07-01", "2026-07-01")
SERVING = ("2023-01-01", "2026-07-01")
CANDIDATE_FEATURES = {
    "M1": FEATURES,
    "M2": [f for f in FEATURES if f not in WEATHER_FEATURES],
}


def selected_model(out_dir: Path) -> str:
    report = json.loads((out_dir / "validation.json").read_text("utf-8"))
    return report["selected"]


def run_test(warehouse: Path, out_dir: Path) -> dict:
    result_path = out_dir / "test.json"
    if result_path.exists():
        raise SystemExit(f"test는 한 번만 잰다: {result_path}가 이미 있음")
    selected = selected_model(out_dir)
    report: dict = {"selected": selected, "train": RETRAIN, "test": TEST}
    con = duckdb.connect(str(warehouse), read_only=True)
    test_rows = load_frame(con, Window(rows=TEST, history=RETRAIN), only_active_stations=True)
    keys = (test_rows["station_code"], test_rows["day_index"], test_rows["hour_of_day"])

    if selected in ("B0", "B1"):
        con.close()
        report["metrics"] = evaluate(test_rows["rentals"], test_rows[selected.lower()], *keys)
    else:
        features = CANDIDATE_FEATURES[selected]
        train = load_frame(con, Window(rows=RETRAIN, history=RETRAIN), only_active_stations=False)
        con.close()
        log: dict = {}
        booster, _ = fit_lightgbm(
            train, features, log, early_stop_from=shift_months(RETRAIN[1], -2), consume=True
        )
        del train
        gc.collect()
        predicted = booster.predict(feature_matrix(test_rows, features), num_threads=10)
        report["metrics"] = evaluate(test_rows["rentals"], predicted, *keys)
        report["training"] = log
    # 같은 행의 기준선도 함께 남긴다(선택을 바꾸는 데 쓰지 않음, 비교용)
    report["reference_B0"] = evaluate(test_rows["rentals"], test_rows["b0"], *keys)
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def fit_serving(warehouse: Path, out_dir: Path) -> dict:
    selected = selected_model(out_dir)
    serving_dir = out_dir / "serving"
    serving_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(warehouse), read_only=True)
    meta = artifacts.export(con, SERVING, serving_dir / "artifacts")
    report: dict = {"selected": selected, "history": SERVING, "artifacts": meta}
    if selected in CANDIDATE_FEATURES:
        started = time.perf_counter()
        train = load_frame(con, Window(rows=SERVING, history=SERVING), only_active_stations=False)
        con.close()
        report["train_rows"] = int(len(train["rentals"]))
        report["load_seconds"] = round(time.perf_counter() - started, 1)
        log: dict = {}
        booster, _ = fit_lightgbm(
            train,
            CANDIDATE_FEATURES[selected],
            log,
            early_stop_from=shift_months(SERVING[1], -2),
            consume=True,
        )
        booster.save_model(str(serving_dir / "model.txt"))
        report["training"] = log
    else:
        con.close()
        report["note"] = "기준선이 선택되어 LightGBM 모델 없음"
    (serving_dir / "serving.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), "utf-8"
    )
    return report


def load_serving_model(out_dir: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(out_dir / "serving" / "model.txt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v1 test 측정과 서비스용 학습")
    parser.add_argument("step", choices=["test", "serving"])
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("data/models/v1"))
    args = parser.parse_args()
    if args.step == "test":
        run_test(args.warehouse, args.out)
    else:
        print(json.dumps(fit_serving(args.warehouse, args.out), ensure_ascii=False, default=str))
