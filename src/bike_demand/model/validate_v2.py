"""docs/experiments.md v2 validation: M3(M1 특징 + 공개 반기 수준 특징)를 v1의 M1과 비교한다.

기간·LightGBM 설정·조기 종료·지표는 v1(validate.py)과 같다. M1 값은 v1 결과 파일에서 읽는다.
결과: data/models/v1/validation_v2.json

실행: uv run python -m bike_demand.model.validate_v2
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import duckdb
import numpy as np

from bike_demand.model.frames import FEATURES, LEVEL_FEATURES, Window, feature_matrix, load_frame
from bike_demand.model.metrics import evaluate
from bike_demand.model.validate import PARAMS, TRAIN, VALIDATION, fit_lightgbm

M3_FEATURES = [*FEATURES, *LEVEL_FEATURES]


def choose(m1: dict, m3: dict) -> tuple[str, str]:
    """v2 판정 규칙."""
    if m3["mae"] <= m1["mae"] * 0.98 and m3["peak_mae"] < m1["peak_mae"]:
        return "M3", "M3 MAE가 M1보다 2% 이상 낮고 출퇴근 MAE도 낮음"
    return "M1", "M3가 채택 조건을 통과하지 못해 M1 유지"


def main(warehouse: Path, out_dir: Path) -> dict:
    v1 = json.loads((out_dir / "validation.json").read_text("utf-8"))
    m1 = v1["models"]["M1"]
    report: dict = {"train": TRAIN, "validation": VALIDATION, "params": PARAMS, "M1_from_v1": m1}

    con = duckdb.connect(str(warehouse), read_only=True)
    con.execute("SET threads = 10")
    started = time.perf_counter()
    valid = load_frame(con, Window(VALIDATION, TRAIN), only_active_stations=True, with_levels=True)
    train = load_frame(con, Window(TRAIN, TRAIN), only_active_stations=False, with_levels=True)
    con.close()
    report["load_seconds"] = round(time.perf_counter() - started, 1)
    report["level_nonnull_share_train"] = {
        name: float(np.isfinite(train[name]).mean()) for name in LEVEL_FEATURES
    }
    if len(valid["rentals"]) != m1["rows"]:
        raise SystemExit(f"평가 행 수가 v1과 다름: {len(valid['rentals'])} vs {m1['rows']}")

    log: dict = {"features": M3_FEATURES}
    booster, _ = fit_lightgbm(train, M3_FEATURES, log, consume=True)
    del train
    gc.collect()
    predicted = booster.predict(feature_matrix(valid, M3_FEATURES), num_threads=10)
    keys = (valid["station_code"], valid["day_index"], valid["hour_of_day"])
    importance = booster.feature_importance(importance_type="gain")
    log["gain_share"] = {
        name: round(float(value / importance.sum()), 4)
        for name, value in sorted(zip(M3_FEATURES, importance, strict=True), key=lambda x: -x[1])
    }
    report["M3"] = {**evaluate(valid["rentals"], predicted, *keys), **log}
    booster.save_model(str(out_dir / "validation_M3.txt"))
    report["selected"], report["reason"] = choose(m1, report["M3"])
    (out_dir / "validation_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print({k: v for k, v in report["M3"].items() if k != "features"})
    print("selected", report["selected"], report["reason"], flush=True)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v2 validation 측정")
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("data/models/v1"))
    args = parser.parse_args()
    main(args.warehouse, args.out)
