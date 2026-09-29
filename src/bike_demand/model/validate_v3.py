"""docs/experiments.md v3 validation: 검토에서 찾은 누수 두 가지를 고친 방법으로 M1'과 M3'를 잰다.

- T24: 거치대 수·좌표·자치구를 행보다 앞선 대여소 정보 스냅샷에서(asof_stations)
- T25: 조기 종료 단계는 조기 종료 구간 이전 기간으로 계산한 패턴·추세를 씀
  (fit_lightgbm_separate_stop)
- 기간·설정·지표·새 대여소 처리·수준 특징 정의는 v1·v2와 같다. test는 재지 않는다
- 결과: data/models/v1/validation_v3.json

실행: uv run python -m bike_demand.model.validate_v3  (몇 시간 걸림)
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import duckdb

from bike_demand.model.frames import FEATURES, LEVEL_FEATURES, Window, feature_matrix, load_frame
from bike_demand.model.metrics import evaluate
from bike_demand.model.validate import (
    EARLY_STOP_FROM,
    PARAMS,
    TRAIN,
    VALIDATION,
    fit_lightgbm_separate_stop,
)

CANDIDATES = {"M1'": FEATURES, "M3'": [*FEATURES, *LEVEL_FEATURES]}


def choose(results: dict[str, dict]) -> tuple[str, str]:
    """v3 판정 규칙(v2와 같음)."""
    m1, m3 = results["M1'"], results["M3'"]
    if m3["mae"] <= m1["mae"] * 0.98 and m3["peak_mae"] < m1["peak_mae"]:
        return "M3'", "M3' MAE가 M1'보다 2% 이상 낮고 출퇴근 MAE도 낮음"
    return "M1'", "M3'가 조건을 통과하지 못해 M1'"


def main(warehouse: Path, out_dir: Path) -> dict:
    # 새 클론에는 data/models/v1이 없다. 몇 시간 학습한 뒤 모델을 쓰다 실패하지 않게 먼저 만든다
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "train": TRAIN,
        "validation": VALIDATION,
        "early_stop_from": EARLY_STOP_FROM,
        "params": PARAMS,
        "models": {},
    }

    def frame(window: Window, active: bool) -> dict:
        con = duckdb.connect(str(warehouse), read_only=True)
        con.execute("SET threads = 10")
        try:
            return load_frame(con, window, active, with_levels=True, asof_stations=True)
        finally:
            con.close()

    valid = frame(Window(VALIDATION, TRAIN), True)
    keys = (valid["station_code"], valid["day_index"], valid["hour_of_day"])
    for model_id, features in CANDIDATES.items():
        log: dict = {"features": features}
        booster, _ = fit_lightgbm_separate_stop(
            lambda: frame(Window(TRAIN, (TRAIN[0], EARLY_STOP_FROM)), False),
            lambda: frame(Window(TRAIN, TRAIN), False),
            features,
            log,
            EARLY_STOP_FROM,
        )
        started = time.perf_counter()
        predicted = booster.predict(feature_matrix(valid, features), num_threads=10)
        log["predict_seconds"] = round(time.perf_counter() - started, 1)
        importance = booster.feature_importance(importance_type="gain")
        log["gain_share"] = {
            name: round(float(value / importance.sum()), 4)
            for name, value in sorted(zip(features, importance, strict=True), key=lambda x: -x[1])
        }
        report["models"][model_id] = {**evaluate(valid["rentals"], predicted, *keys), **log}
        booster.save_model(str(out_dir / f"validation_v3_{model_id.rstrip(chr(39))}.txt"))
        print(model_id, {k: v for k, v in report["models"][model_id].items() if k != "features"})
        del booster
        gc.collect()

    report["selected"], report["reason"] = choose(report["models"])
    (out_dir / "validation_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("selected", report["selected"], report["reason"], flush=True)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v3 validation 측정")
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("data/models/v1"))
    args = parser.parse_args()
    main(args.warehouse, args.out)
