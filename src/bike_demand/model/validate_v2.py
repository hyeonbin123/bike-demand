"""docs/experiments.md v2 validation: M3(M1 특징 + 공개 반기 수준 특징)를 v1의 M1과 비교한다.

기간·LightGBM 설정·조기 종료·지표는 v1(validate.py)과 같다. M1 값은 v1 결과 파일에서 읽는다.
결과: data/models/v1/validation_v2.json

실행: uv run python -m bike_demand.model.validate_v2
"""

# [파일 역할] 두 번째 측정(v2 validation). v1의 M1 특징에 '그 행의 시각에 이미 공개돼 있었을 가장
# 최근 반기'로 계산한 수준 특징 3개를 더한 M3를 학습해, v1에서 잰 M1과 비교하고 판정 규칙으로 채택
# 여부를 정한다.
# [왜 필요한가] v1의 모든 후보가 validation·test에서 평균을 14~15% 높게 예측했다. 추세 특징이 학습
# 기간 끝에서 한 번 계산한 상수라 학습에 쓰이지 않았기 때문이다(docs/experiments.md v2 '왜'). 행마다
# 공개 시점을 지킨 수준을 넣으면 미래 정보 없이 수준 변화를 배울 수 있다.
# [결과] M3 MAE 0.932로 M1 0.979보다 4.8% 낮고 출퇴근 MAE도 낮아 M3 채택, 수준 편향 +14.7% →
# +6.2%(docs/experiments.md v2 결과). 이 측정은 v1과 같은 방식이라 v3에서 고친 두 누수를 그대로
# 가진다.
# [흐름] v1 결과 파일에서 M1 값 읽기 → validation·train 행 읽기(수준 특징 포함) → 평가 행 수가 v1과
# 같은지 확인 → M3 학습·평가 → 판정 → validation_v2.json.

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

# warehouse 조회(duckdb)와 수준 특징의 결측 비율 계산(numpy).
import duckdb
import numpy as np

# 기간·LightGBM 설정·조기 종료·지표를 v1 모듈에서 그대로 가져와 '특징만 다른' 비교가 되게 한다.
# LEVEL_FEATURES는 station_recent_mean·station_recent_ratio·system_recent_ratio.
from bike_demand.model.frames import FEATURES, LEVEL_FEATURES, Window, feature_matrix, load_frame
from bike_demand.model.metrics import evaluate
from bike_demand.model.validate import PARAMS, TRAIN, VALIDATION, fit_lightgbm

# M3 = M1 특징 전부 + 공개 반기 수준 특징 3개(docs/experiments.md v2 '새 특징').
M3_FEATURES = [*FEATURES, *LEVEL_FEATURES]


# 측정 전에 커밋한 v2 판정 규칙. M3의 MAE가 M1의 0.98배 이하(2% 이상 낮음, 0.960 이하)이고 출퇴근
# MAE도 M1보다 낮으면 M3, 아니면 M1 유지. 수준 편향은 보고만 하고 판정에 쓰지 않는다(계획).
def choose(m1: dict, m3: dict) -> tuple[str, str]:
    """v2 판정 규칙."""
    if m3["mae"] <= m1["mae"] * 0.98 and m3["peak_mae"] < m1["peak_mae"]:
        return "M3", "M3 MAE가 M1보다 2% 이상 낮고 출퇴근 MAE도 낮음"
    return "M1", "M3가 채택 조건을 통과하지 못해 M1 유지"


# v2 validation을 실행하고 out_dir/validation_v2.json을 쓴다. M1은 다시 학습하지 않고 v1 결과
# 파일(validation.json)의 값을 쓰므로, v1을 먼저 돌려 둬야 한다.
def main(warehouse: Path, out_dir: Path) -> dict:
    v1 = json.loads((out_dir / "validation.json").read_text("utf-8"))
    m1 = v1["models"]["M1"]
    report: dict = {"train": TRAIN, "validation": VALIDATION, "params": PARAMS, "M1_from_v1": m1}

    con = duckdb.connect(str(warehouse), read_only=True)
    con.execute("SET threads = 10")
    started = time.perf_counter()
    # with_levels=True면 수준 특징도 만든다. 수준은 격자 전체에서 '행보다 앞서 공개된 반기'만
    # 읽으므로 validation 행에서도 미래 값이 들어가지 않는다.
    valid = load_frame(con, Window(VALIDATION, TRAIN), only_active_stations=True, with_levels=True)
    train = load_frame(con, Window(TRAIN, TRAIN), only_active_stations=False, with_levels=True)
    con.close()
    report["load_seconds"] = round(time.perf_counter() - started, 1)
    # 학습 행 중 수준 특징에 값이 있는 비율을 남긴다. 2023년 초 행은 가리킬 반기 자료가 없어 비고,
    # 비율 특징은 2024-08 행부터 값이 있어 배울 사례가 적다는 한계를 숫자로 확인하려는 것이다(결과:
    # 70%·20%·21%).
    report["level_nonnull_share_train"] = {
        name: float(np.isfinite(train[name]).mean()) for name in LEVEL_FEATURES
    }
    # v1과 같은 평가 행으로 재야 M1 값과 비교할 수 있다. 행 수가 다르면(warehouse가 바뀐 경우 등)
    # 비교 자체가 무의미하므로 학습 전에 멈춘다.
    if len(valid["rentals"]) != m1["rows"]:
        raise SystemExit(f"평가 행 수가 v1과 다름: {len(valid['rentals'])} vs {m1['rows']}")

    log: dict = {"features": M3_FEATURES}
    # v1과 같은 2단계 학습(조기 종료로 라운드 수 → 전체 재학습). consume=True로 행렬을 만든 뒤 특징
    # 열을 지우고, 학습이 끝나면 train 자체도 지워 예측 전에 메모리를 비운다.
    booster, _ = fit_lightgbm(train, M3_FEATURES, log, consume=True)
    del train
    gc.collect()
    predicted = booster.predict(feature_matrix(valid, M3_FEATURES), num_threads=10)
    keys = (valid["station_code"], valid["day_index"], valid["hour_of_day"])
    # 새 특징이 실제로 쓰였는지 보려고 특징별 gain 비율을 남긴다.
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


# 명령줄 실행. --out은 v1 결과가 있는 폴더여야 한다.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v2 validation 측정")
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("data/models/v1"))
    args = parser.parse_args()
    main(args.warehouse, args.out)
