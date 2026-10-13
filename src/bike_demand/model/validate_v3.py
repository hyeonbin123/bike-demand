"""docs/experiments.md v3 validation: 검토에서 찾은 누수 두 가지를 고친 방법으로 M1'과 M3'를 잰다.

- T24: 거치대 수·좌표·자치구를 행보다 앞선 대여소 정보 스냅샷에서(asof_stations)
- T25: 조기 종료 단계는 조기 종료 구간 이전 기간으로 계산한 패턴·추세를 씀
  (fit_lightgbm_separate_stop)
- 기간·설정·지표·새 대여소 처리·수준 특징 정의는 v1·v2와 같다. test는 재지 않는다
- 결과: data/models/v1/validation_v3.json

실행: uv run python -m bike_demand.model.validate_v3  (몇 시간 걸림)
"""

# [파일 역할] 세 번째 측정(v3 validation). 코드 검토에서 찾은 측정 누수 두 가지(docs/experiments.md
# v3)를 고친 방법으로 M1'(M1 특징)과 M3'(M3 특징)를 다시 재고 v2와 같은 규칙으로 고른다. 여기서 고른
# M3'가 지금 서비스 모델이다.
# [왜 필요한가] v1·v2는 (1) 2023년 행에도 가장 최근(26.6월 기준) 대여소 정보의 거치대 수·좌표를
# 붙였고, (2) 조기 종료로 뗀 두 달의 대여가 대여소 패턴 계산에 섞여 있었다(docs/experiments.md v3).
# 평가 행의 목표값이 새지는 않았지만 판정이 바뀔 수 있어 다시 쟀다.
# [결과] M1' 0.978, M3' 0.927로 M3' 채택(MAE −5.3%). v1·v2와 거의 같았고, M3'는 조기 종료가 라운드
# 상한 2,000에 닿았다. test는 이미 두 번 봤으므로 다시 재지 않는다.

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

# warehouse를 읽기 전용으로 열어 행을 만든다.
import duckdb

# v1 모듈의 기간·설정·조기 종료 시작일을 그대로 쓰고, 학습 함수만 누수를 고친
# fit_lightgbm_separate_stop을 쓴다.
from bike_demand.model.frames import FEATURES, LEVEL_FEATURES, Window, feature_matrix, load_frame
from bike_demand.model.metrics import evaluate
from bike_demand.model.validate import (
    EARLY_STOP_FROM,
    PARAMS,
    TRAIN,
    VALIDATION,
    fit_lightgbm_separate_stop,
)

# 작은따옴표(')가 붙은 이름은 v1·v2 후보와 특징은 같고 측정 방법만 고친 것이라는 표시다.
CANDIDATES = {"M1'": FEATURES, "M3'": [*FEATURES, *LEVEL_FEATURES]}


# v2와 같은 판정 규칙: M3'의 MAE가 M1'보다 2% 이상 낮고 출퇴근 MAE도 낮으면 M3', 아니면 M1'.
def choose(results: dict[str, dict]) -> tuple[str, str]:
    """v3 판정 규칙(v2와 같음)."""
    m1, m3 = results["M1'"], results["M3'"]
    if m3["mae"] <= m1["mae"] * 0.98 and m3["peak_mae"] < m1["peak_mae"]:
        return "M3'", "M3' MAE가 M1'보다 2% 이상 낮고 출퇴근 MAE도 낮음"
    return "M1'", "M3'가 조건을 통과하지 못해 M1'"


# v3 validation을 실행하고 out_dir에 후보별 모델 파일과 validation_v3.json을 쓴다.
# 호출 위치: 명령줄 실행과 tests/test_model_validate_v3.py(학습 함수를 가짜로 바꿔 출력 폴더 생성을
# 확인).
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

    # 행을 불러올 때마다 DB를 새로 열고 바로 닫는 도우미. fit_lightgbm_separate_stop이 두 행 묶음을
    # 차례로 불러오는 함수로 받으므로 연결을 오래 붙잡지 않게 한다.
    # asof_stations=True: 거치대 수·좌표·자치구를 행이 속한 달보다 앞선 대여소 정보 스냅샷에서
    # 가져온다. with_levels=True: v2 수준 특징도 만든다(M1'은 행렬을 만들 때 그 열을 고르지 않을
    # 뿐이다).
    def frame(window: Window, active: bool) -> dict:
        con = duckdb.connect(str(warehouse), read_only=True)
        con.execute("SET threads = 10")
        try:
            return load_frame(con, window, active, with_levels=True, asof_stations=True)
        finally:
            con.close()

    # validation 행은 두 후보가 같이 쓰므로 한 번만 불러온다. 대여소 패턴은 train 전체로 계산한다.
    valid = frame(Window(VALIDATION, TRAIN), True)
    keys = (valid["station_code"], valid["day_index"], valid["hour_of_day"])
    for model_id, features in CANDIDATES.items():
        log: dict = {"features": features}
        # 조기 종료 누수 수정: 조기 종료 단계는 패턴·추세를 조기 종료 구간(2024-11) 전까지만으로
        # 계산한 학습 행을 쓰고, 최종 학습은 train 전체로 계산한 행을 쓴다. 학습 행은 0건 행까지
        # 모두(active=False).
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
        # 파일 이름에서 작은따옴표(chr(39))를 떼어 validation_v3_M1.txt, validation_v3_M3.txt로
        # 저장한다.
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
