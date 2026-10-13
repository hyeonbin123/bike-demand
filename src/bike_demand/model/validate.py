"""docs/experiments.md v1 validation: B0, B1, M1, M2를 재고 판정 규칙대로 고른다.

- train: 2023-01-01 ~ 2024-12-31, validation: 2025-01-01 ~ 2025-06-30
- LightGBM 조기 종료: train 안의 2024-11-01 ~ 2024-12-31을 떼어 l1으로 판단
  (50라운드 동안 나아지지 않으면 멈춤).
  정해진 라운드 수로 train 전체에서 다시 학습한 모델을 validation에 쓴다
- 결과: data/models/v1/validation.json (측정값, 설정, 선택)

실행: uv run python -m bike_demand.model.validate  (저장소 루트, 수십 분 걸림)
"""

# [파일 역할] 첫 번째 측정(v1 validation)을 실행한다. 기준선 B0(대여소×요일구분×시간
# 평균)·B1(대여소×시간 평균)과 LightGBM 후보 M1(날씨 포함 전체 특징)·M2(날씨 뺌)를 같은 validation
# 행에서 재고, 측정 전에 커밋한 판정 규칙(choose)으로 하나를 고른다.
# [왜 필요한가] 프로젝트 측정 규칙은 '후보·지표·판정 규칙을 docs/experiments.md에 먼저 적고 커밋 →
# validation으로 고르기 → test는 고른 것 하나로 한 번'이다. 이 파일이 그중 validation 단계다(test는
# model/final.py).
# [재사용] 조기 종료로 라운드 수를 정하고 전체로 다시 학습하는 fit_lightgbm, v3의 누수 없는 변형
# fit_lightgbm_separate_stop, 공통 PARAMS·기간 상수는 validate_v2.py·validate_v3.py·final.py가
# 가져다 쓴다. 그래서 측정마다 LightGBM 설정이 같다.
# [결과] v1 validation에서 M1 MAE 0.979, B0 1.210으로 M1이 골라졌다(docs/experiments.md v1 결과). 이
# 파일의 M1 학습은 뒤에 v3에서 고친 두 누수(최신 대여소 정보, 조기 종료 구간이 섞인 대여소 패턴)를
# 그대로 가진 v1 방식이다.

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from pathlib import Path

# duckdb: dbt가 만든 warehouse(data/warehouse/bike_demand.duckdb)를 읽기 전용으로 열어 대여소×시간
# 행을 뽑는다. lightgbm: M1·M2 모델. numpy: 행을 열 배열로 다룬다.
import duckdb
import lightgbm as lgb
import numpy as np

# frames: 측정용 행과 특징을 만드는 SQL(load_frame)과 특징 이름 목록. 학습·평가·서비스가 같은 특징
# 정의를 쓰게 한곳에 모았다. metrics.evaluate: 계획의 지표(MAE, 출퇴근 MAE, WAPE, 3시간 합 MAE,
# 예측·실제 평균).
from bike_demand.model.frames import (
    CATEGORICAL,
    FEATURES,
    WEATHER_FEATURES,
    Window,
    feature_matrix,
    load_frame,
)
from bike_demand.model.metrics import evaluate

# 기간은 [시작, 끝) 반열린 구간이다. train 2023-01~2024-12, validation
# 2025-01~06(docs/experiments.md v1 '데이터 나누기'). 시간 순서로 나눠 미래 행이 학습에 들어가지
# 않게 한다.
# EARLY_STOP_FROM: 조기 종료용으로 train 안의 마지막 두 달(2024-11~12)을 뗀다. validation을 조기
# 종료에 쓰면 validation이 선택에 두 번 쓰이므로 train 안에서 뗀다(계획에 적은 이유).
TRAIN = ("2023-01-01", "2025-01-01")
VALIDATION = ("2025-01-01", "2025-07-01")
EARLY_STOP_FROM = "2024-11-01"

# 측정 전에 고정한 LightGBM 설정(docs/experiments.md v1 후보 절): num_leaves 127, learning_rate
# 0.05, min_data_in_leaf 200, feature_fraction·bagging_fraction 0.8(bagging_freq 1).
# objective poisson: 목표가 0 이상의 대여 '횟수'라 포아송 손실이 알맞고 예측도 음수가 나오지 않는다.
# metric l1: 조기 종료를 주 지표(MAE)와 같은 기준으로 판단한다.
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
# 최대 2,000 라운드(계획), 조기 종료 구간 l1이 50라운드 동안 나아지지 않으면 멈춘다(모듈 docstring).
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50


# 날짜 문자열을 1970-01-01부터 센 날 수로 바꾼다. frames의 day_index 열(epoch 초 // 86400)과 같은
# 척도라 바로 비교해 조기 종료 구간을 가를 수 있다.
def _day_index(day: str) -> int:
    return int(np.datetime64(day, "D").astype(np.int64))


# 학습 행(열 이름 → 배열)과 특징 목록으로 LightGBM을 두 단계로 학습한다. (1) early_stop_from
# 앞쪽으로 학습하고 뒤쪽 두 달로 조기 종료해 라운드 수를 정한다. (2) 그 라운드 수로 학습 기간 전체에
# 다시 학습한다. 출력은 (최종 모델, 라운드 수)이고 시간·l1 기록은 log 사전에 채운다.
# 왜 다시 학습하나: 조기 종료에 뗀 마지막 두 달도 학습에 쓰기 위해서다.
# consume=True면 행렬을 만든 뒤 원래 특징 열을 지워 메모리를 아낀다(학습 행이 수천만 개라 큰
# 기간에서 필요).
# 호출 위치: 이 파일의 main(M1·M2), validate_v2.py(M3), final.py(test 측정, v3 방식이 아닌 후보의
# 서비스 학습).
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
    # 대여소 코드·자치구 코드는 크기 순서에 뜻이 없는 번호라 LightGBM에 범주형으로 알려 준다.
    categorical = [f for f in features if f in CATEGORICAL]
    split = train["day_index"] < _day_index(early_stop_from)
    matrix = feature_matrix(train, features)
    if consume:  # 행렬을 만든 뒤 특징 열은 더 쓰지 않으므로 메모리에서 뺀다(큰 기간 학습용)
        for name in features:
            train.pop(name, None)
        gc.collect()

    # 1단계: 조기 종료로 라운드 수 정하기. stop_set을 fit_set 기준(reference)으로 만들어 두 묶음이
    # 같은 구간 나누기(bin)를 쓰게 한다. free_raw_data=True로 Dataset을 만든 뒤 원본 배열을 놓아
    # 메모리를 줄인다.
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
    # 조기 종료가 고른 라운드 수와 그때의 l1, 걸린 시간을 결과 JSON에 남긴다(라운드 수가 상한에
    # 닿았는지 등을 나중에 읽으려고).
    best = probe.best_iteration
    log["early_stopping_seconds"] = round(time.perf_counter() - started, 1)
    log["best_iteration"] = best
    log["early_stopping_l1"] = probe.best_score["valid_0"]["l1"]
    # 다음 단계 전에 1단계 모델과 Dataset을 지워 메모리를 돌려받는다.
    del probe, fit_set, stop_set
    gc.collect()

    # 2단계: 같은 행렬 전체(조기 종료 구간 포함)로 정한 라운드 수만큼 다시 학습한다.
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


# v3에서 쓰는 fit_lightgbm의 누수 없는 변형. 행을 바로 받지 않고 '행을 불러오는 함수' 두 개를
# 받는다. load_stop_frame은 대여소 패턴·추세를 조기 종료 구간 시작 전까지만으로 계산한 학습 행,
# load_full_frame은 학습 기간 전체로 계산한 학습 행이다.
# 왜 필요한가: v1·v2의 fit_lightgbm은 조기 종료로 뗀 두 달의 대여가 대여소 패턴(profile_mean) 계산에
# 이미 들어 있어서, 조기 종료가 고른 라운드 수가 낙관적일 수 있었다(docs/experiments.md v3).
# 왜 함수로 받나: 두 행 묶음이 각각 수천만 행이라 동시에 메모리에 두지 않으려고, 한 묶음을 다 쓰고
# 지운 뒤 다음 묶음을 불러온다.
# 호출 위치: validate_v3.py(M1'·M3'), final.py(v3 방식 serving).
def fit_lightgbm_separate_stop(
    load_stop_frame: Callable[[], dict[str, np.ndarray]],
    load_full_frame: Callable[[], dict[str, np.ndarray]],
    features: list[str],
    log: dict,
    early_stop_from: str,
) -> tuple[lgb.Booster, int]:
    """v3(T25): 조기 종료 단계와 최종 학습에 서로 다른 특징 행을 쓴다.

    load_stop_frame은 대여소 패턴·추세를 early_stop_from 이전 기간만으로 계산한 학습 기간 행이다.
    그래야 조기 종료 구간의 대여가 조기 종료 판단에 쓰이는 특징에 섞이지 않는다. 두 행 묶음을
    동시에 메모리에 두지 않도록 차례로 불러온다.
    """
    categorical = [f for f in features if f in CATEGORICAL]
    started = time.perf_counter()
    # 1단계: 조기 종료 전용 행으로 라운드 수를 정한다. 행렬과 목표 열만 남기고 원래 행 사전은 바로
    # 지운다.
    stop_frame = load_stop_frame()
    split = stop_frame["day_index"] < _day_index(early_stop_from)
    matrix = feature_matrix(stop_frame, features)
    rentals = stop_frame["rentals"]
    del stop_frame
    gc.collect()
    fit_set = lgb.Dataset(
        matrix[split], rentals[split], feature_name=features,
        categorical_feature=categorical, free_raw_data=True,
    )  # fmt: skip
    stop_set = lgb.Dataset(matrix[~split], rentals[~split], reference=fit_set, free_raw_data=True)
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
    del probe, fit_set, stop_set, matrix, rentals
    gc.collect()

    # 2단계: 1단계 자료를 모두 지운 뒤 전체 기간 패턴으로 만든 학습 행을 불러와 같은 라운드 수로
    # 다시 학습한다.
    started = time.perf_counter()
    full_frame = load_full_frame()
    matrix = feature_matrix(full_frame, features)
    rentals = full_frame["rentals"]
    del full_frame
    gc.collect()
    full_set = lgb.Dataset(
        matrix, rentals, feature_name=features, categorical_feature=categorical,
        free_raw_data=True,
    )  # fmt: skip
    booster = lgb.train(PARAMS, full_set, num_boost_round=best)
    log["refit_seconds"] = round(time.perf_counter() - started, 1)
    del full_set, matrix, rentals
    gc.collect()
    return booster, best


# 측정 전에 커밋한 v1 판정 규칙을 코드로 옮긴 것이다. 규칙 1: M1의 MAE가 B0보다 3% 이상 낮고(B0 ×
# 0.97 이하) 출퇴근 MAE도 B0보다 낮으면 M1. 규칙 2: 아니면 B0·B1 중 MAE가 낮은 것. M2는 날씨의
# 기여를 보려는 비교용이라 고르지 않는다(규칙 3).
# 선택과 함께 사람이 읽을 이유 문장을 돌려줘 결과 JSON에 남긴다.
def choose(results: dict[str, dict]) -> tuple[str, str]:
    """판정 규칙 (docs/experiments.md v1). (선택 ID, 이유)"""
    b0, m1 = results["B0"], results["M1"]
    if m1["mae"] <= b0["mae"] * 0.97 and m1["peak_mae"] < b0["peak_mae"]:
        return "M1", "M1 MAE가 B0보다 3% 이상 낮고 출퇴근 MAE도 낮음"
    best = min(("B0", "B1"), key=lambda k: results[k]["mae"])
    return best, "M1이 규칙 1을 통과하지 못해 B0·B1 중 MAE가 낮은 것"


# v1 validation 전체를 실행하고 data/models/v1/validation.json을 쓴다. 순서: validation 행 → 기준선
# B0·B1 평가 → train 행 → M1·M2 학습·평가(모델 파일도 저장) → 판정.
def main(warehouse: Path, out_dir: Path) -> dict:
    # 측정은 warehouse를 읽기만 하므로 읽기 전용으로 연다(DuckDB 파일은 쓰기 연결을 한 프로세스만
    # 가질 수 있다).
    con = duckdb.connect(str(warehouse), read_only=True)
    con.execute("SET threads = 10")
    report: dict = {"train": TRAIN, "validation": VALIDATION, "params": PARAMS, "models": {}}

    started = time.perf_counter()
    # validation 행: 대여소 패턴은 train 기간(history)으로만 계산해 validation 값이 특징에 새지 않게
    # 한다. only_active_stations=True로 validation 기간에 대여가 한 건도 없는 대여소를 뺀다(계획의
    # '평가 대상 행').
    valid = load_frame(con, Window(rows=VALIDATION, history=TRAIN), only_active_stations=True)
    report["validation_load_seconds"] = round(time.perf_counter() - started, 1)
    # evaluate가 출퇴근 시간과 '대여소×날짜×3시간 구간' 합을 계산하는 데 쓰는 열들.
    keys = (valid["station_code"], valid["day_index"], valid["hour_of_day"])
    # 기준선 예측값은 load_frame이 b0·b1 열로 미리 만들어 둔다(train에 칸이 없는 새 대여소는 전체
    # 대여소 평균으로 채움).
    for baseline in ("B0", "B1"):
        report["models"][baseline] = evaluate(valid["rentals"], valid[baseline.lower()], *keys)
        print(baseline, report["models"][baseline], flush=True)

    started = time.perf_counter()
    # train 행은 0건 행까지 모두 쓴다(only_active_stations=False). 행을 다 읽은 뒤에는 DB 연결을
    # 바로 닫는다.
    train = load_frame(con, Window(rows=TRAIN, history=TRAIN), only_active_stations=False)
    con.close()
    report["train_rows"] = int(len(train["rentals"]))
    report["train_load_seconds"] = round(time.perf_counter() - started, 1)
    print("train rows", report["train_rows"], flush=True)

    # M1은 전체 특징, M2는 날씨 5개를 뺀 특징. 둘의 차이를 '날씨의 기여'로 보고한다.
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
        # 특징별 gain을 전체 합에 대한 비율로 바꿔 큰 순으로 남긴다(결과 문서의 '특징 기여'가 이
        # 값이다).
        importance = booster.feature_importance(importance_type="gain")
        log["gain_share"] = {
            name: round(float(value / importance.sum()), 4)
            for name, value in sorted(zip(features, importance, strict=True), key=lambda x: -x[1])
        }
        report["models"][model_id] = {**evaluate(valid["rentals"], predicted, *keys), **log}
        print(model_id, {k: v for k, v in report["models"][model_id].items() if k != "features"})
        # 모델 하나를 다 쓸 때마다 파일로 저장하고 메모리에서 지운다(다음 후보 학습에 메모리를
        # 남기려고).
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


# 명령줄 실행: uv run python -m bike_demand.model.validate [--warehouse 경로] [--out 폴더].
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v1 validation 측정")
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("data/models/v1"))
    args = parser.parse_args()
    main(args.warehouse, args.out)
