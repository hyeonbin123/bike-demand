"""학습 재현성 확인(진단, 후보 선택 아님): 같은 창을 다시 학습해 같은 모델이 나오는지 잰다.

docs/experiments.md '측정 코드 변경: 학습 행 순서 고정과 보조 지표' 절의 가설 H1을 재는 도구다.
한 번 실행하면 학습 행을 warehouse에서 불러 LightGBM을 고정 라운드 수로 학습하고, 평가 행을
예측한 뒤 아래를 JSON으로 적는다(모델·예측은 바이트 지문으로만 남긴다).

- 학습 행 순서 지문(station_code·day_index·hour_of_day 배열의 SHA-256)
- 모델 문자열(`Booster.model_to_string()`)의 SHA-256
- 평가 행을 (station_code, day_index, hour_of_day) 순으로 놓은 예측의 SHA-256과 그 예측
  파일(`<결과 이름>.pred.npy`, float64)
- metrics.evaluate의 지표(MAE, Poisson deviance, 예측 평균 비율 등), 걸린 시간, 라이브러리 버전,
  코드 커밋

기본 창은 규칙에 적은 것이다: 학습 행 2023-11-01~2024-01-01, 대여소 패턴·추세(history)
2023-01-01~2023-11-01, 평가 2024-01-01~2024-02-01(그 달에 대여가 있는 대여소), v3 방법(행보다
앞선 대여소 정보 스냅샷 + 수준 특징), M3' 특징 23개, `validate.PARAMS`(seed 42, num_threads 10),
100라운드 고정(조기 종료 없음). CPU만 쓰고 한 번 약 7분 걸린다. warehouse는 읽기 전용으로 연다.

실행 (저장소 루트에서):
  uv run python scripts/check_training_determinism.py --out data/models/determinism/sorted-1.json
  uv run python scripts/check_training_determinism.py --out data/models/determinism/sorted-2.json
  uv run python scripts/check_training_determinism.py --no-sort --out .../no-sort-1.json   (두 번)
  uv run python scripts/check_training_determinism.py --compare A.json B.json
"""

# [파일 역할] 2026-10-24에 `frames.load_frame`이 행을 (station_code, day_index, hour_of_day) 순으로
# 놓도록 바꾼 뒤, 그 변경이 목표(같은 코드·자료·seed면 같은 모델)를 이루는지 확인하는 읽기 전용
# 스크립트다. 결과는 후보 선택에 쓰지 않고 '정렬 커밋을 유지할지'만 정한다(규칙 절의 허용 폭: 차이
# 0). `--no-sort`는 정렬 전 동작(DuckDB가 돌려준 순서 그대로)을 재현해 '정렬 없이는 달랐다'를 같은
# 도구로 보여 주기 위한 것이다.
# [흐름] run: 학습 행 불러오기(load_frame) → 행렬(feature_matrix) → lgb.train(PARAMS, 고정 라운드) →
# 평가 행 불러오기 → 예측 → evaluate → 지문·지표 JSON과 예측 .npy 쓰기. compare: 두 결과 JSON과
# 예측 파일을 읽어 설정이 같은지, 모델·예측 지문이 같은지, 예측 최대 절대 차가 얼마인지 보고한다.
# [왜 새 프로세스마다 한 번인가] 같은 프로세스 안에서 두 번 학습하면 DuckDB 연결·numpy 할당 상태가
# 이어져 우연히 같은 행 순서가 나올 수 있다. 규칙은 '별도 프로세스로 두 번'이라 실행마다 한 번만
# 학습하고 비교는 --compare로 한다.
# [연결] bike_demand.model.frames(load_frame·정렬 키·feature_matrix),
# bike_demand.model.validate.PARAMS(측정·서비스 학습과 같은 설정),
# bike_demand.model.metrics.evaluate, bike_demand.model.provenance(라이브러리 버전·코드 커밋).
# tests/test_check_training_determinism.py가 작은 warehouse 파일과 적은 라운드로 run·compare를
# 확인한다.

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# duckdb: warehouse 파일을 읽기 전용으로 열어 행을 만든다. lightgbm: 고정 라운드 학습과 모델 문자열.
# numpy: 지문 계산(tobytes)과 예측 저장(.npy).
import duckdb
import lightgbm as lgb
import numpy as np

from bike_demand.model import provenance
from bike_demand.model.frames import (
    CATEGORICAL,
    FEATURES,
    LEVEL_FEATURES,
    Window,
    feature_matrix,
    load_frame,
)
from bike_demand.model.metrics import evaluate
from bike_demand.model.validate import PARAMS

KST = timezone(timedelta(hours=9))
# 규칙에 적은 기본 창([시작, 끝) 반열린 구간). 학습 두 달은 '별도 프로세스로 두 번 학습'을 한 번에
# 약 7분으로 끝낼 수 있는 크기이고, 평가는 그 다음 한 달이다. 대여소 패턴·추세는 학습 행보다 앞선
# 기간(history)으로만 계산해 측정과 같은 누수 없는 구조를 지킨다.
DEFAULT_HISTORY = ("2023-01-01", "2023-11-01")
DEFAULT_TRAIN_ROWS = ("2023-11-01", "2024-01-01")
DEFAULT_EVAL_ROWS = ("2024-01-01", "2024-02-01")
# 라운드 수를 고정하는 이유: 조기 종료가 고른 라운드 수 자체가 행 순서에 따라 달라질 수 있어,
# 그것까지 섞이면 '같은 라운드 수에서 같은 모델인가'를 따로 볼 수 없다.
DEFAULT_ROUNDS = 100
# M3'(서비스 모델)의 특징 순서. 학습·서비스와 같은 목록을 쓴다.
MODEL_FEATURES = [*FEATURES, *LEVEL_FEATURES]
# DuckDB 스레드 수. validate_v3·final과 같은 값이다(행 순서가 달라지는 원인 중 하나가 병렬 실행이라
# 측정과 같은 조건으로 둔다).
DUCKDB_THREADS = 10
# 결과 JSON에서 '같은 조건의 재현 확인'인지 가르는 설정 키. compare가 이 키들이 모두 같을 때만
# same_settings=True로 적는다.
SETTING_KEYS = ("sort_rows", "windows", "features", "params", "num_boost_round", "warehouse")


# 결과 JSON 경로에서 예측 파일 경로를 정한다: 같은 폴더, 같은 이름 + '.pred.npy'
# (sorted-1.json → sorted-1.pred.npy). 호출 위치: run, compare, tests.
def prediction_path(out: Path) -> Path:
    out = Path(out)
    return out.with_name(out.stem + ".pred.npy")


# SHA-256 16진수 문자열. 배열은 메모리 바이트(tobytes)로, 문자열은 UTF-8로 재서 같은 값·같은
# dtype이면 같은 지문이 나온다.
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# 세 키 열(대여소 번호, 날짜 번호, 시각)을 이어 붙인 바이트의 SHA-256. 행 순서가 다르면 값이 다르고,
# 같은 순서면 같다. 정렬 전 동작(--no-sort)에서 '두 실행의 행 순서가 실제로 달랐다'를 보이는 값이다.
def row_order_digest(frame: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("station_code", "day_index", "hour_of_day"):
        digest.update(frame[name].tobytes())
    return digest.hexdigest()


# warehouse를 읽기 전용으로 열어 v3 방법(asof_stations, 수준 특징)으로 행을 불러오고 바로 닫는다.
# sort_rows=False면 load_frame의 정렬을 끄고 DuckDB가 돌려준 순서 그대로 둔다(정렬 전 동작 재현).
def load(warehouse: Path, window: Window, active: bool, sort_rows: bool) -> dict[str, np.ndarray]:
    con = duckdb.connect(str(warehouse), read_only=True)
    con.execute(f"SET threads = {DUCKDB_THREADS}")
    try:
        return load_frame(
            con, window, active, with_levels=True, asof_stations=True, sort_rows=sort_rows
        )
    finally:
        con.close()


# 한 번의 확인 실행. 출력: 결과 사전(JSON으로 out에 쓴 것과 같음). 부작용: out(JSON)과 예측 파일
# (.pred.npy)을 쓴다(폴더가 없으면 만든다). warehouse는 읽기만 한다.
# history·train_rows·eval_rows·rounds는 테스트가 작은 창으로 바꿔 넣는 자리이고, 규칙의 측정은
# 기본값으로 돈다.
def run(
    warehouse: Path,
    out: Path,
    sort_rows: bool = True,
    rounds: int = DEFAULT_ROUNDS,
    history: tuple[str, str] = DEFAULT_HISTORY,
    train_rows: tuple[str, str] = DEFAULT_TRAIN_ROWS,
    eval_rows: tuple[str, str] = DEFAULT_EVAL_ROWS,
) -> dict:
    """학습 → 예측 → 지문·지표를 out(JSON)과 예측 파일에 쓰고 결과 사전을 돌려준다."""
    warehouse, out = Path(warehouse), Path(out)
    categorical = [f for f in MODEL_FEATURES if f in CATEGORICAL]
    seconds: dict[str, float] = {}
    started = time.perf_counter()

    # 학습 행: 0건 행까지 모두(active=False). 행렬과 목표만 남기고 사전은 지운다(메모리).
    step = time.perf_counter()
    train = load(warehouse, Window(train_rows, history), False, sort_rows)
    train_order = row_order_digest(train)
    train_count = int(len(train["rentals"]))
    matrix = feature_matrix(train, MODEL_FEATURES)
    target = train["rentals"]
    del train
    seconds["load_train"] = round(time.perf_counter() - step, 1)

    # 고정 라운드 학습. Dataset을 만들 때 구간 경계 표본(bin_construct_sample_cnt)과 bagging이 행
    # 번호를 따르므로, 행 순서가 다르면 여기서 모델이 달라진다.
    step = time.perf_counter()
    dataset = lgb.Dataset(
        matrix, target, feature_name=MODEL_FEATURES, categorical_feature=categorical
    )
    booster = lgb.train(PARAMS, dataset, num_boost_round=rounds)
    model_string = booster.model_to_string()
    seconds["train"] = round(time.perf_counter() - step, 1)
    del dataset, matrix, target

    # 평가 행: 그 달에 대여가 있는 대여소만(평가 대상 규칙). 정렬을 끈 실행이라도 예측 지문은 늘
    # (station_code, day_index, hour_of_day) 순으로 놓은 뒤 재야 두 실행을 같은 행끼리 비교할 수
    # 있다.
    step = time.perf_counter()
    evaluation = load(warehouse, Window(eval_rows, history), True, sort_rows)
    if not sort_rows:
        order = np.lexsort(
            (evaluation["hour_of_day"], evaluation["day_index"], evaluation["station_code"])
        )
        for name in evaluation:
            evaluation[name] = evaluation[name][order]
    seconds["load_eval"] = round(time.perf_counter() - step, 1)

    step = time.perf_counter()
    predicted = booster.predict(
        feature_matrix(evaluation, MODEL_FEATURES), num_threads=PARAMS["num_threads"]
    )
    predicted = np.asarray(predicted, dtype=np.float64)
    metrics = evaluate(
        evaluation["rentals"],
        predicted,
        evaluation["station_code"],
        evaluation["day_index"],
        evaluation["hour_of_day"],
    )
    seconds["predict"] = round(time.perf_counter() - step, 1)
    seconds["total"] = round(time.perf_counter() - started, 1)

    result = {
        "recorded_at": datetime.now(KST).isoformat(timespec="seconds"),
        "sort_rows": bool(sort_rows),
        "warehouse": warehouse.as_posix(),
        "windows": {
            "history": list(history),
            "train_rows": list(train_rows),
            "eval_rows": list(eval_rows),
        },
        "features": MODEL_FEATURES,
        "params": PARAMS,
        "num_boost_round": int(rounds),
        "rows": {"train": train_count, "eval": int(len(predicted))},
        "train_order_sha256": train_order,
        "model_sha256": sha256_bytes(model_string.encode("utf-8")),
        "prediction_sha256": sha256_bytes(predicted.tobytes()),
        "prediction_file": prediction_path(out).name,
        "metrics": metrics,
        "seconds": seconds,
        "versions": provenance.package_versions(),
        "git": provenance.git_state(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path(out), predicted)
    # LF로 쓴다(.gitattributes와 같은 줄 끝). 결과는 data/ 아래라 커밋되지 않지만 형식을 맞춘다.
    with open(out, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


# 두 결과를 대조한다. 출력: 설정이 같은지(다르면 어느 키가 다른지), 학습 행 순서·모델·예측 지문이
# 같은지, 예측 최대 절대 차(규칙의 허용 폭 0), 두 MAE. 예측 파일이 없으면 지문만 비교하고 차이는
# None. 멈추지 않고 보고만 한다(--no-sort 두 번은 다르게 나와야 정상이므로 종료 코드로 판정하지
# 않음).
def compare(first: Path, second: Path) -> dict:
    """두 결과 JSON(과 예측 파일)을 읽어 같은지 보고하는 사전."""
    a = json.loads(Path(first).read_text("utf-8"))
    b = json.loads(Path(second).read_text("utf-8"))
    different = [key for key in SETTING_KEYS if a.get(key) != b.get(key)]
    report: dict = {
        "first": Path(first).as_posix(),
        "second": Path(second).as_posix(),
        "same_settings": not different,
        "different_settings": different,
        "train_order_equal": a["train_order_sha256"] == b["train_order_sha256"],
        "model_equal": a["model_sha256"] == b["model_sha256"],
        "prediction_equal": a["prediction_sha256"] == b["prediction_sha256"],
        "max_abs_prediction_diff": None,
        "mae": [a["metrics"]["mae"], b["metrics"]["mae"]],
        "predicted_mean": [a["metrics"]["predicted_mean"], b["metrics"]["predicted_mean"]],
    }
    pa, pb = prediction_path(Path(first)), prediction_path(Path(second))
    if pa.exists() and pb.exists():
        x, y = np.load(pa), np.load(pb)
        if x.shape == y.shape:
            report["max_abs_prediction_diff"] = float(np.abs(x - y).max()) if len(x) else 0.0
        else:
            report["prediction_shapes"] = [list(x.shape), list(y.shape)]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "학습 재현성 확인(진단): 고정 창을 LightGBM으로 한 번 학습해 모델·예측 지문과 지표를 "
            "JSON으로 남긴다. 새 프로세스로 두 번 돌린 결과를 --compare로 대조한다"
        ),
        epilog=(
            "기본 창: 학습 2023-11-01~2024-01-01, 패턴·추세 2023-01-01~2023-11-01, 평가 "
            "2024-01-01~2024-02-01, M3' 특징, validate.PARAMS, 100라운드. CPU만, 한 번 약 7분"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "결과 JSON 경로. 예측은 같은 이름의 .pred.npy로 옆에 쓴다. 생략하면 "
            "data/models/determinism/<sorted|no-sort>-<시각>.json"
        ),
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="load_frame의 행 정렬을 끄고 DuckDB가 돌려준 순서 그대로 학습한다(정렬 전 동작 재현)",
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS, help=f"학습 라운드 수(기본 {DEFAULT_ROUNDS})"
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=Path("data/warehouse/bike_demand.duckdb"),
        help="dbt warehouse 파일(읽기 전용으로 연다)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("A", "B"),
        help="학습하지 않고 두 결과 JSON을 대조해 보고만 한다",
    )
    args = parser.parse_args()

    if args.compare is not None:
        print(json.dumps(compare(*args.compare), ensure_ascii=False, indent=2))
    else:
        out = args.out
        if out is None:
            label = "no-sort" if args.no_sort else "sorted"
            stamp = f"{datetime.now(KST):%Y%m%dT%H%M%S}"
            out = Path("data/models/determinism") / f"{label}-{stamp}.json"
        result = run(args.warehouse, out, sort_rows=not args.no_sort, rounds=args.rounds)
        shown = ("sort_rows", "rows", "train_order_sha256", "model_sha256", "prediction_sha256")
        summary = {key: result[key] for key in shown}
        summary["mae"] = result["metrics"]["mae"]
        summary["predicted_mean"] = result["metrics"]["predicted_mean"]
        summary["seconds"] = result["seconds"]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"written: {out} and {prediction_path(out)}")
