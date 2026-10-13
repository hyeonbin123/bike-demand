"""측정에서 고른 후보로 test를 한 번 재고, 서비스용 모델 세대를 만든다 (docs/experiments.md).

선택은 가장 최근 측정 버전을 따른다: validation_v3.json → validation_v2.json(M3 채택일 때) → v1.

1. test: 고른 후보를 train+validation(2023-01~2025-06)으로 다시 만들어 test(2025-07~2026-06)에서
   **한 번만** 잰다. 시작할 때 `<결과>.started` 예약 파일을 원자적으로 만들어 동시 실행과 중단 뒤
   다시 재는 것을 막는다(T27). 중단됐으면 사람이 사정을 확인한 뒤 예약 파일을 지운다.
   v2 결과는 test 기간을 두 번째로 본 것이라 그렇게 표시한다. v3는 계획대로 test를 재지 않는다.
2. serving: 같은 후보를 전체 기간(2023-01~warehouse 격자의 마지막 달)으로 학습해
   `serving/generations/<세대>/`에 모델·산출물·설명을 모두 쓴 뒤 `serving/CURRENT`(세대 이름
   한 줄)를 원자적으로 바꾼다(T28). 예측 작업은 CURRENT를 한 번 읽고 그 세대의 파일만 쓴다.
   기간 끝은 격자에서 정하므로 refresh_history로 새 반기를 넣은 뒤 다시 돌리면 그 반기까지 쓴다.

실행: uv run python -m bike_demand.model.final test | serving
"""

# [파일 역할] 측정의 마지막 두 단계를 맡는다. (1) test: validation에서 고른 후보를 test 기간에서 딱
# 한 번 잰다. (2) serving: 같은 후보를 전체 기간으로 학습해 서비스 모델 '세대'를 만들고 게시한다.
# [왜 이렇게] 측정 규칙(docs/experiments.md, README 설계 판단)이 'validation으로 고르고 test는 한
# 번만'이라, 코드가 두 번째 측정을 물리적으로 막는다(예약 파일). 서비스 모델 교체는 예측 작업이 반쯤
# 바뀐 모델·산출물을 섞어 읽지 않도록 세대 폴더 + CURRENT 포인터를 원자적으로 바꾼다(Codex 검토
# 라운드 3, 커밋 806b2c8).
# [현재 상태] v3 판정으로 M3'가 서비스 모델이고 게시된 세대는 v3-M3p-20260917T0904다(README,
# docs/experiments.md). 서비스 학습은 이 PC에서 약 3시간 걸린다.
# [연결] validate.py의 학습 함수(fit_lightgbm, fit_lightgbm_separate_stop)와
# frames.py·metrics.py·artifacts.py를 쓴다. load_serving_model은 predict.py, forecast_gap.py,
# shortage_eval.py가 부른다.

from __future__ import annotations

import gc
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# duckdb: warehouse를 읽기 전용으로 열어 행을 만든다. lightgbm: 학습한 모델 저장과 서비스 모델 읽기.
import duckdb
import lightgbm as lgb

# 프로젝트 모듈: 서비스 산출물, 특징·행 생성, 지표, 그리고 validate.py에 있는 LightGBM 학습
# 절차(측정 때와 같은 고정 설정·조기 종료 방식을 그대로 쓰기 위해).
from bike_demand.model import artifacts
from bike_demand.model.frames import (
    FEATURES,
    LEVEL_FEATURES,
    WEATHER_FEATURES,
    Window,
    feature_matrix,
    load_frame,
    shift_months,
)
from bike_demand.model.metrics import evaluate
from bike_demand.model.validate import fit_lightgbm, fit_lightgbm_separate_stop

KST = timezone(timedelta(hours=9))
# 사전 등록한 test 기간들(docs/experiments.md v1 데이터 나누기): test 때는
# train+validation(2023-01~2025-06)으로 다시 학습하고 2025-07~2026-06에서 잰다. SERVING_START는
# 서비스 학습 시작일이고, 끝은 warehouse 격자에서 정한다(serving_window).
RETRAIN = ("2023-01-01", "2025-07-01")
TEST = ("2025-07-01", "2026-07-01")
SERVING_START = "2023-01-01"
# 후보 ID → 특징 목록. M2는 날씨 제외 비교용, M3·M3'는 v2 수준 특징 포함. 프라임(')이 붙은 v3 후보는
# 특징 목록은 같고 행을 만드는 방법(시점별 대여소 정보, 누수 없는 조기 종료)만 다르다.
CANDIDATE_FEATURES = {
    "M1": FEATURES,
    "M2": [f for f in FEATURES if f not in WEATHER_FEATURES],
    "M3": [*FEATURES, *LEVEL_FEATURES],
    "M1'": FEATURES,
    "M3'": [*FEATURES, *LEVEL_FEATURES],
}


# 가장 최근 측정 결과 파일의 선택을 따른다. v3 결과가 있으면 그것, 없으면 v2가 M3를 채택했을 때만
# v2, 아니면 v1. 호출 위치: run_test, fit_serving.
def selected_model(out_dir: Path) -> tuple[str, str]:
    """(선택 ID, 측정 버전)."""
    v3 = out_dir / "validation_v3.json"
    if v3.exists():
        return json.loads(v3.read_text("utf-8"))["selected"], "v3"
    v2 = out_dir / "validation_v2.json"
    if v2.exists() and json.loads(v2.read_text("utf-8"))["selected"] == "M3":
        return "M3", "v2"
    report = json.loads((out_dir / "validation.json").read_text("utf-8"))
    return report["selected"], "v1"


# M3·M3'만 v2 수준 특징을 쓴다.
def _uses_levels(selected: str) -> bool:
    return selected.startswith("M3")


# O_CREAT | O_EXCL로 '없을 때만 만들기'를 운영체제가 한 번에 처리하게 해, 두 실행이 동시에 '아직
# 없음'을 확인하고 둘 다 test를 재는 경쟁을 막는다(커밋 806b2c8). 파일에 시작 시각과 PID를 적어,
# 중단된 실행이 남긴 것인지 사람이 확인할 수 있게 한다.
def reserve_once(path: Path) -> None:
    """path가 없을 때만 원자적으로 만든다. 이미 있으면 SystemExit."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"이미 시작된 test 실행이 있음: {path}. 중단된 실행이면 확인 후 이 파일을 지운다"
        ) from None
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(f"started {datetime.now(KST).isoformat()} pid {os.getpid()}\n")


# test 한 번 측정. 순서가 중요하다: 선택 확인 → v3면 거부(test 기간을 이미 두 번 봄) → 결과 파일이
# 있으면 거부 → 예약 파일 생성 → 그 뒤에야 test 행을 읽는다. 결과는 data/models/v1/test.json(v1)
# 또는 test_v2.json(v2, 두 번째로 본 값이라고 note에 적음).
def run_test(warehouse: Path, out_dir: Path) -> dict:
    selected, version = selected_model(out_dir)
    if version == "v3":
        raise SystemExit("v3는 계획대로 test를 재지 않는다(test 기간을 이미 두 번 봄)")
    result_path = out_dir / ("test.json" if version == "v1" else f"test_{version}.json")
    if result_path.exists():
        raise SystemExit(f"test는 한 번만 잰다: {result_path}가 이미 있음")
    reserve_once(result_path.with_suffix(".started"))

    levels = _uses_levels(selected)
    report: dict = {"selected": selected, "version": version, "train": RETRAIN, "test": TEST}
    if version != "v1":
        report["note"] = "test 기간을 두 번째로 본 결과(v1에서 M1으로 한 번 봄)"
    con = duckdb.connect(str(warehouse), read_only=True)
    # test 행: 기간에 대여가 있는 대여소만(평가 대상 규칙), 패턴·추세는 RETRAIN 기간으로 계산.
    test_rows = load_frame(con, Window(TEST, RETRAIN), True, with_levels=levels)
    keys = (test_rows["station_code"], test_rows["day_index"], test_rows["hour_of_day"])

    # 기준선이 선택됐으면 학습 없이 b0·b1 열을 예측으로 쓴다. 아니면 RETRAIN 전체로 다시 학습한다.
    # 조기 종료는 학습 기간의 마지막 두 달(2025-05~06)로 라운드 수를 정한 뒤 전체로 다시 학습하는
    # 방식이다(validate.fit_lightgbm). consume=True와 del·gc.collect는 수천만 행 학습 데이터를 예측
    # 전에 메모리에서 내리려는 것이다.
    if selected in ("B0", "B1"):
        con.close()
        report["metrics"] = evaluate(test_rows["rentals"], test_rows[selected.lower()], *keys)
    else:
        features = CANDIDATE_FEATURES[selected]
        train = load_frame(con, Window(RETRAIN, RETRAIN), False, with_levels=levels)
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


# --- 서비스 모델 세대 (T28, T29) ---------------------------------------------------------


# 세대 이름 = 측정 버전 + 후보(따옴표는 파일 이름에 쓰기 좋게 p로) + KST 학습 시작 시각. 이 이름이
# predictions의 model_version 앞부분이 되어 같은 예보라도 모델이 다르면 버전이 다르다(커밋 806b2c8).
def generation_name(version: str, selected: str, now: datetime) -> str:
    """예: v3-M3p-20260917T0412. model_version 앞부분으로도 쓴다(T29)."""
    model = selected.replace("'", "p")
    return f"{version}-{model}-{now.astimezone(KST):%Y%m%dT%H%M}"


# 새 세대를 게시하는 유일한 지점. 세대 폴더에 모든 파일을 다 쓴 뒤에만 부른다(fit_serving 끝).
# 예전에는 폴더 두 개를 이름 바꾸기로 교체해 모델이 잠깐 없거나 옛 모델과 새 산출물이 섞일 수
# 있었다(커밋 806b2c8).
def publish_generation(serving_root: Path, generation: str) -> None:
    """CURRENT를 임시 파일에 쓰고 os.replace로 바꾼다. 읽는 쪽은 옛 이름이나 새 이름만 본다."""
    tmp = serving_root / f"CURRENT.{os.getpid()}.tmp"
    tmp.write_text(generation + "\n", encoding="utf-8")
    os.replace(tmp, serving_root / "CURRENT")


# CURRENT 한 줄을 읽어 세대 폴더를 정한다. 세대 구조(커밋 806b2c8) 이전의 단일 폴더
# 구조(serving/model.txt)도 읽을 수 있게 남겨 두었다. 호출 위치: load_serving_model.
def current_generation(serving_root: Path) -> tuple[str, Path]:
    """(세대 이름, 세대 폴더). CURRENT가 없고 예전 구조(serving/model.txt)면 그 폴더를 쓴다."""
    pointer = serving_root / "CURRENT"
    if pointer.exists():
        name = pointer.read_text(encoding="utf-8").strip()
        return name, serving_root / "generations" / name
    if (serving_root / "model.txt").exists():
        info = json.loads((serving_root / "serving.json").read_text("utf-8"))
        return info.get("version", "v1"), serving_root
    raise FileNotFoundError(f"서비스 모델이 없음: {serving_root}")


# 서비스 학습 기간 = 2023-01-01 ~ 격자 마지막 시간이 든 달의 다음 달 1일(제외). 커밋 b72c9cc에서
# 상수 기간을 이것으로 바꿨다. 상수로 두면 반기 갱신 뒤 다시 학습해도 새 반기를 보지 않고, 산출물의
# 자료 끝도 그대로라 2027-02부터 수준 특징이 모두 결측이 될 수 있었다.
# 격자가 달 중간에서 끝나면(마지막 시간 + 1시간이 다음 달 1일이 아니면) 거부한다. 덜 찬 달이
# 학습·조기 종료 구간에 들어가지 않게 하려는 것이다.
def serving_window(con: duckdb.DuckDBPyConnection) -> tuple[str, str]:
    """(2023-01-01, 격자 마지막 달 다음 달 1일). 반기 갱신 뒤 다시 학습하면 새 자료까지 쓴다."""
    last = con.execute("select max(hour_start) from int_station_hour_grid").fetchone()[0]
    if last is None:
        raise SystemExit("int_station_hour_grid가 비어 있음")
    end = shift_months(f"{last:%Y-%m}-01", 1)
    if last + timedelta(hours=1) != datetime.fromisoformat(end):
        raise SystemExit(f"격자가 달 중간에서 끝남({last}). 끝난 달까지만 서비스 학습에 쓴다")
    return (SERVING_START, end)


# --history-end를 검사한다: 실제 날짜이고, 달의 1일이고, 2023-04-01 이상 격자 끝 이하. 하한
# 2023-04-01은 끝 두 달(조기 종료 구간) 앞에 학습할 달이 한 달 이상 남는 가장 이른 값이다. Codex
# 교차 검토(라운드 11, 커밋 90d35aa)에서 더 이른 값이 세대 폴더를 만든 뒤 LightGBM 'num_data > 0'
# 오류로 실패하는 것이 드러났다.
def _checked_history_end(value: str, latest: str) -> str:
    """--history-end는 격자 끝 이전의 달 첫날만 받는다(자료가 없는 달로 늘리지 않음).

    끝 2개월은 조기 종료에 쓰므로 그 앞에 학습할 달이 한 달 이상 남아야 한다.
    """
    earliest = shift_months(SERVING_START, 3)
    try:
        day = date.fromisoformat(value)
    except ValueError:
        day = None
    if day is None or day.isoformat() != value or day.day != 1 or not earliest <= value <= latest:
        raise SystemExit(
            f"--history-end는 {earliest} 이상 {latest} 이하의 YYYY-MM-01이어야 함: {value}"
        )
    return value


# 서비스 세대 하나를 학습해 게시한다. 기준선이 선택됐으면 LightGBM 모델이 없으므로 거부. v3 선택이면
# v3 방법(asof_stations, 분리 조기 종료)을 그대로 따른다.
def fit_serving(warehouse: Path, out_dir: Path, history_end: str | None = None) -> dict:
    """history_end: 학습 기간 끝(제외). 기본은 warehouse 격자의 마지막 달 다음 달 1일."""
    selected, version = selected_model(out_dir)
    if selected not in CANDIDATE_FEATURES:
        raise SystemExit(f"기준선({selected})이 선택되어 서비스용 LightGBM 모델이 없음")
    levels = _uses_levels(selected)
    asof = version == "v3"
    features = CANDIDATE_FEATURES[selected]
    # 기간을 먼저 정한다. 거부되면 빈 세대 폴더를 남기지 않는다
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        history = serving_window(con)
    finally:
        con.close()
    if history_end is not None:
        history = (SERVING_START, _checked_history_end(history_end, history[1]))
    # 세대 폴더는 이름에 분 단위 시각이 들어가 겹치지 않는다. mkdir에 exist_ok를 주지 않아, 같은
    # 이름이 이미 있으면 덮어쓰지 않고 실패한다.
    serving_root = out_dir / "serving"
    generation = generation_name(version, selected, datetime.now(KST))
    target = serving_root / "generations" / generation
    target.mkdir(parents=True)

    # 필요할 때마다 warehouse를 읽기 전용으로 열고 닫아, 두 큰 행 묶음을 동시에 들고 있지 않게 하는
    # 지연 로더.
    def frame(window: Window) -> dict:
        con = duckdb.connect(str(warehouse), read_only=True)
        try:
            return load_frame(con, window, False, with_levels=levels, asof_stations=asof)
        finally:
            con.close()

    # 서비스 예측이 쓸 산출물(번호표·패턴·추세·반기 수준)을 같은 학습 기간으로 먼저 세대 폴더에
    # 저장한다.
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        meta = artifacts.export(con, history, target / "artifacts", with_levels=levels)
    finally:
        con.close()
    report: dict = {
        "selected": selected,
        "version": version,
        "generation": generation,
        "history": history,
        "artifacts": meta,
    }
    # 조기 종료 구간 = 학습 기간의 마지막 두 달(측정과 같은 규칙).
    stop_from = shift_months(history[1], -2)
    log: dict = {}
    started = time.perf_counter()
    # v3: 조기 종료 단계의 행은 조기 종료 구간 시작 전까지로 계산한 패턴·추세를 쓰고(누수 방지,
    # docs/experiments.md v3), 최종 학습 행은 전체 기간 패턴을 쓴다. 두 묶음을 차례로 불러 메모리에
    # 하나씩만 둔다. v1·v2는 한 묶음으로 조기 종료와 재학습을 한다.
    if asof:
        booster, _ = fit_lightgbm_separate_stop(
            lambda: frame(Window(history, (history[0], stop_from))),
            lambda: frame(Window(history, history)),
            features,
            log,
            stop_from,
        )
    else:
        booster, _ = fit_lightgbm(
            frame(Window(history, history)), features, log, early_stop_from=stop_from, consume=True
        )
    report["seconds"] = round(time.perf_counter() - started, 1)
    report["training"] = log
    # 모델(model.txt)과 학습 보고(serving.json)를 세대 폴더에 쓴 뒤 마지막으로 CURRENT를 바꾼다. 이
    # 순서라 예측 작업은 다 쓰인 세대만 본다.
    booster.save_model(str(target / "model.txt"))
    (target / "serving.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), "utf-8"
    )
    publish_generation(serving_root, generation)
    return report


# CURRENT를 한 번만 읽어 정한 세대에서 모델과 산출물을 함께 읽는다. 둘을 따로 읽다 그 사이 CURRENT가
# 바뀌어 세대가 섞이는 일을 막는다. 호출 위치: predict.py 실행부, forecast_gap.py, shortage_eval.py.
def load_serving_model(out_dir: Path) -> tuple[str, lgb.Booster, artifacts.Artifacts]:
    """(세대 이름, 모델, 산출물). 모두 CURRENT를 한 번 읽어 정한 같은 세대에서 읽는다."""
    name, directory = current_generation(out_dir / "serving")
    booster = lgb.Booster(model_file=str(directory / "model.txt"))
    return name, booster, artifacts.load(directory / "artifacts")


if __name__ == "__main__":
    import argparse

    # 실행: python -m bike_demand.model.final test | serving [--history-end YYYY-MM-01]. 기본 출력
    # 폴더는 data/models/v1(측정 결과 JSON과 serving/이 같이 있음).
    parser = argparse.ArgumentParser(description="test 한 번 측정과 서비스용 학습")
    parser.add_argument("step", choices=["test", "serving"])
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("data/models/v1"))
    parser.add_argument(
        "--history-end", help="serving 학습 기간 끝(제외) YYYY-MM-01, 기본은 격자 끝 다음 달"
    )
    args = parser.parse_args()
    if args.step == "test":
        run_test(args.warehouse, args.out)
    else:
        report = fit_serving(args.warehouse, args.out, args.history_end)
        print(json.dumps(report, ensure_ascii=False, default=str))
