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

from __future__ import annotations

import gc
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import lightgbm as lgb

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
RETRAIN = ("2023-01-01", "2025-07-01")
TEST = ("2025-07-01", "2026-07-01")
SERVING_START = "2023-01-01"
CANDIDATE_FEATURES = {
    "M1": FEATURES,
    "M2": [f for f in FEATURES if f not in WEATHER_FEATURES],
    "M3": [*FEATURES, *LEVEL_FEATURES],
    "M1'": FEATURES,
    "M3'": [*FEATURES, *LEVEL_FEATURES],
}


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


def _uses_levels(selected: str) -> bool:
    return selected.startswith("M3")


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
    test_rows = load_frame(con, Window(TEST, RETRAIN), True, with_levels=levels)
    keys = (test_rows["station_code"], test_rows["day_index"], test_rows["hour_of_day"])

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


def generation_name(version: str, selected: str, now: datetime) -> str:
    """예: v3-M3p-20260917T0412. model_version 앞부분으로도 쓴다(T29)."""
    model = selected.replace("'", "p")
    return f"{version}-{model}-{now.astimezone(KST):%Y%m%dT%H%M}"


def publish_generation(serving_root: Path, generation: str) -> None:
    """CURRENT를 임시 파일에 쓰고 os.replace로 바꾼다. 읽는 쪽은 옛 이름이나 새 이름만 본다."""
    tmp = serving_root / f"CURRENT.{os.getpid()}.tmp"
    tmp.write_text(generation + "\n", encoding="utf-8")
    os.replace(tmp, serving_root / "CURRENT")


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


def serving_window(con: duckdb.DuckDBPyConnection) -> tuple[str, str]:
    """(2023-01-01, 격자 마지막 달 다음 달 1일). 반기 갱신 뒤 다시 학습하면 새 자료까지 쓴다."""
    last = con.execute("select max(hour_start) from int_station_hour_grid").fetchone()[0]
    if last is None:
        raise SystemExit("int_station_hour_grid가 비어 있음")
    end = shift_months(f"{last:%Y-%m}-01", 1)
    if last + timedelta(hours=1) != datetime.fromisoformat(end):
        raise SystemExit(f"격자가 달 중간에서 끝남({last}). 끝난 달까지만 서비스 학습에 쓴다")
    return (SERVING_START, end)


def _checked_history_end(value: str, latest: str) -> str:
    """--history-end는 격자 끝 이전의 달 첫날만 받는다(자료가 없는 달로 늘리지 않음)."""
    try:
        day = date.fromisoformat(value)
    except ValueError:
        day = None
    if day is None or day.isoformat() != value or day.day != 1 or value > latest:
        raise SystemExit(f"--history-end는 {latest} 이하의 YYYY-MM-01이어야 함: {value}")
    return value


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
    serving_root = out_dir / "serving"
    generation = generation_name(version, selected, datetime.now(KST))
    target = serving_root / "generations" / generation
    target.mkdir(parents=True)

    def frame(window: Window) -> dict:
        con = duckdb.connect(str(warehouse), read_only=True)
        try:
            return load_frame(con, window, False, with_levels=levels, asof_stations=asof)
        finally:
            con.close()

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
    stop_from = shift_months(history[1], -2)
    log: dict = {}
    started = time.perf_counter()
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
    booster.save_model(str(target / "model.txt"))
    (target / "serving.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), "utf-8"
    )
    publish_generation(serving_root, generation)
    return report


def load_serving_model(out_dir: Path) -> tuple[str, lgb.Booster, artifacts.Artifacts]:
    """(세대 이름, 모델, 산출물). 모두 CURRENT를 한 번 읽어 정한 같은 세대에서 읽는다."""
    name, directory = current_generation(out_dir / "serving")
    booster = lgb.Booster(model_file=str(directory / "model.txt"))
    return name, booster, artifacts.load(directory / "artifacts")


if __name__ == "__main__":
    import argparse

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
