# [무엇을 검증하나] 학습 재현성 확인 스크립트 scripts/check_training_determinism.py가 작은 warehouse
# 파일에서 끝까지 돌고(행 불러오기 → 100라운드 대신 몇 라운드 학습 → 예측 → 지문·지표 JSON과 예측
# .npy 쓰기), 같은 입력을 두 번 돌리면 모델·예측 지문이 같다고 `compare`가 보고하는지 확인한다.
# 실제 규칙의 창(2023-11~12 학습, 한 번 약 7분)은 여기서 돌리지 않고, 창과 라운드 수를 작게 넘긴다.
# [왜 중요한가] 이 스크립트는 docs/experiments.md '학습 행 순서 고정' 절의 H1(정렬 뒤 두 번 학습한
# 모델 SHA-256이 같고 예측 최대 절대 차 0)을 재는 유일한 도구다. 긴 실행을 하기 전에 입출력 형식과
# 대조 논리가 맞는지 작은 자료로 확인한다.

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# scripts/는 패키지가 아니라 import할 수 없으므로 파일 경로로 불러온다
# (tests/test_model_provenance.py와 같은 방식). if __name__ == '__main__' 아래 실행 부분은 돌지
# 않는다.
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_training_determinism.py"


@pytest.fixture
def determinism():
    spec = importlib.util.spec_from_file_location("check_training_determinism", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# small_warehouse(메모리)를 DuckDB 파일로 옮기고, v3 방식(asof_stations)이 읽는 stg_stations(대여소
# 정보 스냅샷)를 더한다. 스크립트는 warehouse를 경로로 받아 읽기 전용으로 열기 때문에 파일이
# 필요하다.
@pytest.fixture
def warehouse_file(small_warehouse, tmp_path):
    path = tmp_path / "warehouse.duckdb"
    small_warehouse.execute(f"attach '{path.as_posix()}' as f")
    for table in ("dim_stations", "dim_hours", "int_station_hour_grid"):
        small_warehouse.execute(f"create table f.{table} as select * from {table}")
    small_warehouse.execute("""
        create table f.stg_stations as
        select * from (values (1, '강남구', 10, 37.5, 127.0, '2023-12'),
                              (2, '마포구', 12, 37.6, 126.9, '2023-12'))
            t(station_no, district, docks, lat, lon, snapshot)
    """)
    small_warehouse.execute("detach f")
    return path


# 작은 창: 패턴은 1/1~1/15, 학습 행 1/16~1/23, 평가 행 1/24~1/31(두 대여소 모두 8시에 대여가 있어
# 평가 대상). 라운드 수 3.
SMALL = {
    "history": ("2024-01-01", "2024-01-16"),
    "train_rows": ("2024-01-16", "2024-01-24"),
    "eval_rows": ("2024-01-24", "2024-02-01"),
    "rounds": 3,
}


# 한 번 돌리면 결과 JSON(규칙에 적은 항목)과 예측 파일이 생기고, 같은 설정으로 다시 돌린 결과와
# compare가 '같음'(모델·예측 지문 같음, 예측 최대 절대 차 0)을 보고한다.
def test_run_writes_the_record_and_compare_reports_identical_runs(
    determinism, warehouse_file, tmp_path
):
    first = tmp_path / "out" / "sorted-1.json"
    second = tmp_path / "out" / "sorted-2.json"
    a = determinism.run(warehouse_file, first, sort_rows=True, **SMALL)
    b = determinism.run(warehouse_file, second, sort_rows=True, **SMALL)

    saved = json.loads(first.read_text("utf-8"))
    assert saved["model_sha256"] == a["model_sha256"]
    assert saved["sort_rows"] is True
    assert saved["num_boost_round"] == 3
    assert saved["rows"] == {"train": 8 * 24 * 2, "eval": 8 * 24 * 2}
    assert saved["windows"] == {
        "history": list(SMALL["history"]),
        "train_rows": list(SMALL["train_rows"]),
        "eval_rows": list(SMALL["eval_rows"]),
    }
    assert saved["features"][:2] == ["station_code", "district_code"]
    assert len(saved["features"]) == 23
    assert saved["params"]["seed"] == 42
    assert {"mae", "poisson_deviance", "mean_ratio"} <= set(saved["metrics"])
    assert len(saved["model_sha256"]) == 64 and len(saved["prediction_sha256"]) == 64
    assert len(saved["train_order_sha256"]) == 64
    assert saved["versions"]["lightgbm"]
    predictions = np.load(determinism.prediction_path(first))
    assert predictions.shape == (8 * 24 * 2,)

    assert a["model_sha256"] == b["model_sha256"]
    report = determinism.compare(first, second)
    assert report["same_settings"] is True
    assert report["model_equal"] is True
    assert report["prediction_equal"] is True
    assert report["max_abs_prediction_diff"] == 0.0
    assert report["mae"] == [a["metrics"]["mae"], b["metrics"]["mae"]]


# --no-sort(정렬 전 동작 재현)도 끝까지 돌고 결과에 그렇게 표시된다. 정렬을 켠 결과와 settings가
# 다르다고 compare가 알린다(sort_rows가 다르면 같은 조건의 재현 확인이 아니다).
def test_no_sort_run_is_marked_and_compare_flags_different_settings(
    determinism, warehouse_file, tmp_path
):
    sorted_out = tmp_path / "sorted.json"
    unsorted_out = tmp_path / "no-sort.json"
    determinism.run(warehouse_file, sorted_out, sort_rows=True, **SMALL)
    result = determinism.run(warehouse_file, unsorted_out, sort_rows=False, **SMALL)
    assert result["sort_rows"] is False
    assert determinism.prediction_path(unsorted_out).exists()
    report = determinism.compare(sorted_out, unsorted_out)
    assert report["same_settings"] is False
    assert "sort_rows" in report["different_settings"]
