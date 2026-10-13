# [무엇을 검증하나] 측정 마무리와 서비스 모델 게시를 맡는 model/final.py의 안전장치를 확인한다.
# - test는 한 번만: 예약 파일(.started)을 원자적으로 만들어 동시 실행·재실행을 거부하고, 데이터를
#   읽기 전에 멈춘다. v3 후보로는 test를 재지 않는다
# - 서비스 모델 세대: CURRENT 포인터를 원자적으로 바꾸고, 예전 폴더 구조도 읽으며, 세대 이름이 측정
#   버전·후보·시각을 구분한다
# - 서비스 학습 기간: warehouse 격자 끝(마지막 달 다음 달 1일)을 따르고, 달 중간에서 끝나면
#   거부하며, --history-end는 조기 종료 앞에 학습할 달이 남는 달 첫날로 줄이는 것만 허용한다
# [왜 중요한가] 프로젝트 측정 규칙의 'test는 고른 것 하나로 한 번만'을 코드로 강제하는 곳이다(README
# 설계 판단). 서비스 쪽은 반기 갱신 뒤 재학습이 새 반기를 놓치거나(b72c9cc에서 고침), 잘못된
# 기간으로 빈 세대 폴더를 남기는 일(90d35aa)을 막는다.
# [방식] 실제 LightGBM 학습 대신 fit_lightgbm_separate_stop을 가짜로 바꿔 기간과 행 수만 기록하고,
# warehouse는 임시 DuckDB 파일로 만든다.

import json
import math
import threading
from datetime import datetime, timedelta, timezone

# final: 테스트 대상. artifacts·predict: 게시된 세대의 산출물을 다시 읽어 수준 특징이 채워지는지
# 확인한다.
import duckdb
import pytest

from bike_demand.model import artifacts, final, predict

KST = timezone(timedelta(hours=9))


# 같은 예약 파일로 두 번째 시작은 '이미 시작된' SystemExit. 중단 뒤 모르고 다시 재는 것을 막는다.
def test_reserve_once_allows_only_one_start(tmp_path):
    path = tmp_path / "test_v2.started"
    final.reserve_once(path)
    with pytest.raises(SystemExit, match="이미 시작된"):
        final.reserve_once(path)


# 스레드 8개가 Barrier로 동시에 예약을 시도해도 정확히 하나만 통과한다. '파일이 있는지 보고
# 만들기'가 아니라 O_CREAT | O_EXCL 한 번으로 원자적으로 만들기 때문이다.
def test_concurrent_reservations_let_exactly_one_through(tmp_path):
    path = tmp_path / "test.started"
    barrier = threading.Barrier(8)
    outcomes = []

    def attempt():
        barrier.wait()
        try:
            final.reserve_once(path)
            outcomes.append("ok")
        except SystemExit:
            outcomes.append("refused")

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["ok"] + ["refused"] * 7


# 예약 파일이 이미 있으면 run_test가 test 데이터를 읽기 전에 멈춘다. load_frame을 '부르면 실패'로
# 바꿔, 거부 전에 test 행을 한 번이라도 보는 일이 없는지 확인한다.
def test_run_test_refuses_before_touching_data_when_reserved(tmp_path, monkeypatch):
    (tmp_path / "validation.json").write_text(json.dumps({"selected": "M1"}), "utf-8")
    (tmp_path / "test.started").write_text("started", "utf-8")
    monkeypatch.setattr(final, "load_frame", lambda *a, **k: pytest.fail("test 데이터를 읽음"))
    with pytest.raises(SystemExit, match="이미 시작된"):
        final.run_test(tmp_path / "wh.duckdb", tmp_path)


# validation_v3.json이 있으면 선택은 (M3', v3)이고, v3는 계획대로 test를 재지 않는다(test 기간을
# 이미 두 번 봄, docs/experiments.md v3).
def test_v3_never_measures_test(tmp_path):
    (tmp_path / "validation_v3.json").write_text(json.dumps({"selected": "M3'"}), "utf-8")
    assert final.selected_model(tmp_path) == ("M3'", "v3")
    with pytest.raises(SystemExit, match="v3"):
        final.run_test(tmp_path / "wh.duckdb", tmp_path)


# 세대 두 개를 차례로 게시하면 CURRENT가 가리키는 세대와 폴더가 바뀌고, 교체에 쓴 임시 파일은 남지
# 않는다. 예측 작업은 CURRENT를 한 번 읽고 그 세대 파일만 쓰므로, 학습 중에도 반쯤 쓴 모델을 읽지
# 않는다(README 설계 판단).
def test_generation_pointer_switches_atomically_and_reads_one_generation(tmp_path):
    root = tmp_path / "serving"
    for name in ("v2-M3-20260917T0100", "v3-M3p-20260917T0400"):
        (root / "generations" / name).mkdir(parents=True)
    final.publish_generation(root, "v2-M3-20260917T0100")
    assert final.current_generation(root) == (
        "v2-M3-20260917T0100",
        root / "generations" / "v2-M3-20260917T0100",
    )
    final.publish_generation(root, "v3-M3p-20260917T0400")
    assert final.current_generation(root)[0] == "v3-M3p-20260917T0400"
    assert sorted(p.name for p in root.iterdir()) == ["CURRENT", "generations"]  # 임시 파일 없음


# 세대 구조 이전의 serving/model.txt 구조도 읽을 수 있다(serving.json에 version이 없으면 'v1').
def test_legacy_layout_is_still_readable(tmp_path):
    root = tmp_path / "serving"
    root.mkdir()
    (root / "model.txt").write_text("", "utf-8")
    (root / "serving.json").write_text(json.dumps({"selected": "M1"}), "utf-8")
    assert final.current_generation(root) == ("v1", root)


# 세대 이름은 '측정 버전-후보-KST 시각'이고 작은따옴표는 p로 바꾼다(M3' → M3p). 이 이름이 예측의
# model_version 앞부분이 되므로, 같은 예보 발표를 다른 모델로 예측해도 버전이 겹치지 않아야 한다.
def test_generation_name_distinguishes_models_for_the_same_forecast():
    now = datetime(2026, 9, 17, 4, 5, tzinfo=KST)
    assert final.generation_name("v3", "M3'", now) == "v3-M3p-20260917T0405"
    assert final.generation_name("v2", "M3", now) != final.generation_name("v3", "M3'", now)


# --- 서비스 학습 기간은 warehouse 격자 끝을 따른다 ------------------------------------------


# 서비스 학습이 읽는 테이블(dim_stations, dim_hours, int_station_hour_grid, stg_stations)만 갖춘
# DuckDB 파일. 격자 끝(last_hour)을 바꿔 '새 반기가 들어온 warehouse'와 '달 중간에서 끝난
# warehouse'를 만든다. 반기마다 대여 수가 1, 2, 3…으로 달라 수준 특징 계산을 확인할 수 있다.
# stg_stations는 v3의 '행보다 앞선 대여소 정보 스냅샷'(asof_stations)용이다.
def serving_warehouse(path, last_hour: str):
    """대여소 하나가 반기마다 시간당 1, 2, 3, …대를 빌리는 2023-01 ~ last_hour 격자(파일)."""
    db = duckdb.connect(str(path))
    db.execute("""
        create table dim_stations as
        select 'ST-1' as station_id, 1 as station_no, '강남구' as district, 10 as docks,
               37.5 as lat, 127.0 as lon
    """)
    db.execute(f"""
        create table dim_hours as
        select h as hour_start, hour(h) as hour_of_day, isodow(h) as day_of_week,
               month(h) as month, dayofyear(h) as day_of_year,
               false as is_holiday, isodow(h) >= 6 as is_offday,
               10.0 as temp_c, 0.0 as rain_mm, 1.0 as wind_ms, 50.0 as humidity_pct,
               false as is_new_snow
        from unnest(generate_series(timestamp '2023-01-01', timestamp '{last_hour}',
                                    interval 1 hour)) t(h)
    """)
    db.execute("""
        create table int_station_hour_grid as
        select 'ST-1' as station_id, hour_start,
               ((year(hour_start) - 2023) * 2 + (month(hour_start) > 6)::int + 1) as rentals
        from dim_hours
    """)
    db.execute("""
        create table stg_stations as
        select * from (values (1, '강남구', 10, 37.5, 127.0, '2022-12'),
                              (1, '강남구', 10, 37.5, 127.0, '2026-12'))
            t(station_no, district, docks, lat, lon, snapshot)
    """)
    db.close()
    return path


# 모델 저장만 흉내 내는 가짜 LightGBM 모델.
class FakeBooster:
    def save_model(self, path):
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("fake")


# 실제 학습(수십 분~수 시간) 대신 조기 종료 시작 달과 최종 학습 행 수만 기록하는 가짜 학습 함수를
# 끼운다. 기간 계산이 맞는지만 빠르게 본다. 학습 함수가 불리지 않으면 기록이 빈 사전으로 남는다.
@pytest.fixture
def fake_training(monkeypatch):
    """학습 대신 최종 학습 행 수와 조기 종료 시작 달만 기록한다."""
    seen = {}

    def fit(load_stop_frame, load_full_frame, features, log, early_stop_from):
        seen["stop_from"] = early_stop_from
        seen["rows"] = len(load_full_frame()["rentals"])
        return FakeBooster(), 1

    monkeypatch.setattr(final, "fit_lightgbm_separate_stop", fit)
    return seen


# 출력 폴더에 'v3 validation에서 M3' 선택' 결과 파일을 만들어 서비스 학습이 v3 후보를 쓰게 한다.
def selected_v3(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_v3.json").write_text(json.dumps({"selected": "M3'"}), "utf-8")
    return out


# 격자가 2026-12-31 23시까지면 학습 기간은 [2023-01-01, 2027-01-01)이고, 산출물 meta와 보고서에 같은
# 기간이 적힌다. 조기 종료는 끝 2개월(2026-11부터), 최종 학습 행은 4년 전체 35,064시간.
# 또 게시된 산출물로 2027-02의 수준 특징을 만들면 결측이 없다(2026 하반기가 공개된 반기로 들어감).
# 고치기 전에는 기간이 2026-06에 고정돼 새 반기를 보지 못했다.
def test_serving_follows_warehouse_end(tmp_path, fake_training):
    """refresh_history로 2026 하반기를 넣은 뒤 다시 학습하면 그 반기까지 학습·산출물에 쓴다."""
    warehouse = serving_warehouse(tmp_path / "wh.duckdb", "2026-12-31 23:00:00")
    out = selected_v3(tmp_path / "models")
    report = final.fit_serving(warehouse, out)

    _, directory = final.current_generation(out / "serving")
    meta = json.loads((directory / "artifacts" / "meta.json").read_text("utf-8"))
    assert meta["history"] == ["2023-01-01", "2027-01-01"]
    assert list(report["history"]) == ["2023-01-01", "2027-01-01"]
    assert fake_training == {"stop_from": "2026-11-01", "rows": 35064}  # 2023~2026 전체 시간
    loaded = artifacts.load(directory / "artifacts")
    levels = predict.level_features("ST-1", datetime(2027, 2, 1, 8, tzinfo=KST), loaded)
    assert not any(math.isnan(value) for value in levels.values())


# 지금 warehouse(2026-06-30 23시까지)에서는 [2023-01-01, 2026-07-01)로, 이미 게시된 서비스 세대의
# 기간과 같다(기간 계산 방식을 바꿔도 지금 모델과 어긋나지 않음).
def test_serving_window_matches_current_data(tmp_path):
    """지금 warehouse(2026-06-30 23시까지)면 게시된 세대와 같은 기간이다."""
    warehouse = serving_warehouse(tmp_path / "wh.duckdb", "2026-06-30 23:00:00")
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        assert final.serving_window(con) == ("2023-01-01", "2026-07-01")
    finally:
        con.close()


# 격자가 달 중간(12-15)에서 끝나면 '달 중간' SystemExit. 기간을 먼저 정하므로 세대 폴더를 만들지
# 않고 학습도 하지 않는다.
def test_serving_refuses_partial_month(tmp_path, fake_training):
    warehouse = serving_warehouse(tmp_path / "wh.duckdb", "2026-12-15 23:00:00")
    out = selected_v3(tmp_path / "models")
    with pytest.raises(SystemExit, match="달 중간"):
        final.fit_serving(warehouse, out)
    assert not (out / "serving" / "generations").exists()
    assert fake_training == {}


# --history-end는 격자 끝 이전의 'YYYY-MM-01'만 받는다. 격자 끝보다 뒤(2027-02-01), 달 중간(07-15),
# 자릿수가 틀린 날짜(2026-7-1)는 거부하고 세대 폴더를 남기지 않는다. 2026-07-01로 줄이면 기간·조기
# 종료(2026-05부터)·행 수(30,648)가 그에 맞게 바뀐다.
def test_serving_history_end_can_only_shorten_to_a_month_start(tmp_path, fake_training):
    warehouse = serving_warehouse(tmp_path / "wh.duckdb", "2026-12-31 23:00:00")
    for bad in ("2027-02-01", "2026-07-15", "2026-7-1"):
        with pytest.raises(SystemExit, match="history-end"):
            final.fit_serving(warehouse, selected_v3(tmp_path / f"refused-{bad}"), bad)
        assert not (tmp_path / f"refused-{bad}" / "serving").exists()
    out = selected_v3(tmp_path / "models")
    report = final.fit_serving(warehouse, out, "2026-07-01")
    assert list(report["history"]) == ["2023-01-01", "2026-07-01"]
    assert fake_training == {"stop_from": "2026-05-01", "rows": 30648}


# 회귀 테스트(90d35aa): 끝 2개월이 조기 종료라 2023-03-01 이하로 자르면 학습 구간이 비어 LightGBM이
# 실패했다(세대 폴더는 이미 만든 뒤). 이제 하한 2023-04-01 미만은 미리 거부하고 학습 함수도 부르지
# 않는다. 2023-04-01이면 2023-01~03(2,160시간)을 쓰고 조기 종료는 2023-02부터다.
def test_serving_history_end_leaves_training_months_before_early_stopping(tmp_path, fake_training):
    """끝 2개월은 조기 종료에 쓰므로 그 앞에 학습할 달이 남는 값만 받는다(T59).

    2023-03-01 이하는 학습 구간이 비어 세대 폴더를 만든 뒤 LightGBM이 num_data > 0으로 실패했다.
    """
    warehouse = serving_warehouse(tmp_path / "wh.duckdb", "2023-12-31 23:00:00")
    for bad in ("2023-03-01", "2023-01-01", "2022-01-01", "0001-01-01"):
        with pytest.raises(SystemExit, match="history-end"):
            final.fit_serving(warehouse, selected_v3(tmp_path / f"refused-{bad}"), bad)
        assert not (tmp_path / f"refused-{bad}" / "serving").exists()
    assert fake_training == {}
    report = final.fit_serving(warehouse, selected_v3(tmp_path / "models"), "2023-04-01")
    assert list(report["history"]) == ["2023-01-01", "2023-04-01"]
    assert fake_training == {"stop_from": "2023-02-01", "rows": 2160}  # 2023-01~03 시간 수
