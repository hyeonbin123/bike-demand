import json
import math
import threading
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from bike_demand.model import artifacts, final, predict

KST = timezone(timedelta(hours=9))


def test_reserve_once_allows_only_one_start(tmp_path):
    path = tmp_path / "test_v2.started"
    final.reserve_once(path)
    with pytest.raises(SystemExit, match="이미 시작된"):
        final.reserve_once(path)


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


def test_run_test_refuses_before_touching_data_when_reserved(tmp_path, monkeypatch):
    (tmp_path / "validation.json").write_text(json.dumps({"selected": "M1"}), "utf-8")
    (tmp_path / "test.started").write_text("started", "utf-8")
    monkeypatch.setattr(final, "load_frame", lambda *a, **k: pytest.fail("test 데이터를 읽음"))
    with pytest.raises(SystemExit, match="이미 시작된"):
        final.run_test(tmp_path / "wh.duckdb", tmp_path)


def test_v3_never_measures_test(tmp_path):
    (tmp_path / "validation_v3.json").write_text(json.dumps({"selected": "M3'"}), "utf-8")
    assert final.selected_model(tmp_path) == ("M3'", "v3")
    with pytest.raises(SystemExit, match="v3"):
        final.run_test(tmp_path / "wh.duckdb", tmp_path)


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


def test_legacy_layout_is_still_readable(tmp_path):
    root = tmp_path / "serving"
    root.mkdir()
    (root / "model.txt").write_text("", "utf-8")
    (root / "serving.json").write_text(json.dumps({"selected": "M1"}), "utf-8")
    assert final.current_generation(root) == ("v1", root)


def test_generation_name_distinguishes_models_for_the_same_forecast():
    now = datetime(2026, 9, 17, 4, 5, tzinfo=KST)
    assert final.generation_name("v3", "M3'", now) == "v3-M3p-20260917T0405"
    assert final.generation_name("v2", "M3", now) != final.generation_name("v3", "M3'", now)


# --- 서비스 학습 기간은 warehouse 격자 끝을 따른다 ------------------------------------------


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


class FakeBooster:
    def save_model(self, path):
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("fake")


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


def selected_v3(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_v3.json").write_text(json.dumps({"selected": "M3'"}), "utf-8")
    return out


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


def test_serving_window_matches_current_data(tmp_path):
    """지금 warehouse(2026-06-30 23시까지)면 게시된 세대와 같은 기간이다."""
    warehouse = serving_warehouse(tmp_path / "wh.duckdb", "2026-06-30 23:00:00")
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        assert final.serving_window(con) == ("2023-01-01", "2026-07-01")
    finally:
        con.close()


def test_serving_refuses_partial_month(tmp_path, fake_training):
    warehouse = serving_warehouse(tmp_path / "wh.duckdb", "2026-12-15 23:00:00")
    out = selected_v3(tmp_path / "models")
    with pytest.raises(SystemExit, match="달 중간"):
        final.fit_serving(warehouse, out)
    assert not (out / "serving" / "generations").exists()
    assert fake_training == {}


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
