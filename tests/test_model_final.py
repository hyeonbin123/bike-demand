import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from bike_demand.model import final

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
