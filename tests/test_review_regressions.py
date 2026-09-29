"""Codex 검토(라운드 4)에서 합성 자료로 확인한 경계들을 회귀 테스트로 고정한다(T36).

원래 재현: work/round4-model-review/check_model_review.py, work/round4-pipeline-review/check.py.
실제 DB·API·모델 학습 없이 돈다.
"""

import json
import logging
import threading
import traceback
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import conftest
import httpx
import numpy as np
import pytest

from bike_demand.ingest import weather
from bike_demand.model import final, frames

FEATURES = frames.FEATURES + frames.LEVEL_FEATURES


def ordered(frame, names):
    order = np.lexsort((frame["hour_of_day"], frame["day_index"], frame["station_code"]))
    return frames.feature_matrix(frame, names)[order]


@pytest.fixture
def warehouse_with_snapshots(two_year_warehouse):
    two_year_warehouse.execute("""
        create table stg_stations as
        select * from (values (1, '강남구', 10, 37.5, 127.0, '2022-12'),
                              (1, '강남구', 10, 37.5, 127.0, '2026-06'))
            t(station_no, district, docks, lat, lon, snapshot)
    """)
    return two_year_warehouse


def test_t22_future_station_info_does_not_change_past_features(warehouse_with_snapshots):
    db = warehouse_with_snapshots
    window = frames.Window(("2024-06-01", "2024-08-01"), ("2023-01-01", "2024-06-01"))
    before = frames.load_frame(db, window, False, with_levels=True, asof_stations=True)
    db.execute("update dim_stations set docks = 99")
    db.execute("update stg_stations set docks = 99 where snapshot = '2026-06'")
    after = frames.load_frame(db, window, False, with_levels=True, asof_stations=True)
    np.testing.assert_array_equal(ordered(before, FEATURES), ordered(after, FEATURES))
    assert set(frames.load_frame(db, window, False)["docks"]) == {99.0}  # 옛 경로는 새어 들어감


def test_t23_early_stopping_targets_do_not_change_fit_and_stop_features(warehouse_with_snapshots):
    db = warehouse_with_snapshots
    train, stop = ("2023-01-01", "2025-01-01"), "2024-11-01"
    stop_window, full_window = frames.Window(train, (train[0], stop)), frames.Window(train, train)
    kwargs = {"with_levels": True, "asof_stations": True}
    stop_before = frames.load_frame(db, stop_window, False, **kwargs)
    full_before = frames.load_frame(db, full_window, False, **kwargs)
    db.execute(
        "update int_station_hour_grid set rentals = rentals * 10 "
        "where hour_start >= '2024-11-01' and hour_start < '2025-01-01'"
    )
    stop_after = frames.load_frame(db, stop_window, False, **kwargs)
    full_after = frames.load_frame(db, full_window, False, **kwargs)
    np.testing.assert_array_equal(ordered(stop_before, FEATURES), ordered(stop_after, FEATURES))
    assert not np.array_equal(
        ordered(full_before, ["profile_mean"]), ordered(full_after, ["profile_mean"])
    )


class FakeConnection:
    def close(self):
        pass


SAMPLE = {
    "rentals": np.array([1.0]),
    "station_code": np.array([0.0]),
    "day_index": np.array([0]),
    "hour_of_day": np.array([0.0]),
    "b0": np.array([1.0]),
}


def test_t27_failed_result_write_keeps_reservation_and_blocks_rerun(tmp_path):
    original_write = Path.write_text

    def fail_result(path, *args, **kwargs):
        if path.name == "test.json":
            raise OSError("result write failed")
        return original_write(path, *args, **kwargs)

    with (
        patch.object(final, "selected_model", return_value=("B0", "v1")),
        patch.object(final.duckdb, "connect", return_value=FakeConnection()) as connect,
        patch.object(final, "load_frame", return_value=SAMPLE) as load,
        patch.object(final, "evaluate", return_value={"mae": 0.0}),
        patch.object(Path, "write_text", new=fail_result),
    ):
        with pytest.raises(OSError):
            final.run_test(tmp_path / "never-opened.duckdb", tmp_path)
        assert (tmp_path / "test.started").exists()
        with pytest.raises(SystemExit):
            final.run_test(tmp_path / "never-opened.duckdb", tmp_path)
        assert connect.call_count == load.call_count == 1


def test_t27_concurrent_runs_read_test_data_once(tmp_path):
    barrier = threading.Barrier(8)
    outcomes = []

    def attempt():
        barrier.wait()
        try:
            final.run_test(tmp_path / "never-opened.duckdb", tmp_path)
            outcomes.append("ok")
        except SystemExit:
            outcomes.append("refused")

    with (
        patch.object(final, "selected_model", return_value=("B0", "v1")),
        patch.object(final.duckdb, "connect", return_value=FakeConnection()) as connect,
        patch.object(final, "load_frame", return_value=SAMPLE) as load,
        patch.object(final, "evaluate", return_value={"mae": 0.0}),
    ):
        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert sorted(outcomes) == ["ok"] + ["refused"] * 7
    assert connect.call_count == load.call_count == 1


def test_t28_reader_keeps_one_generation_even_if_pointer_switches_mid_read(tmp_path):
    serving = tmp_path / "serving"
    for name in ("old-generation", "new-generation"):
        (serving / "generations" / name).mkdir(parents=True)
    final.publish_generation(serving, "old-generation")
    loaded = {}

    def model_then_switch(model_file):
        loaded["model"] = Path(model_file)
        final.publish_generation(serving, "new-generation")
        return "fake-booster"

    def artifact_load(directory):
        loaded["artifacts"] = Path(directory)
        return "fake-artifacts"

    with (
        patch.object(final.lgb, "Booster", side_effect=model_then_switch),
        patch.object(final.artifacts, "load", side_effect=artifact_load),
    ):
        name, _, _ = final.load_serving_model(tmp_path)
    assert name == "old-generation"
    assert loaded["model"].parent == loaded["artifacts"].parent
    assert final.current_generation(serving)[0] == "new-generation"

    with patch.object(final.os, "replace", side_effect=OSError("publish failed")):
        with pytest.raises(OSError):
            final.publish_generation(serving, "failed-generation")
    assert final.current_generation(serving)[0] == "new-generation"


@pytest.mark.parametrize("place", ["item", "error", "nested"])
def test_t25_escaped_key_is_rejected_everywhere_without_leaking(tmp_path, place, monkeypatch):
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)
    secret = "fake-round4-check-key"
    escaped = "".join(chr(92) + f"u{ord(c):04x}" for c in secret)
    body = {
        "response": {
            "header": {"resultCode": "00"},
            "body": {"totalCount": 1, "items": {"item": [{"tm": "2023-01-01 01:00"}]}},
        }
    }
    if place == "item":
        body["response"]["body"]["items"]["item"][0]["ta"] = secret
    elif place == "error":
        body["response"]["header"] = {"resultCode": "99", "resultMsg": secret}
    else:
        body["extra"] = {"a": [{"b": secret}]}
    text = json.dumps(body).replace(secret, escaped)
    assert secret not in text

    records: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = Collect()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=text)))
    raw = tmp_path / "raw"
    try:
        with pytest.raises(weather.ApiError):
            try:
                list(weather.download(raw, ["2023-01"], secret, date(2023, 2, 10), client))
            except weather.ApiError:
                assert secret not in traceback.format_exc()
                raise
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        client.close()
    assert all(secret not in line for line in records)
    assert not weather.raw_path(raw, "2023-01").exists()


@pytest.mark.parametrize("failure", ["upgrade", "engine"])
def test_t30_temporary_database_dropped_when_upgrade_or_engine_fails(failure):
    statements: list[str] = []
    disposed: list[bool] = []

    class Admin:
        @contextmanager
        def connect(self):
            yield self

        def execute(self, statement):
            statements.append(str(statement))

        def dispose(self):
            disposed.append(True)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    upgrade = fail if failure == "upgrade" else (lambda *args: None)
    with (
        patch.object(conftest, "create_engine", return_value=Admin()),
        patch.object(conftest, "make_engine", side_effect=fail),
    ):
        with pytest.raises(RuntimeError, match="synthetic failure"):
            with conftest.temporary_database(upgrade=upgrade):
                raise AssertionError("unreachable")
    assert statements[0].startswith('create database "bike_demand_test_')
    assert statements[1].startswith('drop database if exists "bike_demand_test_')
    assert statements[0].split('"')[1] == statements[1].split('"')[1]
    assert disposed == [True]


def _unreachable_admin():
    """접속하면 서버에 닿지 못했다는 오류를 내는 관리 엔진."""
    from sqlalchemy.exc import OperationalError

    admin = Mock()
    admin.connect.side_effect = OperationalError("connect", {}, Exception("timeout expired"))
    return admin


def test_temporary_database_admin_engine_gives_up_connecting_quickly():
    """DB가 꺼져 있으면 Windows의 psycopg는 거부된 연결을 알아채지 못해 멈춘다. 관리 엔진도 서비스
    엔진처럼 접속 제한 시간이 있어야 DB 테스트가 멈추지 않고 건너뛴다(2026-09-29 검토)."""
    admin = _unreachable_admin()
    with patch.object(conftest, "create_engine", return_value=admin) as create:
        with pytest.raises(conftest.DatabaseUnavailable, match="OperationalError"):
            with conftest.temporary_database():
                raise AssertionError("unreachable")
    connect_args = create.call_args.kwargs.get("connect_args", {})
    assert 0 < connect_args.get("connect_timeout", 0) <= 10
    admin.dispose.assert_called_once()


def test_pg_engine_remembers_an_unreachable_server(monkeypatch, request):
    monkeypatch.setattr(conftest, "_db_unavailable", None, raising=False)
    with patch.object(conftest, "create_engine", return_value=_unreachable_admin()):
        with pytest.raises(pytest.skip.Exception, match="OperationalError"):
            request.getfixturevalue("pg_engine")
    assert conftest._db_unavailable == "OperationalError"


def test_pg_engine_skips_without_connecting_after_an_unreachable_server(monkeypatch, request):
    """한 번 접속에 실패하면 나머지 DB 테스트는 제한 시간을 다시 기다리지 않고 바로 건너뛴다."""
    monkeypatch.setattr(conftest, "_db_unavailable", "OperationalError", raising=False)
    with patch.object(conftest, "create_engine", return_value=_unreachable_admin()) as create:
        with pytest.raises(pytest.skip.Exception, match="OperationalError"):
            request.getfixturevalue("pg_engine")
    create.assert_not_called()
