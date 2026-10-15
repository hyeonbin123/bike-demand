"""Codex 검토(라운드 4)에서 합성 자료로 확인한 경계들을 회귀 테스트로 고정한다(T36).

원래 재현: work/round4-model-review/check_model_review.py, work/round4-pipeline-review/check.py.
실제 DB·API·모델 학습 없이 돈다.
"""

# [무엇을 검증하나] 코드 검토에서 찾아 고친 문제들이 다시 생기지 않게 잡아 두는 회귀 테스트
# 묶음이다.
# - 측정 누수: 미래 대여소 정보가 과거 행 특징을 바꾸지 않음, 조기 종료 구간의 대여가 조기 종료용
#   특징에 섞이지 않음(docs/experiments.md v3)
# - test 한 번만: 결과 쓰기가 실패해도 예약이 남아 재실행을 막고, 동시 실행 8개 중 하나만 test
#   자료를 읽음
# - 서비스 세대: 읽는 도중 CURRENT가 바뀌어도 모델과 산출물을 같은 세대에서 읽음, 게시 실패 시 이전
#   포인터 유지
# - 인증키: 유니코드 이스케이프로 숨긴 키도 거부하고 어디에도 남기지 않음
# - 테스트 인프라: 임시 DB 정리, DB가 꺼졌을 때 빨리 건너뛰기, DB가 필수인 CI에서는 건너뛰지 않고
#   실패
# [방식] 실제 DB·API·학습 없이 conftest의 메모리 warehouse, unittest.mock의 patch, 가짜 객체로 돈다.

import json
import logging
import threading
import traceback
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

# conftest를 모듈로 직접 가져와 temporary_database·pg_engine의 내부(create_engine, make_engine,
# _db_unavailable)를 바꿔 끼운다.
import conftest
import httpx
import numpy as np
import pytest

from bike_demand.ingest import weather
from bike_demand.model import final, frames

# M3 계열 전체 특징(v1 특징 + v2 수준 특징).
FEATURES = frames.FEATURES + frames.LEVEL_FEATURES


# load_frame 결과의 행 순서는 보장되지 않으므로 (대여소 코드, 날짜, 시간) 순으로 정렬한 특징 행렬로
# 바꿔 비교한다.
def ordered(frame, names):
    order = np.lexsort((frame["hour_of_day"], frame["day_index"], frame["station_code"]))
    return frames.feature_matrix(frame, names)[order]


# two_year_warehouse에 대여소 정보 스냅샷 두 개(22.12월, 26.6월)를 더한 warehouse. 26.6월 스냅샷은
# 2024년 행에서 보면 '미래'다.
@pytest.fixture
def warehouse_with_snapshots(two_year_warehouse):
    two_year_warehouse.execute("""
        create table stg_stations as
        select * from (values (1, '강남구', 10, 37.5, 127.0, '2022-12'),
                              (1, '강남구', 10, 37.5, 127.0, '2026-06'))
            t(station_no, district, docks, lat, lon, snapshot)
    """)
    return two_year_warehouse


# 2024-06~07 행을 만든 뒤 미래 정보(dim_stations와 26.6월 스냅샷의 거치대 수)를 99로 바꿔도,
# asof_stations 경로의 특징은 한 값도 바뀌지 않아야 한다. 마지막 줄은 대조군: 옛 경로(asof_stations
# 없음)는 dim_stations를 그대로 붙여 99가 새어 들어온다.
def test_t22_future_station_info_does_not_change_past_features(warehouse_with_snapshots):
    db = warehouse_with_snapshots
    window = frames.Window(("2024-06-01", "2024-08-01"), ("2023-01-01", "2024-06-01"))
    before = frames.load_frame(db, window, False, with_levels=True, asof_stations=True)
    db.execute("update dim_stations set docks = 99")
    db.execute("update stg_stations set docks = 99 where snapshot = '2026-06'")
    after = frames.load_frame(db, window, False, with_levels=True, asof_stations=True)
    np.testing.assert_array_equal(ordered(before, FEATURES), ordered(after, FEATURES))
    assert set(frames.load_frame(db, window, False)["docks"]) == {99.0}  # 옛 경로는 새어 들어감


# 조기 종료 구간(2024-11~12)의 대여를 10배로 바꿔도, 조기 종료 단계용 행(패턴·추세를 그 구간
# 전까지만으로 계산)의 특징은 그대로여야 한다. 대조군으로 전체 기간 패턴을 쓰는 최종 학습 행의
# profile_mean은 바뀐다(그 구간도 최종 학습에는 정상적으로 들어감).
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


# duckdb 연결 대신 쓰는 가짜(닫기만 함).
class FakeConnection:
    def close(self):
        pass


# load_frame 대신 돌려줄 한 행짜리 test 자료(기준선 B0 평가에 필요한 열만).
SAMPLE = {
    "rentals": np.array([1.0]),
    "station_code": np.array([0.0]),
    "day_index": np.array([0]),
    "hour_of_day": np.array([0.0]),
    "b0": np.array([1.0]),
}


# 선택이 B0인 test에서 결과 test.json 쓰기만 실패하게 만든다. 예약 파일(test.started)은 남아 있어야
# 하고, 다시 실행하면 SystemExit로 거부되며, test 자료는 처음 한 번만 읽혔다(connect·load_frame 호출
# 1번). 실패 뒤 다시 돌려 test 기간을 두 번 보는 일을 막는다.
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


# run_test 8개를 동시에 시작해도 하나만 통과하고 나머지는 거부되며, test 자료를 읽은 것도 한
# 번뿐이다. 예약을 test 자료를 읽기 전에 원자적으로 잡기 때문이다.
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


# load_serving_model이 모델 파일을 읽는 순간 CURRENT를 새 세대로 바꿔도, 모델과 산출물은 둘 다 처음
# 읽은 옛 세대 폴더에서 온다(CURRENT를 한 번만 읽음). 서로 다른 세대의 모델과 산출물을 섞어 쓰면
# 대여소 코드·패턴이 어긋난다.
# 이어서 게시 중 os.replace가 실패하면 CURRENT는 이전 세대 그대로다.
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


# ASOS 응답의 항목 값, 오류 메시지, 알 수 없는 중첩 필드에 키를 \uXXXX로 숨겨 넣는다(원문에는 키
# 문자열이 없음). 세 경우 모두 ApiError로 거부하고, traceback·모든 로그 기록(루트 로거 DEBUG)에 키가
# 없으며, 그달 원본을 저장하지 않는다.
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


# 임시 DB를 만든 직후 마이그레이션이나 엔진 생성이 실패해도, 같은 이름의 DB를 DROP하고 관리 엔진을
# 정리한다. 가짜 관리 엔진이 실행한 SQL 문을 기록해 CREATE와 DROP의 DB 이름이 같은지 본다. 이전에는
# 실패한 테스트가 서버에 임시 DB를 남길 수 있었다.
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


# connect()를 부르면 OperationalError를 내는 Mock 관리 엔진.
def _unreachable_admin():
    """접속하면 서버에 닿지 못했다는 오류를 내는 관리 엔진."""
    from sqlalchemy.exc import OperationalError

    admin = Mock()
    admin.connect.side_effect = OperationalError("connect", {}, Exception("timeout expired"))
    return admin


# 관리 엔진을 만들 때 connect_timeout(0초 초과 10초 이하)을 넘기는지, 접속 실패를
# DatabaseUnavailable로 바꾸고 엔진을 정리하는지 확인한다(e1d0ffb: 고치기 전에는 DB를 끈 상태에서
# 테스트가 멈춰 강제 종료해야 했다).
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


# pg_engine fixture를 직접 불러, 접속 실패 때 테스트를 건너뛰고(skip) 실패 이유를 모듈 변수에
# 기억하는지 본다. monkeypatch로 기억 상태를 테스트 뒤 원래대로 되돌린다. CI는
# BIKE_DEMAND_REQUIRE_DB=1로 돌므로, 기본 동작(건너뛰기)을 보는 이 테스트는 그 값을 지운 상태에서
# 잰다.
def test_pg_engine_remembers_an_unreachable_server(monkeypatch, request):
    monkeypatch.delenv(conftest.REQUIRE_DB_ENV, raising=False)
    monkeypatch.setattr(conftest, "_db_unavailable", None, raising=False)
    with patch.object(conftest, "create_engine", return_value=_unreachable_admin()):
        with pytest.raises(pytest.skip.Exception, match="OperationalError"):
            request.getfixturevalue("pg_engine")
    assert conftest._db_unavailable == "OperationalError"


# 이미 실패를 기억한 상태면 관리 엔진을 만들지도 않고(create_engine 호출 없음) 바로 건너뛴다. 나머지
# DB 테스트가 각각 5초씩 다시 기다리지 않게 한다.
def test_pg_engine_skips_without_connecting_after_an_unreachable_server(monkeypatch, request):
    """한 번 접속에 실패하면 나머지 DB 테스트는 제한 시간을 다시 기다리지 않고 바로 건너뛴다."""
    monkeypatch.delenv(conftest.REQUIRE_DB_ENV, raising=False)
    monkeypatch.setattr(conftest, "_db_unavailable", "OperationalError", raising=False)
    with patch.object(conftest, "create_engine", return_value=_unreachable_admin()) as create:
        with pytest.raises(pytest.skip.Exception, match="OperationalError"):
            request.getfixturevalue("pg_engine")
    create.assert_not_called()


# BIKE_DEMAND_REQUIRE_DB=1(CI 설정)이면 접속 실패가 건너뛰기가 아니라 실패여야 한다. 첫 접속 실패와
# 이미 기억한 실패 두 경로를 모두 본다. 이게 없으면 CI의 Postgres 서비스 설정이 틀려도 DB 테스트가
# 모두 건너뛰어진 채 초록으로 끝난다. "true"는 켜짐으로 보지 않는다(정확히 "1"만).
@pytest.mark.parametrize("remembered", [None, "OperationalError"])
def test_pg_engine_fails_instead_of_skipping_when_the_database_is_required(
    monkeypatch, request, remembered
):
    monkeypatch.setenv(conftest.REQUIRE_DB_ENV, "1")
    monkeypatch.setattr(conftest, "_db_unavailable", remembered, raising=False)
    with patch.object(conftest, "create_engine", return_value=_unreachable_admin()):
        with pytest.raises(pytest.fail.Exception, match="OperationalError"):
            request.getfixturevalue("pg_engine")
    assert conftest._db_unavailable == "OperationalError"


# 켜짐 값은 정확히 "1"뿐이다. 다른 값이면 기본 동작(건너뛰기)을 유지한다.
def test_pg_engine_require_flag_accepts_only_one(monkeypatch, request):
    monkeypatch.setenv(conftest.REQUIRE_DB_ENV, "true")
    monkeypatch.setattr(conftest, "_db_unavailable", "OperationalError", raising=False)
    with pytest.raises(pytest.skip.Exception, match="OperationalError"):
        request.getfixturevalue("pg_engine")
