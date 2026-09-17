from datetime import date, datetime

import httpx
from sqlalchemy import func, select

from bike_demand.ingest import realtime
from bike_demand.pipelines import collect_realtime
from bike_demand.serving.models import RealtimeSnapshot

SECRET = "fake-key"


def day_of(parquet_path):
    """bronze/date=YYYY-MM-DD/파일 → 그 날짜."""
    return date.fromisoformat(parquet_path.parent.name.removeprefix("date="))


def bike_list(request: httpx.Request) -> httpx.Response:
    rows = [
        {"rackTotCnt": "10", "stationName": f"{n}. 대여소", "parkingBikeTotCnt": "3",
         "shared": "30", "stationLatitude": "37.5", "stationLongitude": "127.0",
         "stationId": f"ST-{n}"}
        for n in (1, 2)
    ]  # fmt: skip
    body = {
        "rentBikeStatus": {
            "list_total_count": 2,
            "RESULT": {"CODE": "INFO-000", "MESSAGE": "정상"},
            "row": rows,
        }
    }
    return httpx.Response(200, json=body)


def test_snapshot_taken_just_before_midnight_is_loaded_under_its_own_day(pg_engine, tmp_path):
    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    just_before_midnight = datetime(2026, 9, 16, 23, 59, 59, 900000, tzinfo=realtime.KST)
    client = httpx.Client(transport=httpx.MockTransport(bike_list))

    result = collect_realtime(pg_engine, raw, bronze, SECRET, just_before_midnight, client)

    assert result["day"] == "2026-09-16"
    assert result["snapshots_inserted"] == 2 and result["new_stations"] == 2
    assert (bronze / "date=2026-09-16" / "snapshots.parquet").exists()
    with pg_engine.connect() as conn:
        fetched = conn.execute(select(func.max(RealtimeSnapshot.fetched_at))).scalar_one()
    assert fetched.astimezone(realtime.KST).date() == date(2026, 9, 16)


def test_latest_trip_month_and_last_day(tmp_path):
    from bike_demand.pipelines import last_day, latest_trip_month

    raw = tmp_path / "trips"
    raw.mkdir()
    for name in ("대여이력_2606.csv", "대여이력_2612.csv", "대여이력_2609.csv"):
        (raw / name).write_bytes(b"")
    assert latest_trip_month(raw) == "2026-12"
    assert last_day("2026-12") == "2026-12-31"
    assert last_day("2028-02") == "2028-02-29"


def test_retry_after_midnight_recovers_the_previous_day(pg_engine, tmp_path, monkeypatch):
    """23:59:59 수집 성공 → 적재 실패 → 00:02 다음 실행에서 전날 스냅샷도 적재된다(T35)."""
    from bike_demand.serving import load

    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    client = httpx.Client(transport=httpx.MockTransport(bike_list))
    real_load = load.load_realtime_file
    calls = []
    failures = [RuntimeError("DB 잠깐 끊김")]

    def flaky_load(engine, path):
        calls.append(day_of(path))
        if failures:
            raise failures.pop()
        return real_load(engine, path)

    monkeypatch.setattr(load, "load_realtime_file", flaky_load)
    before = datetime(2026, 9, 16, 23, 59, 59, tzinfo=realtime.KST)
    after = datetime(2026, 9, 17, 0, 2, tzinfo=realtime.KST)
    try:
        collect_realtime(pg_engine, raw, bronze, SECRET, before, client)
    except RuntimeError:
        pass
    result = collect_realtime(pg_engine, raw, bronze, SECRET, after, client)

    assert calls == [date(2026, 9, 16), date(2026, 9, 16), date(2026, 9, 17)]
    assert result["days"]["2026-09-16"]["snapshots_inserted"] == 2
    assert result["days"]["2026-09-17"]["snapshots_inserted"] == 2
    with pg_engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(RealtimeSnapshot)).scalar_one()
    assert count == 4

    # 이미 다 적재한 전날은 다시 건드리지 않는다
    calls.clear()
    one_am = datetime(2026, 9, 17, 1, 0, tzinfo=realtime.KST)
    collect_realtime(pg_engine, raw, bronze, SECRET, one_am, client)
    assert calls == [date(2026, 9, 17)]


def test_failures_through_the_midnight_hour_are_recovered_later(pg_engine, tmp_path, monkeypatch):
    """23:59와 0시대 재시도가 모두 적재에 실패해도 1시대 이후 정상 실행이 전날을 회수한다(T39)."""
    from bike_demand.serving import load

    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    client = httpx.Client(transport=httpx.MockTransport(bike_list))
    real_load = load.load_realtime_file
    calls = []
    db_down = {"value": True}

    def flaky_load(engine, path):
        calls.append(day_of(path))
        if db_down["value"]:
            raise RuntimeError("DB 장애")
        return real_load(engine, path)

    monkeypatch.setattr(load, "load_realtime_file", flaky_load)
    for at in (datetime(2026, 9, 16, 23, 59, 59), datetime(2026, 9, 17, 0, 2)):
        try:
            collect_realtime(
                pg_engine, raw, bronze, SECRET, at.replace(tzinfo=realtime.KST), client
            )
        except RuntimeError:
            pass
    assert not (bronze / "date=2026-09-16" / "loaded.txt").exists()

    db_down["value"] = False
    calls.clear()
    late = datetime(2026, 9, 17, 3, 10, tzinfo=realtime.KST)
    result = collect_realtime(pg_engine, raw, bronze, SECRET, late, client)
    assert calls == [date(2026, 9, 16), date(2026, 9, 17)]
    assert result["days"]["2026-09-16"]["snapshots_inserted"] == 2
    assert result["days"]["2026-09-17"]["snapshots_inserted"] == 4  # 00:02와 03:10
    loaded = (bronze / "date=2026-09-16" / "loaded.txt").read_text(encoding="utf-8").split()
    assert loaded == ["235959.json"]

    # 다음 실행은 오늘만 적재하고, 같은 원본을 다시 넣어도 행이 늘지 않는다
    calls.clear()
    again = datetime(2026, 9, 17, 3, 20, tzinfo=realtime.KST)
    collect_realtime(pg_engine, raw, bronze, SECRET, again, client)
    assert calls == [date(2026, 9, 17)]
    with pg_engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(RealtimeSnapshot)).scalar_one()
    assert count == 8


def test_overlapping_runs_do_not_rebuild_and_load_at_the_same_time(tmp_path, monkeypatch):
    """적재 구간은 한 실행만: 잠금이 있으면 원본만 남기고 LoadBusy, 오래된 잠금은 지운다(T42)."""
    import os
    import time

    import pytest

    from bike_demand import pipelines

    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    client = httpx.Client(transport=httpx.MockTransport(bike_list))
    at = datetime(2026, 9, 17, 12, 0, tzinfo=realtime.KST)
    lock = bronze / pipelines.LOCK_NAME
    bronze.mkdir(parents=True)
    lock.touch()
    monkeypatch.setattr(
        pipelines.load, "load_realtime_file", lambda *a: pytest.fail("적재하면 안 됨")
    )
    with pytest.raises(pipelines.LoadBusy):
        pipelines.collect_realtime(None, raw, bronze, SECRET, at, client)
    assert len(list((raw / "date=2026-09-17").glob("*.json"))) == 1  # 수집분은 남음
    assert lock.exists() and not (bronze / "date=2026-09-17" / "loaded.txt").exists()

    old = time.time() - pipelines.STALE_LOCK_SECONDS - 60
    os.utime(lock, (old, old))
    monkeypatch.setattr(pipelines.load, "load_realtime_file", lambda *a: (2, 0))
    later = datetime(2026, 9, 17, 12, 10, tzinfo=realtime.KST)
    result = pipelines.collect_realtime(None, raw, bronze, SECRET, later, client)
    assert result["days"]["2026-09-17"]["parquet_rows"] == 4
    assert not lock.exists()
    loaded = (bronze / "date=2026-09-17" / "loaded.txt").read_text(encoding="utf-8").split()
    assert loaded == ["120000.json", "121000.json"]


def test_taken_over_lock_cannot_mark_snapshots_that_were_not_loaded(tmp_path, monkeypatch):
    """느린 실행 A의 잠금을 30분 뒤 B가 가져가도, 각자 만든 파일을 적재하므로 기록이 적재를 앞서지
    않는다. A는 끝날 때 B의 잠금을 지우지 않는다(T44, codex 라운드 6 재현)."""
    import os
    import time

    import pyarrow.parquet as pq

    from bike_demand import pipelines

    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    client = httpx.Client(transport=httpx.MockTransport(bike_list))
    lock = bronze / pipelines.LOCK_NAME
    day_dir = bronze / "date=2026-09-17"
    a_run = datetime(2026, 9, 17, 23, 40, tzinfo=realtime.KST)
    b_run = datetime(2026, 9, 17, 23, 50, tzinfo=realtime.KST)
    a_old: list[bytes] = []
    loaded_by_run: list[set[str]] = []

    def names_in(path):
        rows = pq.read_table(path).to_pylist()
        return {
            datetime.fromisoformat(r["fetched_at"]).astimezone(realtime.KST).strftime("%H%M%S")
            + ".json"
            for r in rows
        }

    def load_file(engine, path):
        if not a_old:
            # A 적재 도중: A가 만든 옛 내용(23:40분)을 기억하고, 잠금을 30분 넘게 만든 뒤 B 실행
            a_old.append(path.read_bytes())
            old = time.time() - pipelines.STALE_LOCK_SECONDS - 60
            os.utime(lock, (old, old))
            pipelines.collect_realtime(None, raw, bronze, SECRET, b_run, client)
            b_marked = set((day_dir / "loaded.txt").read_text(encoding="utf-8").split())
            assert b_marked == {"234000.json", "235000.json"}
            assert b_marked <= loaded_by_run[-1]  # B가 기록한 원본은 B가 실제로 적재했다
        elif not loaded_by_run:
            # B 적재 직전: 늦게 끝난 A의 재생성이 공용 Parquet을 옛 내용으로 덮는다
            (day_dir / "snapshots.parquet").write_bytes(a_old[0])
        loaded_by_run.append(names_in(path))
        return 0, 0

    monkeypatch.setattr(pipelines.load, "load_realtime_file", load_file)
    pipelines.collect_realtime(None, raw, bronze, SECRET, a_run, client)

    a_marked = set((day_dir / "loaded.txt").read_text(encoding="utf-8").split())
    assert a_marked <= loaded_by_run[-1]  # A의 기록(옛 목록)도 A가 적재한 것뿐 → 다음 실행이 회수
    assert not lock.exists()  # B가 가져간 잠금을 A가 지우지 않았고, B는 자기 잠금을 풀었다
    assert not list(day_dir.glob("*.part"))


def test_lock_release_keeps_a_lock_taken_over_by_another_run(tmp_path):
    from bike_demand import pipelines

    lock = tmp_path / pipelines.LOCK_NAME
    with pipelines._load_lock(tmp_path):
        lock.write_text("someone-else", encoding="utf-8")
    assert lock.read_text(encoding="utf-8") == "someone-else"
    lock.unlink()
    with pipelines._load_lock(tmp_path):
        assert lock.exists()
    assert not lock.exists()
