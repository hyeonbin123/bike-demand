from datetime import date, datetime

import httpx
from sqlalchemy import func, select

from bike_demand.ingest import realtime
from bike_demand.pipelines import collect_realtime
from bike_demand.serving.models import RealtimeSnapshot

SECRET = "fake-key"


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
    """23:59:59 수집 성공 → 적재 실패 → 00:02 재시도에서 전날 스냅샷도 적재된다(T35)."""
    from bike_demand.serving import load

    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    client = httpx.Client(transport=httpx.MockTransport(bike_list))
    real_load = load.load_realtime
    calls = []
    failures = [RuntimeError("DB 잠깐 끊김")]

    def flaky_load(engine, bronze_dir, day):
        calls.append(day)
        if failures:
            raise failures.pop()
        return real_load(engine, bronze_dir, day)

    monkeypatch.setattr(load, "load_realtime", flaky_load)
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

    # 한 시간 뒤(1시대)에는 전날을 다시 건드리지 않는다
    calls.clear()
    one_am = datetime(2026, 9, 17, 1, 0, tzinfo=realtime.KST)
    collect_realtime(pg_engine, raw, bronze, SECRET, one_am, client)
    assert calls == [date(2026, 9, 17)]
