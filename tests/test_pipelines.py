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
