from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from bike_demand.api import main
from bike_demand.serving.models import Prediction, RealtimeSnapshot, Station

KST = timezone(timedelta(hours=9))
NOW = datetime(2026, 9, 17, 8, 25, tzinfo=KST)


@pytest.fixture
def client(pg_engine):
    main.app.dependency_overrides[main.get_engine] = lambda: pg_engine
    main.app.dependency_overrides[main.get_now] = lambda: NOW
    try:
        yield TestClient(main.app)
    finally:
        main.app.dependency_overrides.clear()


def insert(engine, table, rows):
    with engine.begin() as conn:
        conn.execute(table.__table__.insert(), rows)


def add_stations(engine):
    insert(
        engine,
        Station,
        [
            {"station_id": "ST-1", "station_no": 1, "station_name": "강남역", "district": "강남구",
             "lat": 37.49812, "lon": 127.02761, "docks": 20, "source": "history"},
            {"station_id": "ST-2", "station_no": 2, "station_name": "역삼역", "district": "강남구",
             "lat": 37.5, "lon": 127.03, "docks": 10, "source": "history"},
            {"station_id": "ST-3", "station_no": 3, "station_name": "합정역", "district": "마포구",
             "lat": None, "lon": None, "docks": 15, "source": "history"},
        ],
    )  # fmt: skip


def add_predictions(engine, version, value, created_at, stations=("ST-1", "ST-2", "ST-3")):
    first = NOW.replace(minute=0)
    rows = [
        {"station_id": s, "hour_start": first + timedelta(hours=h), "model_version": version,
         "predicted_rentals": value, "created_at": created_at}
        for s in stations for h in range(-8, 16)
    ]  # fmt: skip
    insert(engine, Prediction, rows)


def test_health_without_data_and_with_data(client, pg_engine):
    body = client.get("/health").json()
    assert body == {"status": "ok", "database": "ok", "latest_snapshot_at": None,
                    "latest_prediction_hour": None}  # fmt: skip
    add_stations(pg_engine)
    insert(
        pg_engine, RealtimeSnapshot, [{"station_id": "ST-1", "fetched_at": NOW, "bike_count": 1}]
    )
    add_predictions(pg_engine, "v2-fcst202609170500", 1.0, NOW)
    body = client.get("/health").json()
    assert body["latest_snapshot_at"] == "2026-09-17T08:25:00+09:00"
    assert body["latest_prediction_hour"] == "2026-09-17T23:00:00+09:00"


def test_stations_filter_and_null_coordinates(client, pg_engine):
    add_stations(pg_engine)
    assert [s["station_id"] for s in client.get("/stations").json()] == ["ST-1", "ST-2", "ST-3"]
    gangnam = client.get("/stations", params={"district": "강남구"}).json()
    assert [s["station_id"] for s in gangnam] == ["ST-1", "ST-2"]
    assert gangnam[0]["lat"] == 37.49812  # 좌표는 반올림하지 않음
    assert client.get("/stations").json()[2]["lat"] is None


def test_station_detail_uses_newest_version_only(client, pg_engine):
    add_stations(pg_engine)
    insert(pg_engine, RealtimeSnapshot, [
        {"station_id": "ST-1", "fetched_at": NOW - timedelta(minutes=m), "bike_count": b,
         "rack_count": 20}
        for m, b in ((20, 5), (10, 4))
    ])  # fmt: skip
    add_predictions(pg_engine, "v1-old", 9.999, NOW - timedelta(hours=3))
    # 새 버전에는 ST-1이 없음 → 옛 버전으로 채우지 않고 빈 목록
    add_predictions(pg_engine, "v2-new", 1.234, NOW - timedelta(hours=1), stations=("ST-2",))

    body = client.get("/stations/ST-1").json()
    assert body["snapshot"] == {"fetched_at": "2026-09-17T08:15:00+09:00", "bike_count": 4,
                                "rack_count": 20}  # fmt: skip
    assert body["model_version"] == "v2-new" and body["predictions"] == []

    detail = client.get("/stations/ST-2").json()
    assert [p["hour_start"][11:16] for p in detail["predictions"]] == [
        "08:00", "09:00", "10:00", "11:00", "12:00", "13:00",
    ]  # fmt: skip
    assert detail["predictions"][0]["predicted_rentals"] == 1.23
    assert client.get("/stations/ST-404").status_code == 404


def test_version_tie_breaks_on_name(client, pg_engine):
    add_stations(pg_engine)
    same_time = NOW - timedelta(hours=1)
    add_predictions(pg_engine, "v1-a", 1.0, same_time, stations=("ST-1",))
    add_predictions(pg_engine, "v1-b", 2.0, same_time, stations=("ST-1",))
    assert client.get("/stations/ST-1").json()["model_version"] == "v1-b"


def test_shortage_risk_prorates_filters_and_sorts(client, pg_engine):
    add_stations(pg_engine)
    insert(pg_engine, RealtimeSnapshot, [
        # ST-1: 08:20 기준, 자전거 1대. 3시간 기대 대여 = 2 × 3 = 6 → 부족 5
        {"station_id": "ST-1", "fetched_at": NOW - timedelta(minutes=5), "bike_count": 1},
        # ST-2: 자전거 10대 → 부족 없음
        {"station_id": "ST-2", "fetched_at": NOW - timedelta(minutes=5), "bike_count": 10},
        # ST-3: 스냅샷이 40분 전 → 기본 30분 기준에서 빠짐
        {"station_id": "ST-3", "fetched_at": NOW - timedelta(minutes=40), "bike_count": 0},
    ])  # fmt: skip
    add_predictions(pg_engine, "v2-fcst202609170500", 2.0, NOW - timedelta(hours=1))

    body = client.get("/shortage-risk").json()
    assert body["model_version"] == "v2-fcst202609170500" and body["hours"] == 3
    assert body["note"] == main.NOTE
    assert [s["station_id"] for s in body["stations"]] == ["ST-1"]
    assert body["stations"][0]["expected_rentals"] == 6.0
    assert body["stations"][0]["shortfall"] == 5.0
    assert body["stations"][0]["as_of"] == "2026-09-17T08:20:00+09:00"

    wider = client.get("/shortage-risk", params={"max_snapshot_age_minutes": 60}).json()
    assert [s["station_id"] for s in wider["stations"]] == ["ST-3", "ST-1"]  # 부족 6 > 5
    empty = client.get("/shortage-risk", params={"district": "마포구"}).json()
    assert empty["stations"] == []  # 필터 뒤에 없으면 200 빈 목록
    assert client.get("/shortage-risk", params={"hours": 7}).status_code == 422


def test_shortage_risk_skips_station_missing_needed_hours(client, pg_engine):
    add_stations(pg_engine)
    insert(pg_engine, RealtimeSnapshot, [
        {"station_id": "ST-1", "fetched_at": NOW, "bike_count": 0},
    ])  # fmt: skip
    first = NOW.replace(minute=0)
    insert(pg_engine, Prediction, [
        {"station_id": "ST-1", "hour_start": first + timedelta(hours=h), "model_version": "v",
         "predicted_rentals": 3.0, "created_at": NOW}
        for h in (0, 1, 2)  # 08:25 + 3시간이면 11시도 필요
    ])  # fmt: skip
    assert client.get("/shortage-risk").json()["stations"] == []


def test_shortage_risk_503s(client, pg_engine):
    add_stations(pg_engine)
    assert client.get("/shortage-risk").json() == {"detail": "no predictions"}
    add_predictions(pg_engine, "v", 1.0, NOW)
    response = client.get("/shortage-risk")
    assert response.status_code == 503 and response.json() == {"detail": "no recent snapshot"}


def test_daily_predictions_pick_version_within_the_day(client, pg_engine):
    add_stations(pg_engine)
    add_predictions(pg_engine, "v1-old", 1.0, NOW - timedelta(hours=5))
    tomorrow = NOW.replace(hour=0, minute=0) + timedelta(days=1)
    with pg_engine.begin() as conn:
        conn.execute(Prediction.__table__.insert(), [
            {"station_id": "ST-1", "hour_start": tomorrow + timedelta(hours=h),
             "model_version": "v2-new", "predicted_rentals": 2.0, "created_at": NOW}
            for h in range(3)
        ] + [
            {"station_id": "ST-2", "hour_start": tomorrow + timedelta(hours=h),
             "model_version": "v1-old", "predicted_rentals": 1.0,
             "created_at": NOW - timedelta(hours=5)}
            for h in range(3)
        ])  # fmt: skip
    today = client.get("/predictions/ST-1", params={"date": "2026-09-17"}).json()
    assert today["model_version"] == "v1-old"  # 오늘 행이 있는 버전 중에서 고름
    assert today["hours"][0]["hour_start"] == "2026-09-17T00:00:00+09:00"
    later = client.get("/predictions/ST-1", params={"date": "2026-09-18"}).json()
    assert later["model_version"] == "v2-new"
    # 같은 18일에 ST-1은 새 버전만, ST-2는 옛 버전만 있다: ST-2도 옛 버전으로 채우지 않고
    # 새 버전 + 빈 목록(T34, T38)
    other = client.get("/predictions/ST-2", params={"date": "2026-09-18"}).json()
    assert other["model_version"] == "v2-new" and other["hours"] == []
    none_that_day = client.get("/predictions/ST-1", params={"date": "2026-10-01"}).json()
    assert none_that_day["model_version"] is None and none_that_day["hours"] == []
    assert client.get("/predictions/ST-1", params={"date": "2026-10-01"}).json()["hours"] == []
    assert client.get("/predictions/ST-404", params={"date": "2026-09-17"}).status_code == 404
    assert client.get("/predictions/ST-1").status_code == 422


def test_health_503_when_database_is_down(client):
    from sqlalchemy import create_engine

    broken = create_engine("postgresql+psycopg://x:x@127.0.0.1:1/none?connect_timeout=1")
    main.app.dependency_overrides[main.get_engine] = lambda: broken
    response = client.get("/health")
    assert response.status_code == 503 and response.json() == {"detail": "database unavailable"}


def test_every_endpoint_503_json_when_database_is_down():
    """DB에 접속하지 못하면 /health(위 테스트)만이 아니라 모든 요청이 503 JSON(docs/api.md 공통).

    pg_engine을 쓰지 않아 DB 없이도 돈다.
    """
    from sqlalchemy import create_engine

    broken = create_engine("postgresql+psycopg://x:x@127.0.0.1:1/none?connect_timeout=1")
    main.app.dependency_overrides[main.get_engine] = lambda: broken
    main.app.dependency_overrides[main.get_now] = lambda: NOW
    try:
        client = TestClient(main.app)
        for path in [
            "/stations",
            "/stations/ST-1",
            "/shortage-risk",
            "/predictions/ST-1?date=2026-09-17",
        ]:
            response = client.get(path)
            assert response.status_code == 503, path
            assert response.json() == {"detail": "database unavailable"}, path
    finally:
        main.app.dependency_overrides.clear()
        broken.dispose()


def test_database_unavailable_logs_the_cause(caplog):
    """503은 원인을 숨기지만 로그에는 남긴다(비밀번호·DB 이름 오류도 OperationalError)."""
    from sqlalchemy import create_engine

    broken = create_engine("postgresql+psycopg://x:x@127.0.0.1:1/none?connect_timeout=1")
    main.app.dependency_overrides[main.get_engine] = lambda: broken
    try:
        with caplog.at_level("ERROR", logger="bike_demand.api.main"):
            assert TestClient(main.app).get("/stations").status_code == 503
        assert any(r.getMessage().startswith("database unavailable: ") for r in caplog.records)
    finally:
        main.app.dependency_overrides.clear()
        broken.dispose()
