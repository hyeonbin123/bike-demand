# [무엇을 검증하나] FastAPI 서비스(src/bike_demand/api/main.py)가 docs/api.md의 계약대로 답하는지
# 확인한다: 응답 형식과 KST 시각, 예측 버전 고르기(가장 최근 created_at 한 버전, 섞지 않음),
# /shortage-risk의 부분 시간 비례·필터·정렬·503과 빈 목록의 경계, DB 장애(503)와 쿼리 오류(500)의
# 구분.
# [방식] 앞쪽 테스트는 conftest의 pg_engine(임시 PostgreSQL)에 행을 직접 넣고 TestClient로 요청한다.
# 뒤쪽 장애 테스트는 접속되지 않는 엔진이나 가짜 엔진을 끼워 DB 없이도 돈다.
# [왜 중요한가] 대시보드와 사용자가 보는 숫자가 모두 이 응답에서 나온다. 버전이 섞이거나 오래된
# 스냅샷이 섞이면 서로 다른 시점의 값을 한 화면에 보여 주게 된다.

from datetime import datetime, timedelta, timezone

# TestClient: 서버를 띄우지 않고 앱에 HTTP 요청을 보낸다. main: 의존성(get_engine, get_now)을 바꿔
# 끼울 앱 모듈. 모델 클래스는 테스트 행을 넣는 데 쓴다.
import pytest
from fastapi.testclient import TestClient

from bike_demand.api import main
from bike_demand.serving.models import Prediction, RealtimeSnapshot, Station

# '지금'을 2026-09-17 08:25 KST로 고정해 시간 경계(현재 시간대, 30분 스냅샷 한도, 부분 시간 비례)를
# 정확한 숫자로 검증한다.
KST = timezone(timedelta(hours=9))
NOW = datetime(2026, 9, 17, 8, 25, tzinfo=KST)


# 앱의 DB 엔진을 임시 DB로, 현재 시각을 NOW로 바꿔 끼운 TestClient. FastAPI의 dependency_overrides를
# 써서 앱 코드를 고치지 않고 의존성만 바꾼다. 테스트가 끝나면 원래대로 되돌려 다른 테스트에 새지
# 않게 한다.
@pytest.fixture
def client(pg_engine):
    main.app.dependency_overrides[main.get_engine] = lambda: pg_engine
    main.app.dependency_overrides[main.get_now] = lambda: NOW
    try:
        yield TestClient(main.app)
    finally:
        main.app.dependency_overrides.clear()


# ORM 클래스의 테이블에 행 사전 목록을 한 트랜잭션으로 넣는 도우미.
def insert(engine, table, rows):
    with engine.begin() as conn:
        conn.execute(table.__table__.insert(), rows)


# 대여소 3곳: 강남구 2곳(필터 확인), 마포구 1곳은 좌표가 없다(null 좌표 처리 확인).
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


# 지정한 버전·created_at으로 대여소마다 NOW 기준 8시간 전 ~ 15시간 뒤의 매시간 예측(같은 값)을
# 넣는다. created_at을 바꿔 '어느 버전이 가장 최근인가'를 조절한다.
def add_predictions(engine, version, value, created_at, stations=("ST-1", "ST-2", "ST-3")):
    first = NOW.replace(minute=0)
    rows = [
        {"station_id": s, "hour_start": first + timedelta(hours=h), "model_version": version,
         "predicted_rentals": value, "created_at": created_at}
        for s in stations for h in range(-8, 16)
    ]  # fmt: skip
    insert(engine, Prediction, rows)


# /health: 자료가 없으면 마지막 스냅샷·예측 시간이 null이고 상태는 ok. 자료를 넣으면 마지막 스냅샷
# 시각과 최신 버전의 마지막 예측 시간이 KST(+09:00)로 나온다. 08시 + 15시간 = 23시가 마지막 예측
# 시간이다.
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


# /stations: station_id 순서, 자치구 정확히 일치 필터, 좌표는 반올림하지 않고 저장값 그대로(계약의
# '숫자 반올림'), 좌표 없는 대여소는 null.
def test_stations_filter_and_null_coordinates(client, pg_engine):
    add_stations(pg_engine)
    assert [s["station_id"] for s in client.get("/stations").json()] == ["ST-1", "ST-2", "ST-3"]
    gangnam = client.get("/stations", params={"district": "강남구"}).json()
    assert [s["station_id"] for s in gangnam] == ["ST-1", "ST-2"]
    assert gangnam[0]["lat"] == 37.49812  # 좌표는 반올림하지 않음
    assert client.get("/stations").json()[2]["lat"] is None


# /stations/{id}: 가장 최근 스냅샷(10분 전 것)을 주고, 예측은 created_at이 가장 늦은 버전 하나만
# 쓴다. 새 버전에 이 대여소가 없으면 옛 버전으로 채우지 않고 빈 목록을 준다(버전을 섞지 않는 규칙).
# 예측은 현재 시간이 든 08시부터 6시간, predicted_rentals는 소수 둘째 자리(1.234 → 1.23). 없는
# 대여소는 404.
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


# created_at이 같은 두 버전이 있으면 버전 문자열이 큰 쪽을 고른다(docs/api.md). 같은 조건에서 매번
# 같은 버전이 골라져야 응답이 흔들리지 않는다.
def test_version_tie_breaks_on_name(client, pg_engine):
    add_stations(pg_engine)
    same_time = NOW - timedelta(hours=1)
    add_predictions(pg_engine, "v1-a", 1.0, same_time, stations=("ST-1",))
    add_predictions(pg_engine, "v1-b", 2.0, same_time, stations=("ST-1",))
    assert client.get("/stations/ST-1").json()["model_version"] == "v1-b"


# /shortage-risk 핵심 계산. ST-1은 08:20 스냅샷이라 [08:20, 11:20) 구간 = 08시 × 40/60 + 09시 + 10시
# + 11시 × 20/60 = 2 × 3 = 6대, 자전거 1대라 부족 5. 부족이 0 이하인 ST-2는 빠지고, 40분 전 스냅샷인
# ST-3은 기본 30분 한도에서 빠진다.
# 한도를 60분으로 넓히면 ST-3(부족 6)이 ST-1(5)보다 앞에 온다(부족 큰 순). 자치구 필터 뒤에 남는
# 곳이 없으면 503이 아니라 200 빈 목록, hours 범위(1~6) 밖은 422.
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


# 08:25부터 3시간이면 11시 예측도 일부(25분) 필요하다. 그 시간이 없으면 계산하지 않고 그 대여소를
# 뺀다(필요한 시간의 예측이 하나라도 없으면 제외). 빠진 시간을 0으로 보면 부족을 낮게 잡게 된다.
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


# 503의 두 경우와 순서: 예측이 아예 없으면 'no predictions'(먼저 확인), 예측은 있는데 30분 안
# 스냅샷이 하나도 없으면 'no recent snapshot'. 판단할 자료 자체가 없을 때만 503이고, 필터 때문에 빈
# 경우는 200이다.
def test_shortage_risk_503s(client, pg_engine):
    add_stations(pg_engine)
    assert client.get("/shortage-risk").json() == {"detail": "no predictions"}
    add_predictions(pg_engine, "v", 1.0, NOW)
    response = client.get("/shortage-risk")
    assert response.status_code == 503 and response.json() == {"detail": "no recent snapshot"}


# /predictions/{id}?date=: 버전을 '요청한 날에 어느 대여소든 행이 있는 버전' 중에서 고른다.
# 오늘(17일)은 옛 버전만 행이 있으므로 옛 버전, 18일은 새 버전.
# 18일에 ST-2는 옛 버전 행만 있지만 새 버전 + 빈 목록이 맞다(대여소별로 버전을 고르지 않음,
# docs/api.md). 행이 없는 날은 버전 null, 없는 대여소 404, date 없으면 422.
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


# 닫힌 포트(1번)로 가는 엔진을 끼우면 /health가 503 {'detail': 'database unavailable'}.
# connect_timeout=1로 오래 기다리지 않는다. 이 테스트는 client fixture 때문에 임시 DB가 있어야 돌고,
# DB 없이 같은 것을 보는 테스트는 아래에 따로 있다.
def test_health_503_when_database_is_down(client):
    from sqlalchemy import create_engine

    broken = create_engine("postgresql+psycopg://x:x@127.0.0.1:1/none?connect_timeout=1")
    main.app.dependency_overrides[main.get_engine] = lambda: broken
    response = client.get("/health")
    assert response.status_code == 503 and response.json() == {"detail": "database unavailable"}


# DB 장애 때 /health만이 아니라 모든 경로가 같은 503 JSON으로 답하는지 본다(앱 전체 OperationalError
# 처리기). 처리기를 넣기 전에는 /health 말고는 연결 실패가 처리되지 않은 예외로 나갔다(112e278).
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


# 응답에는 원인을 적지 않지만 서버 로그에는 'database unavailable: …'로 원인을 남기는지 caplog로
# 확인한다. 비밀번호·DB 이름이 틀린 설정 오류도 OperationalError로 오므로 로그가 없으면 원인을 찾을
# 수 없다.
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


# 회귀 테스트(3f2ed20): /health가 자기 안에서 예외를 잡아 공통 처리기를 건너뛰면 로그가 남지 않았다.
# 이제 /health도 공통 처리기로 503과 로그를 낸다.
def test_health_logs_the_cause_like_every_endpoint(caplog):
    """/health도 공통 처리기로 503을 답하고 원인을 로그에 남긴다(T58). DB 없이 돈다."""
    from sqlalchemy import create_engine

    broken = create_engine("postgresql+psycopg://x:x@127.0.0.1:1/none?connect_timeout=1")
    main.app.dependency_overrides[main.get_engine] = lambda: broken
    try:
        with caplog.at_level("ERROR", logger="bike_demand.api.main"):
            response = TestClient(main.app).get("/health")
        assert response.status_code == 503
        assert response.json() == {"detail": "database unavailable"}
        assert any(r.getMessage().startswith("database unavailable: ") for r in caplog.records)
    finally:
        main.app.dependency_overrides.clear()
        broken.dispose()


# 엔진처럼 connect()·with 문을 흉내 내지만 execute에서 ProgrammingError를 내는 가짜 엔진. 실제 DB
# 없이 '접속은 되는데 쿼리가 틀린' 상황을 만든다.
class QueryBugEngine:
    """접속은 되지만 쿼리가 틀린 DB(예: 마이그레이션 전이라 표가 없음)."""

    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *args, **kwargs):
        from sqlalchemy.exc import ProgrammingError

        raise ProgrammingError("select 1", {}, Exception("relation does not exist"))


# 장애 처리를 확인할 모든 API 경로(대시보드 정적 파일 제외).
ALL_PATHS = [
    "/health",
    "/stations",
    "/stations/ST-1",
    "/shortage-risk",
    "/predictions/ST-1?date=2026-09-17",
]


# 쿼리 오류(예: 마이그레이션 전이라 표가 없음)는 503으로 가리지 않고 500이어야 한다(3f2ed20). 쿼리
# 오류까지 503으로 답하면 코드·스키마 문제가 일시적인 DB 장애처럼 보이므로 500으로 둔다(api/main.py
# 처리기 주석). raise_server_exceptions=False로 서버 예외를 500 응답으로 받는다.
def test_query_errors_stay_500_on_every_endpoint():
    """쿼리 오류는 DB 장애로 가리지 않는다: /health를 포함해 모든 요청이 500(T58)."""
    main.app.dependency_overrides[main.get_engine] = lambda: QueryBugEngine()
    main.app.dependency_overrides[main.get_now] = lambda: NOW
    try:
        client = TestClient(main.app, raise_server_exceptions=False)
        for path in ALL_PATHS:
            assert client.get(path).status_code == 500, path
    finally:
        main.app.dependency_overrides.clear()


# 연결 풀이 다 차서 pool_timeout을 넘기면(sqlalchemy TimeoutError) 접속 실패와 같은 503(3f2ed20).
# 연결 하나짜리 풀(SQLite 메모리)을 만들고 그 연결을 잡아 두어 DB 서버 없이 풀 대기 시간 초과를
# 만든다.
def test_pool_timeout_is_503_on_every_endpoint():
    """연결 풀이 다 차서 기다린 시간이 넘으면 DB에 접속하지 못한 것과 같은 503(T58)."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import QueuePool

    busy = create_engine(
        "sqlite://", poolclass=QueuePool, pool_size=1, max_overflow=0, pool_timeout=0.1
    )
    held = busy.connect()  # 하나뿐인 연결을 잡아 둔다
    main.app.dependency_overrides[main.get_engine] = lambda: busy
    main.app.dependency_overrides[main.get_now] = lambda: NOW
    try:
        client = TestClient(main.app, raise_server_exceptions=False)
        for path in ALL_PATHS:
            response = client.get(path)
            assert response.status_code == 503, path
            assert response.json() == {"detail": "database unavailable"}, path
    finally:
        main.app.dependency_overrides.clear()
        held.close()
        busy.dispose()
