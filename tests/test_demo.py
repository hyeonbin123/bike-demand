# [무엇을 검증하나] 닫힌 서비스를 인증키·Airflow·서비스 모델 없이 다시 보여 주는 '고정 시각 데모'의
# 세 부분을 확인한다.
# - 데모 시계(api/main.py): 환경 변수 BIKE_DEMAND_DEMO_NOW가 없으면 '지금'은 실제 시각(기본 동작
#   그대로), 있으면 그 시각이다. 시간대가 없거나 읽을 수 없는 값이면 앱이 시작하지 않는다. /health에
#   demo_now가 나온다
# - 묶음 만들기(scripts/build_demo_bundle.py): 원본 스냅샷 한 번·예보 발표 하나·warehouse
#   대여소·서비스 세대로 대여소·스냅샷·예보·예측 Parquet과 manifest.json을 쓴다. 예측은 서비스 예측
#   (predict.predict_and_store)과 같은 계산·같은 model_version 이름이고, 같은 입력이면 같은 바이트다
# - 묶음 형식과 빈 DB 적재(serving/load.py): manifest의 SHA-256과 다른 파일은 받지 않고, 네 테이블이
#   모두 비었을 때만 넣는다. 넣은 뒤 데모 시각의 /shortage-risk가 기대 순위다(이 부분과 서비스 DB에
#   저장한 예측과의 대조는 임시 PostgreSQL이 있어야 돈다)
# [왜 중요한가] 실시간 API는 지금 상태만 주므로 지난 시각의 화면은 저장한 자료로만 다시 볼 수 있다.
# 데모 시계가 기본 동작을 바꾸거나, 묶음이 서비스와 다른 계산을 하거나, 데모 적재가 개발 DB에
# 섞이면 '그때 화면'이 아닌 것을 보여 주게 된다.

import hashlib
import importlib.util
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

# duckdb: 대여소 warehouse 파일. lightgbm·numpy: 저장 흐름을 시험할 작은 모델. pyarrow: 묶음 파일을
# 다시 써서 형식 검사를 시험한다. sqlalchemy: 적재 결과 조회, 접속이 안 되는 엔진 만들기.
import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select, text

from bike_demand.api import main
from bike_demand.model import artifacts, final, frames, predict
from bike_demand.model.forecast_weather import hourly_weather
from bike_demand.serving import load
from bike_demand.serving.models import Prediction, RealtimeSnapshot, Station, WeatherForecast

KST = timezone(timedelta(hours=9))
# scripts/는 패키지가 아니라 import할 수 없으므로 파일 경로로 불러온다
# (tests/test_model_provenance.py와 같은 방식).
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_demo_bundle.py"
# small_warehouse의 학습 기간(1/1~1/15). 서비스 세대 산출물은 이 기간으로 만든다.
HISTORY = ("2024-01-01", "2024-01-16")
GENERATION = "v9-T-20260101T0000"


# importlib로 스크립트 파일을 모듈로 읽는다. if __name__ == '__main__' 아래 실행 부분은 돌지 않는다.
@pytest.fixture
def builder():
    spec = importlib.util.spec_from_file_location("build_demo_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------
# 데모 시계(DB 없이 돈다)
# ---------------------------------------------------------------------------------------------


# 시간대가 있는 ISO 8601 시각은 같은 순간의 KST 시각으로 읽는다(UTC 'Z'도). 앞뒤 공백은 무시한다.
def test_parse_demo_now_reads_offset_times_as_kst():
    fixed = datetime(2026, 9, 19, 18, 11, tzinfo=KST)
    assert main.parse_demo_now("2026-09-19T18:11:00+09:00") == fixed
    from_utc = main.parse_demo_now("2026-09-19T09:11:00Z")
    assert from_utc == fixed and from_utc.utcoffset() == timedelta(hours=9)
    assert main.parse_demo_now("  2026-09-19T18:11:00+09:00\n") == fixed
    # 범위 경계는 KST 날짜로 본다: UTC 1999-12-31 15시는 KST 2000-01-01 0시라 받는다
    assert main.parse_demo_now("1999-12-31T15:00:00Z") == datetime(2000, 1, 1, tzinfo=KST)


# 변수가 없거나 빈 값이면 데모 시계는 꺼져 있다(docker compose의 '${VAR:-}'처럼 빈 값이 오는 경우).
@pytest.mark.parametrize("raw", [None, "", "   "])
def test_parse_demo_now_is_off_when_unset_or_blank(raw):
    assert main.parse_demo_now(raw) is None


# 시간대가 없는 시각·날짜만·읽을 수 없는 값은 오류다. 시간대 없는 값을 서버 시간대로 짐작하지
# 않는다. KST 날짜가 2000-01-01~2100-12-31 밖이면(응답 계산에서 datetime 범위를 넘을 수 있는 극단
# 값, KST로 옮기다 넘치는 1년 1월 1일 포함) 오류다.
@pytest.mark.parametrize(
    "raw",
    [
        "2026-09-19T18:11:00",
        "2026-09-19",
        "tomorrow",
        "2026-13-01T00:00:00+09:00",
        "1999-12-31T23:59:59+09:00",
        "2101-01-01T00:00:00+09:00",
        "0001-01-01T00:00:00+14:00",
    ],
)
def test_parse_demo_now_rejects_values_without_offset_or_unreadable(raw):
    with pytest.raises(main.DemoClockError, match=main.DEMO_NOW_ENV):
        main.parse_demo_now(raw)


# 변수가 없으면 get_now는 지금 시각(KST)이다. 데모 기능을 넣기 전과 같은 동작이다.
def test_get_now_is_the_real_clock_when_the_demo_clock_is_off(monkeypatch):
    monkeypatch.delenv(main.DEMO_NOW_ENV, raising=False)
    before = datetime.now(KST)
    now = main.get_now()
    after = datetime.now(KST)
    assert before <= now <= after
    assert now.utcoffset() == timedelta(hours=9)


# 변수가 있으면 get_now는 그 시각을 그대로 돌려주고 흐르지 않는다.
def test_get_now_is_the_fixed_time_when_the_demo_clock_is_on(monkeypatch):
    monkeypatch.setenv(main.DEMO_NOW_ENV, "2026-09-19T18:11:00+09:00")
    assert main.get_now() == datetime(2026, 9, 19, 18, 11, tzinfo=KST)
    assert main.get_now() == main.get_now()


# 잘못된 값이면 요청을 받기 전에 앱 시작(lifespan)에서 멈춘다. 요청마다 500을 내는 대신 서버가
# 뜨지 않게 해 설정 실수를 바로 드러낸다.
def test_app_does_not_start_with_an_unreadable_demo_clock(monkeypatch):
    monkeypatch.setenv(main.DEMO_NOW_ENV, "2026-09-19 18시 11분")
    with pytest.raises(main.DemoClockError):
        with TestClient(main.app):
            pass


# 올바른 값이면 앱이 뜨고, 시작 로그에 고정한 시각을 남긴다(실시간 서비스로 착각하지 않게).
def test_app_starts_with_a_valid_demo_clock_and_logs_it(monkeypatch, caplog):
    monkeypatch.setenv(main.DEMO_NOW_ENV, "2026-09-19T18:11:00+09:00")
    with caplog.at_level("WARNING", logger="bike_demand.api.main"):
        with TestClient(main.app) as client:
            assert client.get("/").status_code == 200
    assert any("2026-09-19T18:11:00+09:00" in r.getMessage() for r in caplog.records)


# /health의 쿼리들(select 1, 최근 스냅샷 시각, 최신 버전)에 '자료 없음'(None)으로 답하는 가짜
# 엔진. DB 없이 /health 응답의 형식만 본다.
class EmptyServiceEngine:
    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *args, **kwargs):
        return self

    def scalar_one(self):
        return None

    def scalar_one_or_none(self):
        return None


# 데모 시계가 꺼져 있으면 demo_now는 null이고 나머지 응답은 전과 같다. 켜져 있으면 그 시각(KST).
def test_health_reports_the_demo_clock(monkeypatch):
    main.app.dependency_overrides[main.get_engine] = EmptyServiceEngine
    try:
        client = TestClient(main.app)
        monkeypatch.delenv(main.DEMO_NOW_ENV, raising=False)
        assert client.get("/health").json() == {
            "status": "ok",
            "database": "ok",
            "latest_snapshot_at": None,
            "latest_prediction_hour": None,
            "demo_now": None,
        }
        monkeypatch.setenv(main.DEMO_NOW_ENV, "2026-09-19T09:11:00Z")
        assert client.get("/health").json()["demo_now"] == "2026-09-19T18:11:00+09:00"
    finally:
        main.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------------------------
# 묶음 형식(DB 없이 돈다)
# ---------------------------------------------------------------------------------------------
DEMO_NOW_TEXT = "2026-09-19T18:15:00+09:00"
VERSION = f"{GENERATION}-fcst202609191700"


# 손으로 만든 작은 묶음. 과거 이력 대여소 3곳(ST-A·ST-B 강남구, ST-C 마포구), 18:10 스냅샷 한 번
# (ST-D는 실시간에만 있는 대여소, ST-E는 자전거 수가 '-'라 스냅샷에서 빠지는 행), 예보 두 행, 예측은
# ST-A 시간당 4대·ST-B 1대·ST-C 0.5대(18~23시). 스냅샷·예보는 수집기가 만드는 bronze 형식(모든 값
# 문자열)이다.
def bundle_tables():
    fetched = "2026-09-19T18:10:00+09:00"
    stations = [
        {"station_id": "ST-A", "station_no": 1, "station_name": "가역", "district": "강남구",
         "lat": 37.5, "lon": 127.0, "docks": 10},
        {"station_id": "ST-B", "station_no": 2, "station_name": "나역", "district": "강남구",
         "lat": 37.51, "lon": 127.01, "docks": 10},
        {"station_id": "ST-C", "station_no": 3, "station_name": "다역", "district": "마포구",
         "lat": 37.55, "lon": 126.9, "docks": 10},
    ]  # fmt: skip
    snapshots = [
        {"fetched_at": fetched, "rack_count": "10", "station_name": name, "bike_count": bikes,
         "shared": "0", "lat": "37.5", "lon": "127.0", "station_id": station_id}
        for station_id, name, bikes in (
            ("ST-A", "1. 가역", "1"), ("ST-B", "2. 나역", "5"), ("ST-C", "3. 다역", "0"),
            ("ST-D", "4. 라역", "3"), ("ST-E", "5. 마역", "-"),
        )
    ]  # fmt: skip
    forecasts = [
        {"base_datetime": "2026-09-19T17:00:00+09:00", "fcst_datetime": "2026-09-19T18:00:00+09:00",
         "category": category, "value": value, "nx": "60", "ny": "127",
         "fetched_at": "2026-09-19T17:15:02+09:00"}
        for category, value in (("TMP", "22"), ("PCP", "강수없음"))
    ]  # fmt: skip
    predictions = [
        {"station_id": station_id, "hour_start": f"2026-09-19T{hour:02d}:00:00+09:00",
         "model_version": VERSION, "predicted_rentals": value}
        for station_id, value in (("ST-A", 4.0), ("ST-B", 1.0), ("ST-C", 0.5))
        for hour in range(18, 24)
    ]  # fmt: skip
    return {
        "stations": stations,
        "snapshots": snapshots,
        "forecasts": forecasts,
        "predictions": predictions,
    }


# 쓴 묶음의 manifest가 파일마다 SHA-256·바이트 수·행 수를 적고, 읽으면 쓴 행이 그대로 나온다.
def test_bundle_round_trip_records_file_fingerprints(tmp_path):
    directory = tmp_path / "bundle"
    tables = bundle_tables()
    manifest = load.write_demo_bundle(directory, tables, {"demo_now": DEMO_NOW_TEXT})
    assert manifest["format"] == load.DEMO_FORMAT
    assert manifest["format_version"] == load.DEMO_FORMAT_VERSION
    for name, rows in tables.items():
        data = (directory / f"{name}.parquet").read_bytes()
        entry = manifest["files"][f"{name}.parquet"]
        assert entry == {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "rows": len(rows),
        }
    on_disk = (directory / load.DEMO_MANIFEST).read_bytes()
    assert json.loads(on_disk.decode("utf-8")) == manifest
    assert b"\r\n" not in on_disk  # LF(.gitattributes와 같은 줄 끝)
    bundle = load.read_demo_bundle(directory)
    assert bundle.manifest == manifest
    assert bundle.stations == tables["stations"]
    assert bundle.snapshots == tables["snapshots"]
    assert bundle.forecasts == tables["forecasts"]
    assert bundle.predictions == tables["predictions"]


# 바뀐 파일(SHA-256 불일치), 빠진 파일, 다른 형식 이름, 열이 다른 파일은 DB에 닿기 전에 거부한다.
def test_tampered_or_incomplete_bundles_are_rejected(tmp_path):
    def fresh(name):
        directory = tmp_path / name
        load.write_demo_bundle(directory, bundle_tables(), {})
        return directory

    changed = fresh("changed")
    path = changed / "predictions.parquet"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(load.DemoBundleError, match="predictions.parquet"):
        load.read_demo_bundle(changed)

    missing = fresh("missing")
    (missing / "snapshots.parquet").unlink()
    with pytest.raises(load.DemoBundleError, match="snapshots.parquet"):
        load.read_demo_bundle(missing)

    with pytest.raises(load.DemoBundleError, match=load.DEMO_MANIFEST):
        load.read_demo_bundle(tmp_path / "nothing-here")

    other = fresh("other-format")
    manifest = json.loads((other / load.DEMO_MANIFEST).read_text("utf-8"))
    manifest["format_version"] = load.DEMO_FORMAT_VERSION + 1
    (other / load.DEMO_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(load.DemoBundleError, match="format"):
        load.read_demo_bundle(other)

    # 지문은 맞지만 열이 다른 파일(예측 값 열 이름이 다름): manifest를 그 파일에 맞춰 고쳐도 거부
    columns = fresh("columns")
    wrong = pa.table({"station_id": ["ST-A"], "hour_start": ["2026-09-19T18:00:00+09:00"],
                      "model_version": [VERSION], "value": [1.0]})  # fmt: skip
    pq.write_table(wrong, columns / "predictions.parquet")
    data = (columns / "predictions.parquet").read_bytes()
    manifest = json.loads((columns / load.DEMO_MANIFEST).read_text("utf-8"))
    manifest["files"]["predictions.parquet"] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "rows": 1,
    }
    (columns / load.DEMO_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(load.DemoBundleError, match="predictions.parquet"):
        load.read_demo_bundle(columns)


# warehouse dim_stations와 같은 열 타입(dbt 결과: 번호·거치대 INTEGER, 좌표 DOUBLE)의 표를 만든다.
# 실제 표에는 열이 더 있지만(정보 스냅샷, 첫·마지막 대여 등) 읽는 쪽은 일곱 열만 고른다.
def make_station_table(path, rows):
    con = duckdb.connect(str(path))
    con.execute("""
        create table dim_stations (station_id varchar, station_no integer, station_name varchar,
                                   district varchar, lat double, lon double, docks integer,
                                   info_snapshot varchar)
    """)
    con.executemany("insert into dim_stations values (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.close()


# load_stations_from_warehouse에서 떼어 낸 읽기 함수: dim_stations 7열을 station_id 순으로.
def test_read_warehouse_stations_returns_dim_stations_in_id_order(tmp_path):
    path = tmp_path / "wh.duckdb"
    make_station_table(path, [
        ("ST-2", 2, "좌표없음", None, None, None, None, "26.6월"),
        ("ST-1", 1, "강남역", "강남구", 37.49, 127.02, 20, "26.6월"),
    ])  # fmt: skip
    assert load.read_warehouse_stations(path) == [
        {"station_id": "ST-1", "station_no": 1, "station_name": "강남역", "district": "강남구",
         "lat": 37.49, "lon": 127.02, "docks": 20},
        {"station_id": "ST-2", "station_no": 2, "station_name": "좌표없음", "district": None,
         "lat": None, "lon": None, "docks": None},
    ]  # fmt: skip


# load_forecasts에서 떼어 낸 변환 함수: bronze 문자열을 시간대 있는 시각·정수 격자로, 값은 그대로.
def test_forecast_rows_parse_bronze_strings():
    rows = load.forecast_rows(bundle_tables()["forecasts"])
    assert rows[1] == {
        "base_datetime": datetime(2026, 9, 19, 17, tzinfo=KST),
        "fcst_datetime": datetime(2026, 9, 19, 18, tzinfo=KST),
        "category": "PCP",
        "nx": 60,
        "ny": 127,
        "value": "강수없음",
    }


# ---------------------------------------------------------------------------------------------
# 묶음 만들기(scripts/build_demo_bundle.py, DB 없이 돈다)
# ---------------------------------------------------------------------------------------------
SNAPSHOT_AT = "2026-09-19T18:10:02.500000+09:00"
FORECAST_FETCHED_AT = "2026-09-19T17:15:02+09:00"
FORECAST_BASE = datetime(2026, 9, 19, 17, tzinfo=KST)
# 원본 실시간 응답의 행(모든 값이 문자열). ST-1·ST-2는 학습 산출물에 있는 대여소, ST-3은 warehouse
# 대여소 표에는 있지만 산출물에는 없는 대여소, ST-9는 실시간에만 있는 대여소, ST-8은 자전거 수가
# '-'라 적재 때 스냅샷에서 빠지는 행이다.
SNAPSHOT_ROWS = [
    {"rackTotCnt": rack, "stationName": name, "parkingBikeTotCnt": bikes, "shared": "0",
     "stationLatitude": lat, "stationLongitude": lon, "stationId": station_id}
    for station_id, name, bikes, rack, lat, lon in (
        ("ST-1", "1. 강남역", "1", "20", "37.49", "127.02"),
        ("ST-2", "2. 합정역", "5", "10", "37.55", "126.91"),
        ("ST-3", "3. 신규역", "0", "15", "37.60", "127.10"),
        ("ST-9", "909. 새 대여소", "2", "12", "37.51", "127.05"),
        ("ST-8", "808. 값 오류", "-", "8", "37.52", "127.06"),
    )
]  # fmt: skip
# 예보 항목: 18~21시 예보(→ hour_start 17~20시, 예보 시각 − 1시간). 시간마다 기온을 다르게 둔다.
FORECAST_HOURS = (18, 19, 20, 21)


def forecast_values(hour):
    return (("TMP", str(hour)), ("PCP", "강수없음"), ("WSD", "1.5"), ("REH", "60"),
            ("SNO", "적설없음"))  # fmt: skip


# 수집기가 저장하는 원본 실시간 스냅샷 JSON(fetched_at + rentBikeStatus.row + pages).
def write_raw_snapshot(path, fetched_at=SNAPSHOT_AT, rows=SNAPSHOT_ROWS):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rentBikeStatus": {"list_total_count": len(rows), "RESULT": {"CODE": "INFO-000"},
                           "row": rows},
        "pages": [],
        "fetched_at": fetched_at,
    }  # fmt: skip
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# 수집기가 저장하는 원본 단기예보 JSON(발표 시각·격자·items·fetched_at).
def write_raw_forecast(path, fetched_at=FORECAST_FETCHED_AT):
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [
        {"baseDate": "20260919", "baseTime": "1700", "category": category,
         "fcstDate": "20260919", "fcstTime": f"{hour:02d}00", "fcstValue": value,
         "nx": 60, "ny": 127}
        for hour in FORECAST_HOURS
        for category, value in forecast_values(hour)
    ]  # fmt: skip
    payload = {
        "base_datetime": FORECAST_BASE.isoformat(),
        "nx": "60",
        "ny": "127",
        "pages": [],
        "items": items,
        "fetched_at": fetched_at,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# 특징 이름만 서비스 모델과 같은, 무작위 자료로 5라운드 학습한 작은 LightGBM(test_model_predict.py와
# 같은 방식). 값의 정확도가 아니라 묶음을 만드는 흐름을 시험하는 용도다.
def tiny_booster():
    rng = np.random.default_rng(0)
    x = rng.random((400, len(frames.FEATURES)), dtype=np.float32)
    y = rng.poisson(2, 400).astype(np.float32)
    data = lgb.Dataset(x, y, feature_name=frames.FEATURES)
    return lgb.train({"objective": "poisson", "verbose": -1, "num_leaves": 4}, data, 5)


# 묶음 만들기의 입력 일체: 원본 스냅샷·예보 JSON, dim_stations만 있는 warehouse 파일(대여소 이름
# 포함), 서비스 세대 폴더(CURRENT → generations/<세대>/model.txt + artifacts/).
@pytest.fixture
def demo_inputs(small_warehouse, tmp_path):
    snapshot = tmp_path / "raw" / "bikelist" / "date=2026-09-19" / "181002.json"
    write_raw_snapshot(snapshot)
    forecast_file = tmp_path / "raw" / "vilage_fcst" / "nx=60_ny=127" / "base=202609191700.json"
    write_raw_forecast(forecast_file)
    warehouse = tmp_path / "warehouse.duckdb"
    make_station_table(warehouse, [
        ("ST-1", 1, "강남역", "강남구", 37.49, 127.02, 20, "26.6월"),
        ("ST-2", 2, "합정역", "마포구", 37.55, 126.91, 10, "26.6월"),
        ("ST-3", 3, "신규역", "강남구", 37.60, 127.10, 15, "26.6월"),
    ])  # fmt: skip
    models = tmp_path / "models"
    generation = models / "serving" / "generations" / GENERATION
    artifacts.export(small_warehouse, HISTORY, generation / "artifacts")
    tiny_booster().save_model(str(generation / "model.txt"))
    (models / "serving" / "CURRENT").write_text(GENERATION + "\n", encoding="utf-8")
    return SimpleNamespace(
        snapshot=snapshot, forecast=forecast_file, warehouse=warehouse, models=models
    )


# 묶음을 만드는 호출(DB 대조는 끔). 출력 폴더를 따로 주면 그곳에 쓴다.
def build(builder, inputs, out, **kwargs):
    return builder.build(
        snapshot=inputs.snapshot,
        forecast=inputs.forecast,
        warehouse=inputs.warehouse,
        models=inputs.models,
        out=out,
        compare_db=False,
        **kwargs,
    )


# 서비스 DB에 위 원본을 적재했을 때 predict.latest_stations가 돌려줄 예측 대상(대여소ID 순): 과거
# 이력 대여소는 warehouse 값, ST-9는 실시간 값(자치구 없음, 거치대 = rackTotCnt). ST-8은 스냅샷 행이
# 없어 빠진다. 묶음 만들기와 따로 손으로 적어 둔 기대값이다.
SERVICE_TARGETS = [
    {"station_id": "ST-1", "district": "강남구", "docks": 20, "lat": 37.49, "lon": 127.02},
    {"station_id": "ST-2", "district": "마포구", "docks": 10, "lat": 37.55, "lon": 126.91},
    {"station_id": "ST-3", "district": "강남구", "docks": 15, "lat": 37.6, "lon": 127.1},
    {"station_id": "ST-9", "district": None, "docks": 12, "lat": 37.51, "lon": 127.05},
]


# 서비스 예측이 쓰는 날씨: 원본 예보 항목을 forecast_weather.hourly_weather로 바꾼 것.
def service_weather():
    return hourly_weather(
        (datetime(2026, 9, 19, hour, tzinfo=KST), category, value)
        for hour in FORECAST_HOURS
        for category, value in forecast_values(hour)
    )


# 원본 두 파일·warehouse·세대로 묶음을 만들면: 대여소 3곳·스냅샷 원본 5행·예보 20행, 예측은 스냅샷에
# 자전거 수가 있는 4곳 × 예보가 있는 17~20시 = 16행. 버전 이름은 '세대-fcst발표시각', 예측 시각은
# 예보를 받은 시각(17:15), 데모 시각은 스냅샷 다음 분(18:11). manifest의 원천 지문이 원본·세대
# 파일과 같고 원본은 바뀌지 않는다. 예측 값은 같은 대여소·시간을 predict.feature_rows + 모델로 직접
# 계산한 값과 같다.
def test_build_writes_the_bundle_from_raw_files_and_the_current_generation(
    builder, demo_inputs, tmp_path
):
    raw_before = demo_inputs.snapshot.read_bytes(), demo_inputs.forecast.read_bytes()
    out = tmp_path / "out"
    manifest = build(builder, demo_inputs, out)
    bundle = load.read_demo_bundle(out)
    assert manifest == bundle.manifest
    assert manifest["generation"] == GENERATION
    assert manifest["model_version"] == VERSION
    assert manifest["demo_now"] == "2026-09-19T18:11:00+09:00"
    assert manifest["predicted_at"] == FORECAST_FETCHED_AT
    assert manifest["snapshot_fetched_at"] == SNAPSHOT_AT
    assert manifest["forecast_base"] == FORECAST_BASE.isoformat()
    rows = {name: entry["rows"] for name, entry in manifest["files"].items()}
    assert rows == {"stations.parquet": 3, "snapshots.parquet": 5, "forecasts.parquet": 20,
                    "predictions.parquet": 16}  # fmt: skip
    assert {r["model_version"] for r in bundle.predictions} == {VERSION}
    assert sorted({r["hour_start"] for r in bundle.predictions}) == [
        f"2026-09-19T{hour:02d}:00:00+09:00" for hour in (17, 18, 19, 20)
    ]
    # 원천 지문과 원본 보존
    sources = manifest["sources"]
    assert sources["snapshot"]["sha256"] == hashlib.sha256(raw_before[0]).hexdigest()
    assert sources["forecast"]["sha256"] == hashlib.sha256(raw_before[1]).hexdigest()
    model_file = demo_inputs.models / "serving" / "generations" / GENERATION / "model.txt"
    model_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
    assert sources["generation"]["files"]["model.txt"] == model_sha
    assert sources["warehouse"]["dim_stations_rows"] == 3
    assert (demo_inputs.snapshot.read_bytes(), demo_inputs.forecast.read_bytes()) == raw_before
    # 예측 값: 손으로 적은 예측 대상과 날씨로 직접 계산한 값과 같다
    loaded = artifacts.load(model_file.parent / "artifacts")
    booster = lgb.Booster(model_file=str(model_file))
    hours = [datetime(2026, 9, 19, hour, tzinfo=KST) for hour in (17, 18, 19, 20)]
    matrix, keys = predict.feature_rows(
        SERVICE_TARGETS, hours, service_weather(), loaded, booster.feature_name()
    )
    expected = {
        (s, h.isoformat()): float(p)
        for (s, h), p in zip(keys, booster.predict(matrix), strict=True)
    }
    actual = {
        (r["station_id"], r["hour_start"]): r["predicted_rentals"] for r in bundle.predictions
    }
    assert actual == expected
    # 출력 폴더 밖에 임시 폴더를 남기지 않는다
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["out", "raw", "warehouse.duckdb", "models"]
    )


# 같은 입력으로 두 번 만들면 네 자료 파일이 바이트까지 같다(manifest는 만든 시각이 달라 제외).
def test_build_is_byte_reproducible(builder, demo_inputs, tmp_path):
    first = build(builder, demo_inputs, tmp_path / "first")
    second = build(builder, demo_inputs, tmp_path / "second")
    assert first["files"] == second["files"]


# 이미 있는 출력 폴더는 덮어쓰지 않는다. 스냅샷 시각보다 뒤에 받은 예보로는 만들지 않는다(그 시각의
# 서비스가 갖고 있지 않던 예보로 예측한 화면이 됨). 거부하면 출력 폴더를 만들지 않는다.
def test_build_refuses_an_existing_output_and_a_forecast_from_after_the_snapshot(
    builder, demo_inputs, tmp_path
):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(SystemExit, match="이미 있음"):
        build(builder, demo_inputs, existing)
    assert list(existing.iterdir()) == []

    write_raw_forecast(demo_inputs.forecast, fetched_at="2026-09-19T18:20:00+09:00")
    with pytest.raises(SystemExit, match="예보"):
        build(builder, demo_inputs, tmp_path / "late")
    assert not (tmp_path / "late").exists()


# 같은 입력이면 묶음의 예측이 서비스 예측 코드(predict.predict_and_store)가 저장할 행과 같다: 시간
# 고르기(예측 시각이 든 정시부터 48시간 중 예보가 있는 시간), 특징 순서, 버전 이름, 값. 서비스 DB를
# 읽는 두 함수(latest_forecast, latest_stations)만 같은 입력을 돌려주게 바꾸고, 저장은 가짜 엔진이
# 받아 적는다. DB를 읽는 부분까지 맞는지는 아래 PostgreSQL 테스트가 본다.
def test_bundle_predictions_equal_the_service_prediction_path(
    builder, demo_inputs, tmp_path, monkeypatch
):
    class RecordingEngine:
        def __init__(self):
            self.records = []

        def begin(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, statement, params=None):
            self.records.extend(params or [])

    manifest = build(builder, demo_inputs, tmp_path / "out")
    bundle = load.read_demo_bundle(tmp_path / "out")
    generation, booster, loaded = final.load_serving_model(demo_inputs.models)
    monkeypatch.setattr(
        predict, "latest_forecast", lambda engine: (FORECAST_BASE, service_weather())
    )
    monkeypatch.setattr(predict, "latest_stations", lambda engine, max_age, now: SERVICE_TARGETS)
    engine = RecordingEngine()
    predicted_at = datetime.fromisoformat(manifest["predicted_at"])
    version, count = predict.predict_and_store(engine, booster, loaded, generation, predicted_at)
    assert version == manifest["model_version"]
    assert count == len(bundle.predictions) == len(engine.records)
    service = {
        (r["station_id"], r["hour_start"].isoformat(), r["model_version"]): r["predicted_rentals"]
        for r in engine.records
    }
    demo = {
        (r["station_id"], r["hour_start"], r["model_version"]): r["predicted_rentals"]
        for r in bundle.predictions
    }
    assert demo == service


# 묶음과 DB의 같은 버전 예측 대조: 겹치는 (대여소, 시간) 수, 한쪽에만 있는 수, 겹친 곳의 최대 절대
# 차. DB 시각은 다른 시간대 표기여도 같은 순간이면 같은 시간이다.
def test_compare_predictions_reports_overlap_and_largest_difference(builder):
    hour = datetime(2026, 9, 19, 18, tzinfo=KST)
    bundle_rows = [
        {"station_id": "ST-1", "hour_start": hour.isoformat(), "predicted_rentals": 1.0},
        {"station_id": "ST-2", "hour_start": hour.isoformat(), "predicted_rentals": 2.0},
    ]
    stored = [("ST-1", hour.astimezone(UTC), 1.25), ("ST-3", hour, 3.0)]
    assert builder.compare_predictions(bundle_rows, stored) == {
        "bundle_rows": 2,
        "stored_rows": 2,
        "matched": 1,
        "only_in_bundle": 1,
        "only_in_database": 1,
        "max_abs_diff": 0.25,
    }
    assert builder.compare_predictions(bundle_rows, [])["max_abs_diff"] is None


# DB에 접속할 수 없으면 대조를 건너뛰고 이유만 남긴다(묶음은 DB 없이 만든다).
def test_database_comparison_is_skipped_when_the_database_is_down(builder):
    broken = create_engine("postgresql+psycopg://x:x@127.0.0.1:1/none?connect_timeout=1")
    try:
        report = builder.database_comparison(VERSION, [], engine=broken)
    finally:
        broken.dispose()
    assert report["skipped"].startswith("database unavailable")


# 기본 데모 시각은 스냅샷 수집 시각을 다음 분으로 올린 값이다(스냅샷이 '지금'보다 뒤에 오지 않게).
@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        ("2026-09-19T18:10:02.500000+09:00", "2026-09-19T18:11:00+09:00"),
        ("2026-09-19T18:10:00+09:00", "2026-09-19T18:10:00+09:00"),
        ("2026-09-19T09:10:00.000001+00:00", "2026-09-19T18:11:00+09:00"),
    ],
)
def test_default_demo_now_is_the_next_whole_minute_after_the_snapshot(builder, snapshot, expected):
    assert builder.default_demo_now(datetime.fromisoformat(snapshot)).isoformat() == expected


# ---------------------------------------------------------------------------------------------
# 빈 DB 적재와 데모 시각 화면(임시 PostgreSQL이 있어야 돈다)
# ---------------------------------------------------------------------------------------------
TABLES = (Station, RealtimeSnapshot, WeatherForecast, Prediction)


def table_counts(engine):
    with engine.connect() as conn:
        return {
            model.__tablename__: conn.execute(select(func.count()).select_from(model)).scalar_one()
            for model in TABLES
        }


# 빈 DB에 손으로 만든 묶음을 넣으면: 과거 이력 대여소 3곳 + 실시간에만 있는 2곳(ST-D, 자전거 수가
# '-'인 ST-E), 스냅샷 4행(ST-E 행은 건너뜀), 예보 2행, 예측 18행. 실제 시계로는 스냅샷이 오래돼
# /shortage-risk가 503이고, 데모 시각 18:15로는 [18:10, 21:10) 구간이 ST-A 4대 × 3시간 = 12대(자전거
# 1대 → 부족 11), ST-C 0.5 × 3 = 1.5대(자전거 0대 → 부족 1.5) 순이다. ST-B는 부족이 없고 ST-D는
# 예측이 없어 빠진다. /health와 대여소 상세도 데모 시각 기준이다.
def test_demo_bundle_loads_into_an_empty_database_and_replays_the_fixed_time(
    pg_engine, tmp_path, monkeypatch
):
    directory = tmp_path / "bundle"
    info = {"demo_now": DEMO_NOW_TEXT, "model_version": VERSION}
    load.write_demo_bundle(directory, bundle_tables(), info)
    result = load.load_demo(pg_engine, directory)
    assert result == load.DemoLoadResult(
        stations=3,
        realtime_stations=2,
        snapshots=4,
        skipped_rows=1,
        forecasts=2,
        predictions=18,
        demo_now=DEMO_NOW_TEXT,
        model_version=VERSION,
    )
    assert table_counts(pg_engine) == {
        "stations": 5,
        "realtime_snapshots": 4,
        "weather_forecasts": 2,
        "predictions": 18,
    }
    with pg_engine.connect() as conn:
        sources = dict(conn.execute(select(Station.station_id, Station.source)).all())
    assert sources == {"ST-A": "history", "ST-B": "history", "ST-C": "history",
                       "ST-D": "realtime", "ST-E": "realtime"}  # fmt: skip

    main.app.dependency_overrides[main.get_engine] = lambda: pg_engine
    try:
        client = TestClient(main.app)
        monkeypatch.delenv(main.DEMO_NOW_ENV, raising=False)
        stale = client.get("/shortage-risk")
        assert stale.status_code == 503 and stale.json() == {"detail": "no recent snapshot"}

        monkeypatch.setenv(main.DEMO_NOW_ENV, DEMO_NOW_TEXT)
        body = client.get("/shortage-risk", params={"hours": 3}).json()
        assert body["generated_at"] == DEMO_NOW_TEXT and body["model_version"] == VERSION
        ranking = [
            (s["station_id"], s["expected_rentals"], s["shortfall"]) for s in body["stations"]
        ]
        assert ranking == [("ST-A", 12.0, 11.0), ("ST-C", 1.5, 1.5)]
        health = client.get("/health").json()
        assert health["demo_now"] == DEMO_NOW_TEXT
        assert health["latest_snapshot_at"] == "2026-09-19T18:10:00+09:00"
        assert health["latest_prediction_hour"] == "2026-09-19T23:00:00+09:00"
        detail = client.get("/stations/ST-A").json()
        assert [p["hour_start"][11:16] for p in detail["predictions"]] == [
            "18:00", "19:00", "20:00", "21:00", "22:00", "23:00",
        ]  # fmt: skip
    finally:
        main.app.dependency_overrides.clear()


# 네 테이블 중 하나라도 행이 있으면 아무것도 넣지 않고 거부한다(개발 DB에 섞이지 않게). 한 번 넣은
# DB에 같은 묶음을 다시 넣는 것도 거부된다.
def test_demo_load_refuses_a_database_that_is_not_empty(pg_engine, tmp_path):
    directory = tmp_path / "bundle"
    load.write_demo_bundle(directory, bundle_tables(), {})
    with pg_engine.begin() as conn:
        conn.execute(Station.__table__.insert(), [{"station_id": "ST-X", "source": "history"}])
    with pytest.raises(load.DemoLoadRefused, match="stations"):
        load.load_demo(pg_engine, directory)
    assert table_counts(pg_engine) == {
        "stations": 1,
        "realtime_snapshots": 0,
        "weather_forecasts": 0,
        "predictions": 0,
    }
    with pg_engine.begin() as conn:
        conn.execute(text("delete from stations"))
    load.load_demo(pg_engine, directory)
    with pytest.raises(load.DemoLoadRefused, match="predictions"):
        load.load_demo(pg_engine, directory)


# 실제 서비스 경로와의 대조: 묶음의 원본 형식 파일을 서비스 적재 함수로 DB에 넣고
# predict.predict_and_store를 예측 시각으로 돌리면, 저장된 예측이 묶음의 예측과 키·값 모두 같다.
# 묶음 만들기의 DB 대조(database_comparison)도 이 DB에서 모두 겹치고 최대 차 0이라고 보고한다.
def test_bundle_predictions_equal_what_the_service_stores(
    pg_engine, builder, demo_inputs, tmp_path
):
    out = tmp_path / "out"
    manifest = build(builder, demo_inputs, out)
    load.load_stations_from_warehouse(pg_engine, demo_inputs.warehouse)
    load.load_realtime_file(pg_engine, out / "snapshots.parquet")
    load.load_forecasts(pg_engine, out / "forecasts.parquet")
    generation, booster, loaded = final.load_serving_model(demo_inputs.models)
    predicted_at = datetime.fromisoformat(manifest["predicted_at"])
    version, count = predict.predict_and_store(pg_engine, booster, loaded, generation, predicted_at)
    assert version == manifest["model_version"]
    with pg_engine.connect() as conn:
        stored = conn.execute(
            select(Prediction.station_id, Prediction.hour_start, Prediction.predicted_rentals)
        ).all()
    bundle = load.read_demo_bundle(out)
    assert count == len(bundle.predictions)
    assert {(s, h.isoformat()): v for s, h, v in stored} == {
        (r["station_id"], r["hour_start"]): r["predicted_rentals"] for r in bundle.predictions
    }
    report = builder.database_comparison(version, bundle.predictions, engine=pg_engine)
    assert report == {
        "bundle_rows": count,
        "stored_rows": count,
        "matched": count,
        "only_in_bundle": 0,
        "only_in_database": 0,
        "max_abs_diff": 0.0,
    }
