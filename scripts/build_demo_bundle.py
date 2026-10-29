"""고정 시각 데모 묶음을 만든다(읽기 전용: 원본·warehouse·서비스 세대는 읽기만 한다).

닫힌 서비스의 한 시각 화면을 인증키·Airflow·서비스 모델 없이 서비스 DB 하나로 다시 보여 주려고,
그 시각에 서비스가 갖고 있던 자료를 작은 묶음 폴더로 쓴다(형식은 bike_demand.serving.load의 데모
묶음).

- stations.parquet: warehouse dim_stations 전체(서비스 stations 표의 과거 이력 대여소와 같은 행)
- snapshots.parquet: 실시간 스냅샷 원본 JSON 하나를 수집기(ingest.realtime.to_parquet)로 바꾼 행
- forecasts.parquet: 단기예보 발표 원본 JSON 하나를 수집기(ingest.forecast.to_parquet)로 바꾼 행
- predictions.parquet: 위 자료와 CURRENT 서비스 세대로 서비스 예측(model.predict)과 같은 계산을
  한 값
- manifest.json: 파일·원천 SHA-256, 세대 이름·파일 지문, 모델 버전, 데모 시각, 코드 커밋·패키지 버전

실행 (저장소 루트에서, 콘솔이 cp949면 PYTHONUTF8=1):
  uv run python scripts/build_demo_bundle.py  # 18:00 스냅샷 → data/demo/2026-09-19T1800/
  uv run python scripts/build_demo_bundle.py --snapshot <원본 JSON> --forecast <원본 JSON>
      --out <새 폴더> [--demo-now 2026-09-19T18:05:00+09:00] [--no-compare-db]
묶음을 쓰는 법: 빈 데이터베이스에 alembic upgrade head → python -m bike_demand.serving.load demo
--bundle <폴더> → BIKE_DEMAND_DEMO_NOW=<manifest의 demo_now>로 API를 띄운다(docs/api.md
'데모 시계').
"""

# [파일 역할] 서비스가 멈춘 뒤에도 README 화면 시각(2026-09-19 18시대)의 서비스 상태를 다시 볼 수
# 있게, 디스크에 남은 원본 두 개와 서비스 세대로 데모 묶음을 만든다. 실시간 API는 지금 상태만
# 주므로 지난 시각의 스냅샷은 다시 받을 수 없고, 작은 묶음으로 남기는 것이 그 화면을 보존하는
# 방법이다.
# [왜 이렇게]
# - 스냅샷·예보는 수집기의 변환 함수로 바꾼다: 원본을 임시 폴더의 수집기와 같은 경로(raw_path)로
#   복사해 to_parquet을 그대로 부른다. 변환 규칙을 여기 다시 적으면 수집기와 어긋날 수 있다.
# - 예측은 predict.predict_and_store와 같은 단계로 계산한다: 예측 시각이 든 정시부터 48시간 중
#   예보가 있는 시간, feature_rows, 모델의 feature_name() 순서, 버전 이름 '세대-fcst발표시각'.
#   같은지는 tests/test_demo.py가 predict_and_store와 직접 비교해 확인한다.
# - 예측 대상은 이 묶음을 서비스 DB에 넣었을 때 predict.latest_stations가 고를 대여소다: 스냅샷에
#   자전거 수가 있는 대여소(station_id 순), 정적 정보는 stations 표의 규칙(과거 이력 대여소는
#   warehouse 값, 실시간에만 있는 대여소는 스냅샷 값, 자치구 없음).
# - 예측 시각 = 예보를 받은 시각: 서비스는 예보 발표를 받자마자 예측했다(forecast_and_predict DAG).
# - 출력은 임시 폴더에 다 쓴 뒤 이름을 바꿔 한 번에 내놓는다. 중간에 실패해도 반쯤 쓴 묶음이 남지
#   않고, 이미 있는 폴더는 덮어쓰지 않는다.
# [주의] 실제 서비스는 예보를 받은 때(17:15)의 최근 스냅샷으로 대상 대여소를 정했고, 이 묶음은
# 묶음의 스냅샷(기본 18:00)으로 정한다. 두 스냅샷의 대여소 구성이 다르면 학습에 없던 대여소의
# 번호(목록 순서로 매김)와 그 예측이 실제 저장값과 다를 수 있다. 실시간에만 있는 대여소의
# 거치대·좌표도 실제 DB에는 처음 본 스냅샷 값이 남아 있었다. 그래서 서비스 DB가 켜져 있으면 저장된
# 같은 버전 예측과 대조해 manifest에 남긴다(--no-compare-db로 끔).
# [연결] bike_demand.serving.load(묶음 형식·데모 적재), bike_demand.model.predict·final·provenance,
# bike_demand.ingest.realtime·forecast, bike_demand.api.main(데모 시계 값 검사).
# 테스트: tests/test_demo.py.

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# lightgbm: CURRENT 세대의 모델 파일을 읽는다. pyarrow.parquet: 수집기가 쓴 임시 bronze를 다시
# 읽는다.
# sqlalchemy: 서비스 DB 예측과의 대조(선택).
import lightgbm as lgb
import pyarrow.parquet as pq
from sqlalchemy import Engine, select
from sqlalchemy.exc import OperationalError

from bike_demand.api.main import DemoClockError, parse_demo_now
from bike_demand.ingest import forecast as forecast_ingest
from bike_demand.ingest import realtime as realtime_ingest
from bike_demand.model import artifacts, final, predict, provenance
from bike_demand.model.forecast_weather import hourly_weather
from bike_demand.serving import load
from bike_demand.serving.db import make_engine
from bike_demand.serving.models import Prediction

KST = timezone(timedelta(hours=9))
# 코드 상태(git)를 읽을 저장소 루트. 현재 폴더가 아니라 이 파일 위치로 정해 어디서 실행해도 같다.
REPO_ROOT = Path(__file__).resolve().parents[1]

# 기본 입력: README 화면(docs/images/dashboard.png, 마지막 수집 시각 09. 19. 18:00:01)의 실제 원본.
# 스냅샷 2026-09-19 18:00:01과 그때 최신이던 17:00 발표 예보(17:15에 받음)다. 이 기본값으로 만든
# 묶음의 3시간 부족 순위 상위 5곳(대여소·자전거·예상·부족)이 화면의 표와 같다. 18:10:02 스냅샷은
# 화면과 다르다. 모두 커밋하지 않는 data/ 아래에 있다.
DEFAULT_SNAPSHOT = Path("data/raw/realtime/bikelist/date=2026-09-19/180001.json")
DEFAULT_FORECAST = Path("data/raw/weather/vilage_fcst/nx=60_ny=127/base=202609191700.json")
DEFAULT_WAREHOUSE = Path("data/warehouse/bike_demand.duckdb")
DEFAULT_MODELS = Path("data/models/v1")
DEFAULT_OUT = Path("data/demo/2026-09-19T1800")

# 예측 시간 수. Airflow forecast_and_predict DAG가 부르는 `model.predict --hours 48`, 그리고
# predict_and_store의 기본값과 같다.
HORIZON_HOURS = 48
# 데모 시각은 스냅샷 시각부터 이 시간 안이어야 한다. /shortage-risk와 대시보드의 스냅샷 나이
# 기본값이 30분이라, 더 늦은 시각이면 데모 화면이 503(no recent snapshot)이 된다.
DEMO_WINDOW = timedelta(minutes=30)
# manifest에 그대로 옮기는 출처 표기(README '데이터 출처'와 같은 문장).
ATTRIBUTION = [
    "서울특별시 공공자전거 대여이력 정보, 대여소 정보, 실시간 대여정보: 서울 열린데이터광장 "
    "(공공누리 제1유형: 출처표시)",
    "종관기상관측(ASOS) 자료, 단기예보: 기상청",
]


# 기본 데모 시각: 스냅샷 수집 시각(KST)을 다음 분으로 올린 값(이미 정각 분이면 그대로). 스냅샷이
# '지금' 보다 뒤에 오지 않게 하면서 화면의 생성 시각을 보기 좋은 분 단위로 맞춘다.
def default_demo_now(fetched_at: datetime) -> datetime:
    local = fetched_at.astimezone(KST)
    floor = local.replace(second=0, microsecond=0)
    return floor if floor == local else floor + timedelta(minutes=1)


# 원천 파일 하나의 기록: 원본 폴더 규칙 안의 이름(부모 폴더/파일, 예 'date=2026-09-19/180001.json'),
# SHA-256, 바이트 수. 절대 경로는 PC마다 달라 남기지 않는다.
def file_source(path: Path) -> dict:
    return {
        "file": f"{path.parent.name}/{path.name}",
        "sha256": provenance.sha256_file(path),
        "bytes": path.stat().st_size,
    }


# 스냅샷 원본 JSON 하나 → 수집기 bronze 행(모든 값 문자열). 원본을 scratch 아래 수집기 경로
# (date=YYYY-MM-DD/HHMMSS.json)로 복사하고 ingest.realtime.to_parquet으로 그 날짜 폴더를 바꾼다.
# 원본 파일은 읽기만 한다.
def snapshot_bronze(raw: Path, scratch: Path) -> list[dict]:
    payload = json.loads(raw.read_text(encoding="utf-8"))
    fetched_at = datetime.fromisoformat(payload["fetched_at"])
    raw_dir = scratch / "realtime"
    copy = realtime_ingest.raw_path(raw_dir, fetched_at)
    copy.parent.mkdir(parents=True)
    shutil.copyfile(raw, copy)
    out = scratch / "realtime.parquet"
    realtime_ingest.to_parquet(raw_dir, out, fetched_at.astimezone(KST).date())
    return pq.read_table(out).to_pylist()


# 예보 발표 원본 JSON 하나 → 수집기 bronze 행. 원본을 scratch 아래 수집기 경로
# (nx=..._ny=.../base=YYYYMMDDHHMM.json, raw_path가 발표 시각·격자 형식도 검사)로 복사하고
# ingest.forecast.to_parquet으로 바꾼다(항목마다 발표·격자 검사, fetched_at은 KST 문자열로).
def forecast_bronze(raw: Path, scratch: Path) -> list[dict]:
    payload = json.loads(raw.read_text(encoding="utf-8"))
    base = datetime.fromisoformat(payload["base_datetime"])
    raw_dir = scratch / "forecast"
    copy = forecast_ingest.raw_path(raw_dir, base, int(payload["nx"]), int(payload["ny"]))
    copy.parent.mkdir(parents=True)
    shutil.copyfile(raw, copy)
    out = scratch / "forecast.parquet"
    forecast_ingest.to_parquet(raw_dir, out)
    return pq.read_table(out).to_pylist()


# 이 묶음을 서비스 DB에 넣었을 때 predict.latest_stations가 돌려줄 예측 대상(station_id 순).
# stations
# 표의 규칙: 과거 이력 대여소는 warehouse 값이고, 실시간에만 있는 대여소는 스냅샷에서 등록한 값이다
# (적재가 이미 있는 대여소를 실시간 값으로 덮어쓰지 않음). 스냅샷에서 건너뛰는 행(자전거 수를 넣을
# 수 없음)의 대여소는 대상이 아니다.
def prediction_targets(history: list[dict], realtime: load.RealtimeRows) -> list[dict]:
    info = {row["station_id"]: row for row in history}
    for station in realtime.stations:
        info.setdefault(station["station_id"], station)
    in_snapshot = sorted({row["station_id"] for row in realtime.snapshots})
    return [
        {
            key: info[station_id].get(key)
            for key in ("station_id", "district", "docks", "lat", "lon")
        }
        for station_id in in_snapshot
    ]


# 서비스 예측과 같은 계산: 예측 시각이 든 정시부터 HORIZON_HOURS개 중 예보가 덮는 시간 × 대상
# 대여소의 특징(모델 feature_name() 순서) → 예측. 행은 bronze처럼 hour_start를 오프셋 붙은 ISO
# 문자열로 둔다.
# 반환: (model_version, 예측 행 목록).
def predict_rows(
    targets: list[dict],
    weather: dict[datetime, dict[str, float]],
    base: datetime,
    predicted_at: datetime,
    generation: str,
    booster: lgb.Booster,
    loaded: artifacts.Artifacts,
) -> tuple[str, list[dict]]:
    first = predicted_at.astimezone(KST).replace(minute=0, second=0, microsecond=0)
    candidates = [first + timedelta(hours=i) for i in range(HORIZON_HOURS)]
    hours = [hour for hour in candidates if hour in weather]
    if not hours:
        raise SystemExit("예측할 시간의 예보가 없음(예보가 예측 시각 뒤 48시간을 덮지 않음)")
    matrix, keys = predict.feature_rows(targets, hours, weather, loaded, booster.feature_name())
    values = booster.predict(matrix)
    version = f"{generation}-fcst{base.astimezone(KST):%Y%m%d%H%M}"
    rows = [
        {
            "station_id": station_id,
            "hour_start": hour.isoformat(),
            "model_version": version,
            "predicted_rentals": float(value),
        }
        for (station_id, hour), value in zip(keys, values, strict=True)
    ]
    return version, rows


# 묶음 예측과 DB에 저장된 같은 버전 예측의 대조. 키는 (대여소, 시간 순간)이라 DB 시각이 다른 시간대
# 표기여도 같은 순간이면 같다. 반환: 행 수, 겹친 수, 한쪽에만 있는 수, 겹친 곳의 최대 절대 차(겹친
# 곳이 없으면 None). stored는 (station_id, hour_start, predicted_rentals) 튜플 목록.
def compare_predictions(bundle_rows: list[dict], stored) -> dict:
    bundle = {
        (r["station_id"], datetime.fromisoformat(r["hour_start"])): r["predicted_rentals"]
        for r in bundle_rows
    }
    database = {(station_id, hour): value for station_id, hour, value in stored}
    matched = bundle.keys() & database.keys()
    diffs = [abs(bundle[key] - database[key]) for key in matched]
    return {
        "bundle_rows": len(bundle_rows),
        "stored_rows": len(stored),
        "matched": len(matched),
        "only_in_bundle": len(bundle.keys() - database.keys()),
        "only_in_database": len(database.keys() - bundle.keys()),
        "max_abs_diff": max(diffs) if diffs else None,
    }


# 서비스 DB(engine, 없으면 DATABASE_URL 또는 로컬 개발 DB)의 같은 버전 예측과 대조한다. DB에 접속할
# 수 없으면 대조를 건너뛰고 이유만 남긴다(묶음은 DB 없이 만든다). 테이블이 없는 DB 같은 그 밖의
# 오류는 그대로 올린다(조용히 넘기지 않음). 읽기만 한다.
def database_comparison(
    version: str, bundle_rows: list[dict], engine: Engine | None = None
) -> dict:
    owned = engine is None
    engine = make_engine() if owned else engine
    try:
        with engine.connect() as conn:
            stored = conn.execute(
                select(
                    Prediction.station_id, Prediction.hour_start, Prediction.predicted_rentals
                ).where(Prediction.model_version == version)
            ).all()
    except OperationalError as exc:
        reason = type(exc.orig).__name__ if exc.orig is not None else type(exc).__name__
        return {"skipped": f"database unavailable ({reason})"}
    finally:
        if owned:
            engine.dispose()
    return compare_predictions(bundle_rows, stored)


# 임시 작업 폴더를 지운다. 지우지 못하면(다른 프로그램이 파일을 잡고 있는 등) 조용히 넘기지 않고
# 남은 경로를 표준 오류로 알린다. 묶음 결과에는 영향이 없다.
def _remove(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"임시 폴더를 지우지 못함(손으로 지워도 됨): {path} ({exc})", file=sys.stderr)


# 묶음 하나를 만든다. 순서: 출력 폴더가 없는지 확인 → 같은 부모 폴더의 임시 폴더에서 원본 두 개를
# 수집기 변환으로 바꾸고 시각·격자 검사 → warehouse 대여소·CURRENT 세대로 예측 → (선택) DB 대조 →
# 묶음 쓰기 → 이름 바꾸기. 반환: 쓴 manifest 사전.
# 거부(SystemExit): 출력 폴더가 이미 있음, 스냅샷에 넣을 행이 없음, 서울 격자가 아닌 예보,
# 스냅샷보다 뒤에 받은 예보(그 시각의 서비스가 갖고 있지 않던 예보), 스냅샷 시각부터 30분 밖의 데모
# 시각.
def build(
    *,
    snapshot: Path,
    forecast: Path,
    warehouse: Path,
    models: Path,
    out: Path,
    demo_now: datetime | None = None,
    compare_db: bool = True,
    engine: Engine | None = None,
) -> dict:
    """원본 두 개·warehouse·CURRENT 세대 → out 폴더의 데모 묶음. 반환은 manifest."""
    snapshot, forecast, out = Path(snapshot), Path(forecast), Path(out)
    if out.exists():
        raise SystemExit(f"출력 폴더가 이미 있음(덮어쓰지 않음): {out}")
    if demo_now is not None and demo_now.utcoffset() is None:
        # 시간대 없는 시각을 PC 시간대로 짐작하지 않는다(api.main.parse_demo_now와 같은 규칙)
        raise SystemExit("데모 시각에 시간대가 없음")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", suffix=".partial", dir=out.parent))
    try:
        scratch = staging / "bronze"
        snapshots = snapshot_bronze(snapshot, scratch)
        forecasts = forecast_bronze(forecast, scratch)
        if not snapshots:
            raise SystemExit(f"스냅샷 원본에 행이 없음: {snapshot}")
        if not forecasts:
            raise SystemExit(f"예보 원본에 항목이 없음: {forecast}")
        snapshot_at = datetime.fromisoformat(snapshots[0]["fetched_at"])
        base = datetime.fromisoformat(forecasts[0]["base_datetime"])
        predicted_at = datetime.fromisoformat(forecasts[0]["fetched_at"])
        grid = (int(forecasts[0]["nx"]), int(forecasts[0]["ny"]))
        if grid != (forecast_ingest.SEOUL_NX, forecast_ingest.SEOUL_NY):
            raise SystemExit(f"서비스가 쓰는 서울 격자(60·127)의 예보가 아님: {grid}")
        if predicted_at > snapshot_at:
            raise SystemExit(
                f"예보를 스냅샷보다 뒤에 받음({predicted_at.isoformat()} > "
                f"{snapshot_at.isoformat()}): 그 시각의 서비스에는 없던 예보라 묶음을 만들지 않는다"
            )
        demo_at = default_demo_now(snapshot_at) if demo_now is None else demo_now.astimezone(KST)
        if not snapshot_at <= demo_at < snapshot_at + DEMO_WINDOW:
            raise SystemExit(
                f"데모 시각 {demo_at.isoformat()}은 스냅샷 시각 {snapshot_at.isoformat()}부터 30분 "
                "안이어야 함(스냅샷 나이 기본값 30분)"
            )

        history = load.read_warehouse_stations(Path(warehouse))
        realtime = load.realtime_rows(snapshots)
        targets = prediction_targets(history, realtime)
        if not targets:
            raise SystemExit(f"스냅샷에 자전거 수를 넣을 수 있는 행이 없음: {snapshot}")
        # CURRENT를 한 번만 읽어 고른 세대 폴더 하나에서 모델·산출물·지문을 모두 읽는다
        # (final.load_serving_model과 같은 파일, 읽는 사이 CURRENT가 바뀌어도 섞이지 않음)
        generation, directory = final.current_generation(Path(models) / "serving")
        booster = lgb.Booster(model_file=str(directory / "model.txt"))
        loaded = artifacts.load(directory / "artifacts")
        weather = hourly_weather(
            (datetime.fromisoformat(r["fcst_datetime"]), r["category"], r["value"])
            for r in forecasts
        )
        version, predictions = predict_rows(
            targets, weather, base, predicted_at, generation, booster, loaded
        )
        comparison = (
            database_comparison(version, predictions, engine)
            if compare_db
            else {"skipped": "not requested"}
        )

        code = provenance.git_state(REPO_ROOT)
        code.pop("root", None)  # PC마다 다른 절대 경로는 남기지 않는다
        info = {
            "demo_now": demo_at.isoformat(),
            "snapshot_fetched_at": snapshots[0]["fetched_at"],
            "forecast_base": base.isoformat(),
            "predicted_at": forecasts[0]["fetched_at"],
            "generation": generation,
            "model_version": version,
            "horizon_hours": HORIZON_HOURS,
            "counts": {
                "prediction_targets": len(targets),
                "realtime_only_stations": len(
                    {s["station_id"] for s in realtime.stations}
                    - {s["station_id"] for s in history}
                ),
                "snapshot_rows_skipped_on_load": realtime.skipped_rows,
            },
            "sources": {
                "snapshot": file_source(snapshot),
                "forecast": file_source(forecast),
                "warehouse": {"dim_stations_rows": len(history)},
                "generation": {"name": generation, "files": provenance.file_digests(directory)},
            },
            "database_comparison": comparison,
            "code": code,
            "packages": provenance.package_versions(),
            "created_at": datetime.now(KST).isoformat(timespec="seconds"),
            "attribution": ATTRIBUTION,
        }
        tables = {
            "stations": history,
            "snapshots": snapshots,
            "forecasts": forecasts,
            "predictions": predictions,
        }
        manifest = load.write_demo_bundle(staging / "bundle", tables, info)
        os.replace(staging / "bundle", out)
    finally:
        _remove(staging)
    return manifest


# 명령줄: 기본값은 README 화면 시각의 원본 → data/demo/2026-09-19T1800/. 끝나면 묶음 폴더와 주요 값,
# 다음에 칠 명령을 표준 출력에 낸다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="실시간 스냅샷 원본 하나·그때의 예보 발표 하나·warehouse 대여소·CURRENT 서비스 "
        "세대로 고정 시각 데모 묶음(Parquet 네 개 + manifest.json)을 만든다. 입력은 읽기만 한다",
        epilog="묶음은 데이터가 하나도 없는 서비스 DB에만 넣는다: DATABASE_URL을 빈 데이터베이스로 "
        "두고 alembic upgrade head → python -m bike_demand.serving.load demo --bundle <폴더> → "
        "BIKE_DEMAND_DEMO_NOW=<demo_now> uvicorn bike_demand.api.main:app",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="실시간 스냅샷 원본 JSON 하나(수집기가 쓴 date=YYYY-MM-DD/HHMMSS.json)",
    )
    parser.add_argument(
        "--forecast",
        type=Path,
        default=DEFAULT_FORECAST,
        help="단기예보 발표 원본 JSON 하나(nx=60_ny=127/base=YYYYMMDDHHMM.json). 스냅샷보다 "
        "먼저 받은 발표여야 하고, 그 시각에 최신이던 발표를 고르는 것은 부르는 쪽 몫이다",
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=DEFAULT_WAREHOUSE,
        help="dim_stations를 읽을 warehouse. 읽기 전용으로 열지만 dbt가 쓰는 중에는 돌리지 않는다",
    )
    parser.add_argument(
        "--models", type=Path, default=DEFAULT_MODELS, help="serving/CURRENT가 있는 모델 폴더"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="새로 만들 묶음 폴더(이미 있으면 멈춤). 기본은 커밋하지 않는 data/ 아래",
    )
    parser.add_argument(
        "--demo-now",
        help="데모 시각(시간대가 있는 ISO 8601, 스냅샷 시각부터 30분 안). 기본은 스냅샷 수집 "
        "시각을 다음 분으로 올린 값",
    )
    parser.add_argument(
        "--no-compare-db",
        action="store_true",
        help="서비스 DB(DATABASE_URL, 없으면 로컬 개발 DB)에 저장된 같은 버전 예측과 대조하지 "
        "않는다. 기본은 대조하고, 접속할 수 없으면 건너뛴 이유만 manifest에 남긴다",
    )
    args = parser.parse_args(argv)
    try:
        demo_now = parse_demo_now(args.demo_now)
    except DemoClockError as exc:
        raise SystemExit(str(exc)) from None
    try:
        manifest = build(
            snapshot=args.snapshot,
            forecast=args.forecast,
            warehouse=args.warehouse,
            models=args.models,
            out=args.out,
            demo_now=demo_now,
            compare_db=not args.no_compare_db,
        )
    except forecast_ingest.ApiError as exc:
        # 예보 원본의 형식 오류(수집기의 짧은 메시지, 원본 내용은 담지 않음). 그 밖의 오류는 추적
        # 정보와 함께 그대로 올린다
        raise SystemExit(f"예보 원본을 묶음으로 바꾸지 못함: {exc}") from None
    summary = {
        "out": str(args.out),
        "demo_now": manifest["demo_now"],
        "model_version": manifest["model_version"],
        "rows": {name: entry["rows"] for name, entry in manifest["files"].items()},
        "bytes": sum(entry["bytes"] for entry in manifest["files"].values()),
        "database_comparison": manifest["database_comparison"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"다음: 빈 DB에 alembic upgrade head → python -m bike_demand.serving.load demo --bundle "
        f"{args.out} → BIKE_DEMAND_DEMO_NOW={manifest['demo_now']} uvicorn bike_demand.api.main:app"
    )


if __name__ == "__main__":
    main()
