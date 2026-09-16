"""서비스 예측에 필요한 학습 시점 값을 모델 옆에 저장하고 읽는다.

학습 때 대여소ID·자치구를 숫자 코드로 바꾼 규칙, 대여소 패턴(대여소×쉬는 날 여부×시간 평균),
추세는 학습 기간(history)으로 계산한 값이어야 한다. 서비스에서 다시 계산하면 코드가 어긋나거나
학습 뒤의 데이터가 섞이므로, 학습할 때 `frames.history_ctes`와 같은 정의로 파일에 남긴다.

폴더 구성: stations.parquet, profile.parquet, trend.parquet, meta.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from bike_demand.model.frames import FEATURES, history_ctes


def export(con: duckdb.DuckDBPyConnection, history: tuple[str, str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ctes = history_ctes(history)
    queries = {
        "stations": """select c.station_id, c.station_code, c.district, dc.district_code,
                              c.docks, c.lat, c.lon
                       from codes as c left join district_codes as dc using (district)""",
        "profile": "select station_id, is_offday, hour_of_day, profile_mean from profile",
        "trend": "select station_id, station_trend from trend",
    }
    for name, query in queries.items():
        path = (out_dir / f"{name}.parquet").as_posix()
        con.execute(f"copy (with {ctes} {query}) to '{path}' (format parquet)")
    global_trend = con.execute(f"with {ctes} select global_trend from global_trend").fetchone()[0]
    meta = {"history": list(history), "global_trend": global_trend, "features": FEATURES}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return meta


@dataclass
class Artifacts:
    # station_id -> station_code, district, district_code, docks, lat, lon
    stations: dict[str, dict]
    district_codes: dict[str, int]
    profile: dict[tuple[str, bool, int], float]
    trend: dict[str, float | None]
    global_trend: float | None
    max_station_code: int


def load(directory: Path) -> Artifacts:
    meta = json.loads((directory / "meta.json").read_text("utf-8"))
    station_rows = pq.read_table(directory / "stations.parquet").to_pylist()
    stations = {r["station_id"]: r for r in station_rows}
    profile = {
        (r["station_id"], bool(r["is_offday"]), int(r["hour_of_day"])): r["profile_mean"]
        for r in pq.read_table(directory / "profile.parquet").to_pylist()
    }
    trend = {
        r["station_id"]: r["station_trend"]
        for r in pq.read_table(directory / "trend.parquet").to_pylist()
    }
    district_codes = {
        r["district"]: int(r["district_code"])
        for r in stations.values()
        if r["district"] is not None and r["district_code"] is not None
    }
    return Artifacts(
        stations=stations,
        district_codes=district_codes,
        profile=profile,
        trend=trend,
        global_trend=meta["global_trend"],
        max_station_code=max(int(r["station_code"]) for r in stations.values()),
    )
