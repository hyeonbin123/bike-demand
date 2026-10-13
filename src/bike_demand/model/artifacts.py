"""서비스 예측에 필요한 학습 시점 값을 모델 옆에 저장하고 읽는다.

학습 때 대여소ID·자치구를 숫자 코드로 바꾼 규칙, 대여소 패턴(대여소×쉬는 날 여부×시간 평균),
추세는 학습 기간(history)으로 계산한 값이어야 한다. 서비스에서 다시 계산하면 코드가 어긋나거나
학습 뒤의 데이터가 섞이므로, 학습할 때 `frames.history_ctes`와 같은 정의로 파일에 남긴다.

폴더 구성: stations.parquet, profile.parquet, trend.parquet, meta.json
"""

# [파일 역할] 서비스 모델 세대 폴더 안의 artifacts/를 쓰고 읽는다. 서비스 예측(predict.py)은
# warehouse를 열지 않고 이 파일들만으로 학습 때와 같은 특징을 만든다.
# [왜 필요한가] 범주형 번호(station_code, district_code)는 학습 때 매긴 번호와 같아야 LightGBM
# 트리가 같은 대여소를 가리킨다. 대여소 패턴·추세·반기 수준도 학습 기간 정의 그대로여야 한다. 그래서
# 학습 직후 frames.py와 같은 SQL(CTE)로 계산해 파일로 고정한다.
# [연결] model/final.py가 서비스 학습 때 export를 부르고, load_serving_model이 load로 읽는다.
# predict.py, forecast_gap.py, shortage_eval.py가 Artifacts를 받아 쓴다.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# duckdb: warehouse에서 CTE 결과를 Parquet으로 바로 내보낸다(COPY ... TO). pyarrow.parquet: 서비스
# 쪽에서 그 파일을 읽는다.
import duckdb
import pyarrow.parquet as pq

# frames의 CTE 정의와 특징 목록을 그대로 가져와 학습과 같은 정의를 보장한다.
from bike_demand.model.frames import FEATURES, LEVEL_CTES, LEVEL_FEATURES, history_ctes


# history(학습 기간)로 계산한 값을 out_dir에 Parquet 파일들 + meta.json으로 저장하고 meta를
# 돌려준다.
# - stations: 대여소별 번호·자치구 번호·거치대·좌표(최신 dim_stations 기준)
# - profile: 대여소×쉬는 날 여부×시각 평균(profile_mean)
# - trend: 대여소 추세, meta의 global_trend: 전체 추세
# - (with_levels) halves·system_halves: 모든 반기의 대여소 평균과 전체 합. 예측 시점마다 그때 공개된
#   반기를 골라 쓰기 위해 반기 전체를 저장한다
# - meta.json: 학습 기간, 특징 순서, 수준 특징 여부
def export(
    con: duckdb.DuckDBPyConnection,
    history: tuple[str, str],
    out_dir: Path,
    with_levels: bool = False,
) -> dict:
    """with_levels: v2 수준 특징에 쓰는 반기별 대여소 평균·전체 합계도 저장한다(모든 반기)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ctes = history_ctes(history)
    queries = {
        "stations": """select c.station_id, c.station_code, c.district, dc.district_code,
                              c.docks, c.lat, c.lon
                       from codes as c left join district_codes as dc using (district)""",
        "profile": "select station_id, is_offday, hour_of_day, profile_mean from profile",
        "trend": "select station_id, station_trend from trend",
    }
    if with_levels:
        ctes = LEVEL_CTES + ctes
        queries["halves"] = "select station_id, half, half_mean from halves"
        queries["system_halves"] = "select half, total from system_halves"
    # DuckDB COPY로 쿼리 결과를 파이썬을 거치지 않고 파일로 쓴다. 경로는 DuckDB 문자열에 맞게 슬래시
    # 형식(as_posix)으로 넣는다.
    for name, query in queries.items():
        path = (out_dir / f"{name}.parquet").as_posix()
        con.execute(f"copy (with {ctes} {query}) to '{path}' (format parquet)")
    global_trend = con.execute(f"with {ctes} select global_trend from global_trend").fetchone()[0]
    meta = {
        "history": list(history),
        "global_trend": global_trend,
        "features": [*FEATURES, *LEVEL_FEATURES] if with_levels else FEATURES,
        "with_levels": with_levels,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return meta


# load가 돌려주는, 예측에 바로 쓰기 좋은 사전 묶음. profile 키는 (대여소, 쉬는 날 여부, 시각).
@dataclass
class Artifacts:
    # station_id -> station_code, district, district_code, docks, lat, lon
    stations: dict[str, dict]
    district_codes: dict[str, int]
    profile: dict[tuple[str, bool, int], float]
    trend: dict[str, float | None]
    global_trend: float | None
    # 학습 때 가장 큰 대여소 번호. 학습 뒤에 생긴 대여소는 이보다 큰 번호를 받아 '학습에 없던
    # 범주'로 들어간다(predict.py, docs/experiments.md v1 측정 전 보충).
    max_station_code: int
    # v2 수준 특징용. 없으면 빈 dict(해당 특징은 결측)
    halves: dict[tuple[str, int], float] = field(default_factory=dict)
    system_halves: dict[int, float] = field(default_factory=dict)
    # 이 산출물을 만든 자료의 끝(history의 끝, 제외). 이 날까지 다 끝난 반기만 수준 특징에 쓴다
    history_end: date | None = None

    # 6 * (half + 1)은 그 반기 다음 첫 달의 달 번호(연*12 + 월-1)다. 그달 1일이 자료 끝(history_end,
    # 제외) 이하이면 반기 전체가 자료 안에 있다. 호출 위치: predict.py(수준 특징을 만들 때).
    def half_is_complete(self, half: int) -> bool:
        """반기 번호(frames.published_half와 같은 번호)의 기간이 자료 끝 전에 모두 끝났는가.

        자료의 마지막 날 뒤에 걸친 반기는 일부(또는 0건인 하루)만 집계돼 있을 수 있다.
        산출물을 갱신하지 않은 채 그 반기의 공개 시점을 넘기면 그 값을 최근 수준으로 읽게 되므로
        막는다(T53).
        """
        if self.history_end is None:
            return True
        first_month_after = 6 * (half + 1)
        return date(first_month_after // 12, first_month_after % 12 + 1, 1) <= self.history_end


# artifacts 폴더를 읽어 Artifacts로 만든다. 자치구 번호표는 stations 행에서 다시 만든다. 수준
# 파일(halves)이 없는 옛 산출물이면 빈 사전이라 수준 특징이 결측으로 들어간다. meta에 history가 없는
# 옛 산출물은 history_end를 None으로 두어 반기 완료 검사를 하지 않는다.
# 호출 위치: model/final.py load_serving_model.
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
    halves: dict[tuple[str, int], float] = {}
    system_halves: dict[int, float] = {}
    if (directory / "halves.parquet").exists():
        halves = {
            (r["station_id"], int(r["half"])): r["half_mean"]
            for r in pq.read_table(directory / "halves.parquet").to_pylist()
        }
        system_halves = {
            int(r["half"]): r["total"]
            for r in pq.read_table(directory / "system_halves.parquet").to_pylist()
        }
    return Artifacts(
        halves=halves,
        system_halves=system_halves,
        stations=stations,
        district_codes=district_codes,
        profile=profile,
        trend=trend,
        global_trend=meta["global_trend"],
        max_station_code=max(int(r["station_code"]) for r in stations.values()),
        history_end=date.fromisoformat(meta["history"][1]) if meta.get("history") else None,
    )
