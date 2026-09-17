"""v4(T12) 측정: 단기예보 날씨와 ASOS 관측 날씨의 차이, 그 차이로 예측이 흔들리는 정도.

계획은 docs/experiments.md v4 절. 판정 규칙은 없고 결과만 보고한다.

- 계산(순수 함수): 관측 변환(`observed_hourly`), 간격별 날씨 오차(`weather_errors`),
  예보·관측 날씨로 각각 예측(`predict_both`), 흔들림 집계(`ShiftTotals`),
  부족 상위 목록(`top_shortage`)
- 실행(`__main__`): 서비스 DB의 예보 발표·스냅샷, 서비스 모델(CURRENT 세대),
  ASOS 관측 JSON을 읽는다.
  관측은 `--fetch-observed`로 한 번 받아 work/ 아래 JSON으로 두고 bronze는 건드리지 않는다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from bike_demand.api.shortage import expected_rentals
from bike_demand.model.artifacts import Artifacts
from bike_demand.model.frames import WEATHER_FEATURES
from bike_demand.model.predict import feature_rows

KST = timezone(timedelta(hours=9))
HOUR = timedelta(hours=1)
# (가장 짧은 간격, 가장 긴 간격, 이름). 간격 = 예보 대상 시각(hour_start + 1시간) - 발표 시각
LEAD_BUCKETS = ((1, 6, "1-6h"), (7, 24, "7-24h"), (25, 48, "25-48h"))
CONTINUOUS = ("temp_c", "humidity_pct", "wind_ms")
MIN_RAIN_HOURS = 10  # 계획: 비 온 관측 시간이 이보다 적으면 강수 지표는 참고로만


def _float(value: str | None) -> float:
    if value is None or str(value).strip() == "":
        return math.nan
    return float(value)


def observed_hourly(items: Iterable[dict]) -> dict[datetime, dict[str, float]]:
    """ASOS 시간자료 항목 → {hour_start: 날씨 특징}. dbt `stg_weather_hourly`와 같은 규칙.

    관측 시각 t의 값을 hour_start = t - 1시간에 붙인다. 11~3월 강수는 3시간 누적을 앞선 세 시간에
    1/3씩, 눈은 3시간 신적설 > 0이면 앞선 세 시간 모두 1. 빈 강수·신적설은 0으로 본다.
    """
    observations: dict[datetime, dict[str, float]] = {}
    for item in items:
        observed_at = datetime.strptime(item["tm"], "%Y-%m-%d %H:%M").replace(tzinfo=KST)
        rain = _float(item.get("rn"))
        snow = _float(item.get("hr3Fhsc"))
        observations[observed_at] = {
            "temp_c": _float(item.get("ta")),
            "wind_ms": _float(item.get("ws")),
            "humidity_pct": _float(item.get("hm")),
            "rain_mm": 0.0 if math.isnan(rain) else rain,
            "new_snow_3h_cm": 0.0 if math.isnan(snow) else snow,
        }
    out = {}
    for observed_at, obs in observations.items():
        three_hour_at = observed_at + ((3 - observed_at.hour % 3) % 3) * HOUR
        covering = observations.get(three_hour_at)
        if observed_at.month in (11, 12, 1, 2, 3):
            rain = (covering["rain_mm"] if covering else 0.0) / 3
        else:
            rain = obs["rain_mm"]
        new_snow = covering["new_snow_3h_cm"] if covering else 0.0
        out[observed_at - HOUR] = {
            "temp_c": obs["temp_c"],
            "rain_mm": rain,
            "wind_ms": obs["wind_ms"],
            "humidity_pct": obs["humidity_pct"],
            "is_snow": float(new_snow > 0),
        }
    return out


def lead_bucket(hour_start: datetime, base: datetime) -> str | None:
    lead = (hour_start + HOUR - base) / HOUR
    for low, high, name in LEAD_BUCKETS:
        if low <= lead <= high:
            return name
    return None


def _both(pairs, name):
    return [
        (f[name], o[name]) for f, o in pairs if not (math.isnan(f[name]) or math.isnan(o[name]))
    ]


def _share(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _events(pairs, name: str) -> dict:
    """0보다 크면 '있음'으로 본 일치율·재현율·정밀도."""
    both = _both(pairs, name)
    fc = [f > 0 for f, _ in both]
    ob = [o > 0 for _, o in both]
    hits = sum(f and o for f, o in zip(fc, ob, strict=True))
    return {
        "n": len(both),
        "observed_hours": sum(ob),
        "forecast_hours": sum(fc),
        "agreement": _share(sum(f == o for f, o in zip(fc, ob, strict=True)), len(both)),
        "recall": _share(hits, sum(ob)),
        "precision": _share(hits, sum(fc)),
    }


def _summarize(pairs: list[tuple[dict, dict]]) -> dict:
    summary: dict = {"pairs": len(pairs)}
    for name in CONTINUOUS:
        both = _both(pairs, name)
        diffs = [f - o for f, o in both]
        summary[name] = {
            "n": len(diffs),
            "mae": float(np.mean(np.abs(diffs))) if diffs else None,
            "bias": float(np.mean(diffs)) if diffs else None,
        }
    rain = _events(pairs, "rain_mm")
    wet = [(f, o) for f, o in _both(pairs, "rain_mm") if o > 0]
    rain["amount_mae_when_observed"] = float(np.mean([abs(f - o) for f, o in wet])) if wet else None
    rain["reference_only"] = rain["observed_hours"] < MIN_RAIN_HOURS
    summary["rain"] = rain
    snow = _events(pairs, "is_snow")
    snow["measurable"] = snow["observed_hours"] > 0
    summary["snow"] = snow
    return summary


def weather_errors(
    issues: Iterable[tuple[datetime, dict[datetime, dict[str, float]]]],
    observed: dict[datetime, dict[str, float]],
) -> dict[str, dict]:
    """(발표 시각, 그 발표의 hour_start별 예보 특징) 목록과 관측을 간격 구간별로 비교한다.

    같은 시각이 여러 발표에 들어 있으면 발표마다 따로 센다(서비스가 발표마다 다시 예측하므로).
    """
    pairs: dict[str, list] = {name: [] for *_, name in LEAD_BUCKETS}
    for base, weather in issues:
        for hour_start, forecast in weather.items():
            bucket = lead_bucket(hour_start, base)
            if bucket is None or hour_start not in observed:
                continue
            pairs[bucket].append((forecast, observed[hour_start]))
    return {name: _summarize(values) for name, values in pairs.items()}


def predict_both(
    predict: Callable[[np.ndarray], Any],
    stations: list[dict],
    hours: list[datetime],
    forecast: dict[datetime, dict[str, float]],
    observed: dict[datetime, dict[str, float]],
    artifacts: Artifacts,
    features: list[str],
) -> tuple[list[tuple[str, datetime]], np.ndarray, np.ndarray]:
    """같은 대여소·시각을 예보 날씨와 관측 날씨로 각각 예측한다. 날씨 외 특징이 다르면 오류."""
    hours = [h for h in hours if h in forecast and h in observed]
    by_forecast, keys = feature_rows(stations, hours, forecast, artifacts, features)
    by_observed, _ = feature_rows(stations, hours, observed, artifacts, features)
    others = [i for i, name in enumerate(features) if name not in WEATHER_FEATURES]
    if not np.array_equal(by_forecast[:, others], by_observed[:, others], equal_nan=True):
        raise AssertionError("날씨 외 특징이 두 예측에서 다름")
    if not keys:
        return keys, np.empty(0), np.empty(0)
    return keys, np.asarray(predict(by_forecast)), np.asarray(predict(by_observed))


@dataclass
class ShiftTotals:
    """여러 발표에 걸친 예측 흔들림 합계. 비 옴은 관측 기준."""

    abs_diff: dict[str, float] = field(default_factory=lambda: {"all": 0.0, "wet": 0.0, "dry": 0.0})
    rows: dict[str, int] = field(default_factory=lambda: {"all": 0, "wet": 0, "dry": 0})
    forecast_sum: float = 0.0
    observed_sum: float = 0.0
    abs_diff_3h: float = 0.0
    stations_3h: int = 0
    overlaps: list[float] = field(default_factory=list)

    def add(
        self, keys, by_forecast, by_observed, observed, as_of, bike_counts, top_n=50, horizon=3
    ):
        hourly_f: dict[str, dict[datetime, float]] = {}
        hourly_o: dict[str, dict[datetime, float]] = {}
        for (station_id, hour_start), f, o in zip(keys, by_forecast, by_observed, strict=True):
            kind = "wet" if observed[hour_start]["rain_mm"] > 0 else "dry"
            for group in ("all", kind):
                self.abs_diff[group] += abs(float(f) - float(o))
                self.rows[group] += 1
            self.forecast_sum += float(f)
            self.observed_sum += float(o)
            hourly_f.setdefault(station_id, {})[hour_start] = float(f)
            hourly_o.setdefault(station_id, {})[hour_start] = float(o)
        expected_f, expected_o = {}, {}
        for station_id in hourly_f:
            ef = expected_rentals(as_of, horizon, hourly_f[station_id])
            eo = expected_rentals(as_of, horizon, hourly_o[station_id])
            if ef is None or eo is None:
                continue
            expected_f[station_id], expected_o[station_id] = ef, eo
            self.abs_diff_3h += abs(ef - eo)
            self.stations_3h += 1
        if bike_counts is not None:
            listed_f = top_shortage(bike_counts, expected_f, top_n)
            listed_o = top_shortage(bike_counts, expected_o, top_n)
            if listed_f:
                self.overlaps.append(len(set(listed_f) & set(listed_o)) / len(listed_f))

    def report(self) -> dict:
        def mean(group):
            return self.abs_diff[group] / self.rows[group] if self.rows[group] else None

        return {
            "rows": dict(self.rows),
            "mean_abs_diff": {g: mean(g) for g in ("all", "wet", "dry")},
            "mean_ratio_forecast_over_observed": (
                self.forecast_sum / self.observed_sum if self.observed_sum else None
            ),
            "mean_abs_diff_3h_sum": (
                self.abs_diff_3h / self.stations_3h if self.stations_3h else None
            ),
            "top50_overlap": {
                "issues": len(self.overlaps),
                "mean": float(np.mean(self.overlaps)) if self.overlaps else None,
                "min": float(np.min(self.overlaps)) if self.overlaps else None,
            },
        }


def top_shortage(bike_counts: dict[str, int], expected: dict[str, float], n: int) -> list[str]:
    """`/shortage-risk`와 같은 순서: 부족 대수(예상 대여 - 자전거 수) > 0, 큰 순, 같으면 ID순."""
    rows = [
        (value - bike_counts[station_id], station_id)
        for station_id, value in expected.items()
        if station_id in bike_counts and value - bike_counts[station_id] > 0
    ]
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [station_id for _, station_id in rows[:n]]


if __name__ == "__main__":
    import argparse
    import json
    import os
    from datetime import date
    from pathlib import Path

    import httpx
    from dotenv import load_dotenv
    from sqlalchemy import select

    from bike_demand.ingest import weather as asos
    from bike_demand.model.final import load_serving_model
    from bike_demand.model.forecast_weather import hourly_weather
    from bike_demand.serving.db import make_engine
    from bike_demand.serving.models import RealtimeSnapshot, Station, WeatherForecast

    parser = argparse.ArgumentParser(description="v4(T12) 예보 vs 관측 날씨 측정")
    parser.add_argument("--first-issue", default="2026-09-17T02:00")
    parser.add_argument("--last-issue", default="2026-09-23T23:00")
    parser.add_argument("--observed", type=Path, default=Path("work/t12/asos_observed.json"))
    parser.add_argument("--fetch-observed", action="store_true", help="관측 JSON을 API로 받아 둔다")
    parser.add_argument("--models", type=Path, default=Path("data/models/v1"))
    parser.add_argument("--out", type=Path, default=Path("work/t12/result.json"))
    args = parser.parse_args()

    first = datetime.fromisoformat(args.first_issue).replace(tzinfo=KST)
    last = datetime.fromisoformat(args.last_issue).replace(tzinfo=KST)
    if args.fetch_observed:
        load_dotenv()
        key = os.environ.get("DATA_GO_KR_SERVICE_KEY")
        if not key:
            raise SystemExit(".env에 DATA_GO_KR_SERVICE_KEY가 없음")
        end = (last + timedelta(hours=49)).date()
        if end >= date.today():
            raise SystemExit(f"관측은 전날까지만 제공됨: {end} 다음 날 이후에 받는다")
        with httpx.Client(timeout=30) as client:
            items = asos.fetch_range(client, key, first.date(), end)
        args.observed.parent.mkdir(parents=True, exist_ok=True)
        args.observed.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        print("observed items", len(items))
    observed = observed_hourly(json.loads(args.observed.read_text(encoding="utf-8")))

    engine = make_engine()
    generation, booster, artifacts = load_serving_model(args.models)
    features = booster.feature_name()
    issues, skipped = [], {"no_snapshot": []}
    totals = ShiftTotals()
    with engine.connect() as conn:
        bases = (
            conn.execute(
                select(WeatherForecast.base_datetime)
                .where(WeatherForecast.base_datetime.between(first, last))
                .distinct()
                .order_by(WeatherForecast.base_datetime)
            )
            .scalars()
            .all()
        )
        for base in bases:
            rows = conn.execute(
                select(
                    WeatherForecast.fcst_datetime, WeatherForecast.category, WeatherForecast.value
                ).where(WeatherForecast.base_datetime == base)
            ).all()
            forecast = hourly_weather((r[0], r[1], r[2]) for r in rows)
            issues.append((base, forecast))
            snapshot_at = conn.execute(
                select(RealtimeSnapshot.fetched_at)
                .where(RealtimeSnapshot.fetched_at.between(base, base + timedelta(minutes=30)))
                .order_by(RealtimeSnapshot.fetched_at)
                .limit(1)
            ).scalar_one_or_none()
            if snapshot_at is None:
                skipped["no_snapshot"].append(base.isoformat())
                continue
            snapshot = conn.execute(
                select(Station.station_id, Station.district, Station.docks, Station.lat,
                       Station.lon, RealtimeSnapshot.bike_count)
                .join(RealtimeSnapshot, RealtimeSnapshot.station_id == Station.station_id)
                .where(RealtimeSnapshot.fetched_at == snapshot_at)
                .order_by(Station.station_id)
            ).mappings().all()  # fmt: skip
            stations = [dict(r) for r in snapshot]
            as_of = snapshot_at.astimezone(KST)
            first_hour = as_of.replace(minute=0, second=0, microsecond=0)
            hours = [first_hour + i * HOUR for i in range(48)]
            keys, by_f, by_o = predict_both(
                booster.predict, stations, hours, forecast, observed, artifacts, features
            )
            bikes = {r["station_id"]: r["bike_count"] for r in stations}
            totals.add(keys, by_f, by_o, observed, as_of, bikes)
            print(base.astimezone(KST).isoformat(), len(keys), flush=True)

    expected_issues = int((last - first) / timedelta(hours=3)) + 1
    report = {
        "generation": generation,
        "issues": {"expected": expected_issues, "found": len(bases), "skipped": skipped},
        "observed_hours": len(observed),
        "weather_errors": weather_errors(issues, observed),
        "prediction_shift": totals.report(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
