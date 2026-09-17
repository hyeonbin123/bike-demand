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
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np

from bike_demand.api.shortage import expected_rentals, hours_overlapping
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


# 서비스 예측 작업은 발표 15분 뒤에 돈다(DAG `15 2,5,8…`). 대여소 집합은 그 전 6시간 안의 스냅샷,
# 부족 목록은 그 전 30분 안의 스냅샷일 때만 만든다(서비스 예측·API 한도). v4 측정 전 보충.
SERVICE_DELAY = timedelta(minutes=15)
STATION_MAX_AGE = timedelta(hours=6)
LIST_MAX_AGE = timedelta(minutes=30)
GROUPS = ("all", "wet", "dry", "unknown")
MISSING_WEATHER = dict.fromkeys(WEATHER_FEATURES, math.nan)


def issue_as_of(base: datetime) -> datetime:
    return base + SERVICE_DELAY


def observed_end(last_issue: datetime) -> date:
    """마지막 발표의 가장 긴 간격(48시간) 예보가 가리키는 관측 시각의 날짜."""
    return (last_issue + 48 * HOUR).astimezone(KST).date()


def check_fetchable(end: date, today: date) -> None:
    if end >= today:
        raise SystemExit(f"관측은 전날까지만 제공됨: {end} 다음 날 이후에 받는다")


def _both(pairs, name):
    return [
        (h, f[name], o[name])
        for h, f, o in pairs
        if not (math.isnan(f[name]) or math.isnan(o[name]))
    ]


def _share(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _events(pairs, name: str) -> dict:
    """0보다 크면 '있음'. 쌍 수와 고유한 관측 시각 수를 함께 적는다(같은 시각은 발표마다 한 쌍)."""
    both = _both(pairs, name)
    fc = [f > 0 for _, f, _ in both]
    ob = [o > 0 for _, _, o in both]
    hits = sum(f and o for f, o in zip(fc, ob, strict=True))
    return {
        "n": len(both),
        "observed_pairs": sum(ob),
        "observed_unique_hours": len({h for h, _, o in both if o > 0}),
        "forecast_pairs": sum(fc),
        "agreement": _share(sum(f == o for f, o in zip(fc, ob, strict=True)), len(both)),
        "recall": _share(hits, sum(ob)),
        "precision": _share(hits, sum(fc)),
    }


def _summarize(pairs: list[tuple[datetime, dict, dict]]) -> dict:
    summary: dict = {"pairs": len(pairs), "unique_hours": len({h for h, _, _ in pairs})}
    for name in CONTINUOUS:
        diffs = [f - o for _, f, o in _both(pairs, name)]
        summary[name] = {
            "n": len(diffs),
            "mae": float(np.mean(np.abs(diffs))) if diffs else None,
            "bias": float(np.mean(diffs)) if diffs else None,
        }
    rain = _events(pairs, "rain_mm")
    wet = [abs(f - o) for _, f, o in _both(pairs, "rain_mm") if o > 0]
    rain["amount_mae_when_observed"] = float(np.mean(wet)) if wet else None  # PCP 대표값 변환 뒤
    rain["reference_only"] = rain["observed_unique_hours"] < MIN_RAIN_HOURS
    summary["rain"] = rain
    snow = _events(pairs, "is_snow")
    snow["measurable"] = snow["observed_unique_hours"] > 0
    summary["snow"] = snow
    return summary


def weather_errors(
    issues: Iterable[tuple[datetime, dict[datetime, dict[str, float]]]],
    observed: dict[datetime, dict[str, float]],
) -> dict[str, dict]:
    """(발표 시각, 그 발표의 hour_start별 예보 특징) 목록과 관측을 간격 구간별로 비교한다.

    같은 시각이 여러 발표에 들어 있으면 발표마다 따로 센다(서비스가 발표마다 다시 예측하므로).
    관측 행이 없는 시각은 날씨 오차에서만 뺀다.
    """
    pairs: dict[str, list] = {name: [] for *_, name in LEAD_BUCKETS}
    for base, weather in issues:
        for hour_start, forecast in weather.items():
            bucket = lead_bucket(hour_start, base)
            if bucket is None or hour_start not in observed:
                continue
            pairs[bucket].append((hour_start, forecast, observed[hour_start]))
    return {name: _summarize(values) for name, values in pairs.items()}


def rain_state(observed: dict[datetime, dict[str, float]], hours: Iterable[datetime]) -> str:
    """관측 기준 비 옴 구분: 하나라도 비 → wet, 아니면 하나라도 관측 없음 → unknown, 그 밖 dry."""
    missing = False
    for hour in hours:
        values = observed.get(hour)
        if values is None:
            missing = True
        elif values["rain_mm"] > 0:
            return "wet"
    return "unknown" if missing else "dry"


def predict_both(
    predict: Callable[[np.ndarray], Any],
    stations: list[dict],
    hours: list[datetime],
    forecast: dict[datetime, dict[str, float]],
    observed: dict[datetime, dict[str, float]],
    artifacts: Artifacts,
    features: list[str],
) -> tuple[list[tuple[str, datetime]], np.ndarray, np.ndarray]:
    """같은 대여소·시각을 예보 날씨와 관측 날씨로 각각 예측한다. 날씨 외 특징이 다르면 오류.

    예보가 있는 시각은 모두 쓴다. 관측 행이 없는 시각은 날씨 5개를 결측으로 둔다(보충 T46).
    """
    hours = [h for h in hours if h in forecast]
    observed_weather = {h: observed.get(h, MISSING_WEATHER) for h in hours}
    by_forecast, keys = feature_rows(stations, hours, forecast, artifacts, features)
    by_observed, _ = feature_rows(stations, hours, observed_weather, artifacts, features)
    others = [i for i, name in enumerate(features) if name not in WEATHER_FEATURES]
    if not np.array_equal(by_forecast[:, others], by_observed[:, others], equal_nan=True):
        raise AssertionError("날씨 외 특징이 두 예측에서 다름")
    if not keys:
        return keys, np.empty(0), np.empty(0)
    return keys, np.asarray(predict(by_forecast)), np.asarray(predict(by_observed))


def _zeros(kind=float):
    return dict.fromkeys(GROUPS, kind())


@dataclass
class ShiftTotals:
    """여러 발표에 걸친 예측 흔들림 합계. 무리(all·wet·dry·unknown)는 관측 강수 기준(보충 T49)."""

    abs_diff: dict[str, float] = field(default_factory=_zeros)
    rows: dict[str, int] = field(default_factory=lambda: _zeros(int))
    forecast_sum: dict[str, float] = field(default_factory=_zeros)
    observed_sum: dict[str, float] = field(default_factory=_zeros)
    abs_diff_3h: dict[str, float] = field(default_factory=_zeros)
    stations_3h: dict[str, int] = field(default_factory=lambda: _zeros(int))
    issues: dict[str, int] = field(default_factory=lambda: _zeros(int))
    overlaps: dict[str, list[float]] = field(default_factory=lambda: {g: [] for g in GROUPS})
    lists_skipped: dict[str, int] = field(
        default_factory=lambda: {"no_recent_snapshot": 0, "empty_forecast_list": 0}
    )

    def add(
        self, keys, by_forecast, by_observed, observed, as_of, bike_counts, top_n=50, horizon=3
    ):
        """bike_counts=None이면(30분 안 스냅샷 없음) 부족 목록 비교만 뺀다."""
        window = rain_state(observed, hours_overlapping(as_of, horizon))
        for group in ("all", window):
            self.issues[group] += 1
        hourly_f: dict[str, dict[datetime, float]] = {}
        hourly_o: dict[str, dict[datetime, float]] = {}
        for (station_id, hour_start), f, o in zip(keys, by_forecast, by_observed, strict=True):
            f, o = float(f), float(o)
            for group in ("all", rain_state(observed, [hour_start])):
                self.abs_diff[group] += abs(f - o)
                self.rows[group] += 1
                self.forecast_sum[group] += f
                self.observed_sum[group] += o
            hourly_f.setdefault(station_id, {})[hour_start] = f
            hourly_o.setdefault(station_id, {})[hour_start] = o
        expected_f, expected_o = {}, {}
        for station_id in hourly_f:
            ef = expected_rentals(as_of, horizon, hourly_f[station_id])
            eo = expected_rentals(as_of, horizon, hourly_o[station_id])
            if ef is None or eo is None:
                continue
            expected_f[station_id], expected_o[station_id] = ef, eo
            for group in ("all", window):
                self.abs_diff_3h[group] += abs(ef - eo)
                self.stations_3h[group] += 1
        if bike_counts is None:
            self.lists_skipped["no_recent_snapshot"] += 1
            return
        listed_f = top_shortage(bike_counts, expected_f, top_n)
        listed_o = top_shortage(bike_counts, expected_o, top_n)
        if not listed_f:
            self.lists_skipped["empty_forecast_list"] += 1
            return
        overlap = len(set(listed_f) & set(listed_o)) / len(listed_f)
        for group in ("all", window):
            self.overlaps[group].append(overlap)

    def report(self) -> dict:
        def ratio(numerator, denominator):
            return numerator / denominator if denominator else None

        return {
            group: {
                "rows": self.rows[group],
                "mean_abs_diff": ratio(self.abs_diff[group], self.rows[group]),
                "mean_ratio_forecast_over_observed": ratio(
                    self.forecast_sum[group], self.observed_sum[group]
                ),
                "issues": self.issues[group],
                "stations_3h": self.stations_3h[group],
                "mean_abs_diff_3h_sum": ratio(self.abs_diff_3h[group], self.stations_3h[group]),
                "top50_overlap": {
                    "issues": len(self.overlaps[group]),
                    "mean": float(np.mean(self.overlaps[group])) if self.overlaps[group] else None,
                    "min": float(np.min(self.overlaps[group])) if self.overlaps[group] else None,
                },
            }
            for group in GROUPS
        } | {"lists_skipped": dict(self.lists_skipped)}


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
        end = observed_end(last)
        check_fetchable(end, datetime.now(KST).date())
        with httpx.Client(timeout=30) as client:
            items = asos.fetch_range(client, key, first.date(), end)
        args.observed.parent.mkdir(parents=True, exist_ok=True)
        args.observed.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        print("observed items", len(items))
    observed = observed_hourly(json.loads(args.observed.read_text(encoding="utf-8")))

    engine = make_engine()
    generation, booster, artifacts = load_serving_model(args.models)
    features = booster.feature_name()
    issues, skipped = [], {"no_stations": []}
    totals = ShiftTotals()
    grid = (WeatherForecast.nx == 60) & (WeatherForecast.ny == 127)
    with engine.connect() as conn:
        bases = (
            conn.execute(
                select(WeatherForecast.base_datetime)
                .where(grid, WeatherForecast.base_datetime.between(first, last))
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
                ).where(grid, WeatherForecast.base_datetime == base)
            ).all()
            forecast = hourly_weather((r[0], r[1], r[2]) for r in rows)
            issues.append((base, forecast))
            as_of = issue_as_of(base).astimezone(KST)
            snapshot_at = conn.execute(
                select(RealtimeSnapshot.fetched_at)
                .where(RealtimeSnapshot.fetched_at.between(as_of - STATION_MAX_AGE, as_of))
                .order_by(RealtimeSnapshot.fetched_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if snapshot_at is None:
                skipped["no_stations"].append(base.isoformat())
                continue
            snapshot = conn.execute(
                select(Station.station_id, Station.district, Station.docks, Station.lat,
                       Station.lon, RealtimeSnapshot.bike_count)
                .join(RealtimeSnapshot, RealtimeSnapshot.station_id == Station.station_id)
                .where(RealtimeSnapshot.fetched_at == snapshot_at)
                .order_by(Station.station_id)
            ).mappings().all()  # fmt: skip
            stations = [dict(r) for r in snapshot]
            first_hour = as_of.replace(minute=0, second=0, microsecond=0)
            hours = [first_hour + i * HOUR for i in range(48)]
            keys, by_f, by_o = predict_both(
                booster.predict, stations, hours, forecast, observed, artifacts, features
            )
            recent = as_of - snapshot_at <= LIST_MAX_AGE
            bikes = {r["station_id"]: r["bike_count"] for r in stations} if recent else None
            totals.add(keys, by_f, by_o, observed, as_of, bikes)
            print(
                base.astimezone(KST).isoformat(), len(keys), "list" if recent else "-", flush=True
            )

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
