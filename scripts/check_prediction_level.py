"""진단(측정 아님): 서비스 예측의 수준을 지난 해들의 같은 시기 실제와 비교한다.

서비스 기간의 실제 대여 수는 반기마다 공개돼 아직 없다. 그래서 정확도가 아니라, 서비스 모델이 만든
예측의 평균 수준이 지난 해 같은 시기(비 안 온 시간)와 얼마나 다른지만 본다. 읽기만 한다.

- 예측: 서비스 DB `predictions`에서 주어진 버전 접두어의 행. 시각마다 가장 최근 발표로 만든 값 하나
- 실제: warehouse `int_station_hour_grid`(0건 포함)의 각 해 같은 달·일 구간 중
  강수 0, 공휴일 아닌 시간
- 같은 대여소 x 쉬는 날 여부 x 시각끼리 평균을 맞추고, 예측 쪽 시간 수로 가중해 비교한다

실행: uv run python scripts/check_prediction_level.py
      (기본값 --version-prefix v3-M3p- --since 2026-09-17T15:00)
결과는 docs/experiments.md의 "서비스 예측 수준 진단"에 적었다.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
from sqlalchemy import text

from bike_demand.serving.db import make_engine

KST = timezone(timedelta(hours=9))


def predicted_cells(version_prefix: str, since: datetime) -> tuple[dict, dict, list[datetime]]:
    """({(대여소, 쉬는 날, 시각): 평균 예측}, {같은 키: 시간 수}, 예측 시각 목록)."""
    with make_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                select distinct on (station_id, hour_start)
                       station_id, hour_start, predicted_rentals
                from predictions
                where model_version like :prefix and hour_start >= :since
                order by station_id, hour_start, model_version desc
                """
            ),
            {"prefix": version_prefix + "%", "since": since},
        ).all()
    cells: dict[tuple[str, bool, int], list[float]] = defaultdict(list)
    for station_id, hour_start, value in rows:
        local = hour_start.astimezone(KST)
        cells[(station_id, local.isoweekday() >= 6, local.hour)].append(value)
    means = {key: sum(values) / len(values) for key, values in cells.items()}
    weights = {key: len(values) for key, values in cells.items()}
    return means, weights, sorted({row[1] for row in rows})


def actual_cells(con: duckdb.DuckDBPyConnection, year: int, start: str, end: str) -> dict:
    """그 해 start~end(MM-DD, end 제외) 중 강수 0·공휴일 아닌 시간의 셀별 평균 대여 수."""
    rows = con.execute(
        """
        select g.station_id, d.is_offday, d.hour_of_day, avg(g.rentals)
        from int_station_hour_grid g join dim_hours d using (hour_start)
        where g.hour_start >= cast(? as timestamp) and g.hour_start < cast(? as timestamp)
          and d.rain_mm = 0 and not d.is_holiday
        group by 1, 2, 3
        """,
        [f"{year}-{start}", f"{year}-{end}"],
    ).fetchall()
    return {(s, bool(offday), int(hour)): mean for s, offday, hour, mean in rows}


def weighted_ratio(predicted: dict, weights: dict, actual: dict, keys: list) -> tuple:
    n = sum(weights[k] for k in keys)
    p = sum(predicted[k] * weights[k] for k in keys)
    a = sum(actual[k] * weights[k] for k in keys)
    return p / n, a / n, p / a


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="서비스 예측 수준 진단(읽기 전용)")
    parser.add_argument("--version-prefix", default="v3-M3p-")
    parser.add_argument("--since", default="2026-09-17T15:00", help="KST")
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--start", default="09-15", help="비교할 구간 시작 MM-DD")
    parser.add_argument("--end", default="10-01", help="비교할 구간 끝 MM-DD (제외)")
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    args = parser.parse_args()

    since = datetime.fromisoformat(args.since).replace(tzinfo=KST)
    predicted, weights, hours = predicted_cells(args.version_prefix, since)
    if not predicted:
        raise SystemExit("조건에 맞는 예측이 없음")
    first, last = hours[0].astimezone(KST), hours[-1].astimezone(KST)
    print(f"predictions: {sum(weights.values())} rows, {len(hours)} hours, {first} ~ {last}")

    con = duckdb.connect(str(args.warehouse), read_only=True)
    for year in args.years:
        actual = actual_cells(con, year, args.start, args.end)
        keys = [key for key in predicted if key in actual]
        mean_p, mean_a, ratio = weighted_ratio(predicted, weights, actual, keys)
        line = f"{year}: cells {len(keys)}/{len(predicted)}  predicted {mean_p:.3f}"
        line += f"  actual {mean_a:.3f}  ratio {ratio:.3f}"
        for offday, label in ((False, "weekday"), (True, "offday")):
            subset = [key for key in keys if key[1] == offday]
            if subset:
                line += f"  {label} {weighted_ratio(predicted, weights, actual, subset)[2]:.3f}"
        print(line)
    halves = con.execute(
        """
        select year(hour_start), avg(rentals) from int_station_hour_grid
        where month(hour_start) <= 6 group by 1 order by 1
        """
    ).fetchall()
    print("first-half hourly mean per station:", ", ".join(f"{y} {m:.3f}" for y, m in halves))
