"""진단(측정 아님): 서비스 예측의 수준을 지난 해들의 같은 시기 실제와 비교한다.

서비스 기간의 실제 대여 수는 반기마다 공개돼 아직 없다. 그래서 정확도가 아니라, 서비스 모델이 만든
예측의 평균 수준이 지난 해 같은 시기(비 안 온 시간)와 얼마나 다른지만 본다. 읽기만 한다.

- 예측: 서비스 DB `predictions`에서 주어진 버전 접두어의 행 중 `--since`~`--until`(KST, 끝 포함)
  시각, `--last-issue`까지의 발표로 만든 것. 시각마다 그중 가장 최근 발표의 값 하나.
  기본값이 문서 결과를 만든 범위(2026-09-17 15:00 ~ 09-23 19:00, 09-21 20시 발표까지)라
  그 뒤 쌓인 예측은 읽지 않는다. 공휴일(KST 날짜) 시간은 실제 쪽처럼 뺀다. 예보에 비가 있던
  시간은 거르지 않는다(기본 범위의 예보에는 비가 없었음)
- 실제: warehouse `int_station_hour_grid`(0건 포함)의 각 해 같은 달·일 구간 중
  강수 0, 공휴일 아닌 시간
- 같은 대여소 x 쉬는 날 여부 x 시각끼리 평균을 맞추고, 예측 쪽 시간 수로 가중해 비교한다

실행: uv run python scripts/check_prediction_level.py
      (기본값 --version-prefix v3-M3p- --since 2026-09-17T15:00 --until 2026-09-23T19:00
       --last-issue 2026-09-21T20:00 → 407,847행·149시간)
결과는 docs/experiments.md의 "서비스 예측 수준 진단"에 적었다.
"""

# [파일 역할] README 한계 '예측 수준이 높을 수 있다'와 docs/experiments.md '서비스 예측 수준 진단'의
# 숫자(예측 ÷ 실제 2023 0.98, 2024 1.02, 2025 1.15)를 만드는 읽기 전용 스크립트다(커밋 2240c16).
# 판정 규칙이 있는 측정이 아니고, 결과로 모델을 바꾸지 않는다.
# [흐름] (1) 서비스 DB에서 예측을 읽어 (대여소, 쉬는 날, 시각) 칸별 평균과 시간 수를 만든다 → (2)
# warehouse에서 해마다 같은 날짜 구간의 실제 칸별 평균을 만든다 → (3) 두 쪽에 다 있는 칸만, 예측 쪽
# 시간 수로 가중해 평균과 비율을 낸다 → (4) 해마다 상반기 대여소당 시간당 평균을 덧붙인다.
# [이력] 커밋 4eb819c: 처음에는 --since 뒤 모든 예측을 읽어 서비스가 예측을 쌓을수록 기본 실행
# 결과가 문서 값(407,847행·149시간)에서 597,435행·218시간으로 바뀌었다. 그래서 --until과
# --last-issue로 범위를 고정했다. 커밋 d5b8025: 예측 쪽에서도 공휴일 시간을 빼도록 고쳤다.
# [연결] tests/test_check_prediction_level.py가 범위 경계, 공휴일 제외, 공통 칸이 없는 해의 처리를
# 확인한다.

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# duckdb: warehouse(dbt 결과) 파일을 읽기 전용으로 열어 과거 실제 대여를 집계한다. holidays: 예측 쪽
# 공휴일을 거르는 한국 달력(dbt 공휴일 시드와 같은 패키지). sqlalchemy text: 서비스 Postgres에 원시
# SQL을 바인드 변수와 함께 보낸다.
import duckdb
import holidays
from sqlalchemy import Engine, text

from bike_demand.serving.db import make_engine

# KST 고정 오프셋(+9). 한국은 서머타임이 없어 고정 오프셋으로 충분하다.
KST = timezone(timedelta(hours=9))


# 서비스 예측을 칸별 평균으로 요약한다. 출력: 칸별 평균 예측, 칸별 시간 수(가중치), 쓰인 예측 시각
# 목록.
# 왜 시간 수도 돌려주나: 실제 쪽 평균을 예측 쪽과 같은 요일·시각 구성으로 맞춰 비교하려면 칸마다
# 예측이 몇 시간 들어 있는지가 필요하다(weighted_ratio). 호출 위치: 아래 실행부,
# tests/test_check_prediction_level.py.
def predicted_cells(
    engine: Engine, version_prefix: str, since: datetime, until: datetime, last_issue: datetime
) -> tuple[dict, dict, list[datetime]]:
    """({(대여소, 쉬는 날, 시각): 평균 예측}, {같은 키: 시간 수}, 예측 시각 목록).

    since~until(끝 포함)의 시각만, last_issue 이하의 발표(버전 이름 끝의 fcstYYYYMMDDHHMM)로 만든
    행만 읽는다. 그래서 예측이 더 쌓여도 같은 범위를 다시 읽는다.
    KST 날짜가 공휴일인 시간은 뺀다(실제 쪽 `not is_holiday`와 같은 규칙). 남은 시간의 쉬는 날
    여부는 주말이므로 실제 쪽 is_offday와 같다.
    """
    # 같은 (대여소, 시간)을 여러 발표가 예측하므로 distinct on으로 하나만 남긴다. model_version이
    # 'v3-M3p-세대-fcstYYYYMMDDHHMM' 형식이고 발표 시각이 고정 폭 숫자라 문자열 내림차순 첫 행이
    # 가장 최근 발표다. 발표 상한도 split_part로 떼어 낸 fcst 뒤 숫자를 같은 형식의 문자열과
    # 비교한다.
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                select distinct on (station_id, hour_start)
                       station_id, hour_start, predicted_rentals
                from predictions
                where model_version like :prefix
                  and hour_start >= :since and hour_start <= :until
                  and split_part(model_version, '-fcst', 2) <= :last_issue
                order by station_id, hour_start, model_version desc
                """
            ),
            {
                "prefix": version_prefix + "%",
                "since": since,
                "until": until,
                "last_issue": last_issue.astimezone(KST).strftime("%Y%m%d%H%M"),
            },
        ).all()
    # dbt 공휴일 시드(holidays_seed.py)와 같은 달력. 행마다 만들지 않고 한 번만 만든다
    kr = holidays.country_holidays(
        "KR", years=range(since.astimezone(KST).year, until.astimezone(KST).year + 1)
    )
    # KST 날짜가 공휴일인 시간을 뺀다. 모델은 평일 공휴일을 쉬는 날로 예측하므로, 빼지 않으면 추석
    # 같은 날의 낮은 예측이 평일 칸에 섞여 평일·전체 비율을 낮춘다(커밋 d5b8025).
    kept = [row for row in rows if row[1].astimezone(KST).date() not in kr]
    # 칸 키 = (대여소, 주말 여부, 시각). 공휴일을 이미 뺐으므로 주말 여부가 실제 쪽 is_offday와 같은
    # 뜻이다.
    cells: dict[tuple[str, bool, int], list[float]] = defaultdict(list)
    for station_id, hour_start, value in kept:
        local = hour_start.astimezone(KST)
        cells[(station_id, local.isoweekday() >= 6, local.hour)].append(value)
    means = {key: sum(values) / len(values) for key, values in cells.items()}
    weights = {key: len(values) for key, values in cells.items()}
    return means, weights, sorted({row[1] for row in kept})


# 한 해의 같은 날짜 구간에서 비 안 온(rain_mm = 0) 공휴일 아닌 시간의 칸별 실제 평균 대여 수.
# 격자(0건 포함)를 쓰므로 시간당 평균이 된다. 관측 날씨가 빈 시간은 rain_mm = 0 비교가 참이 아니라서
# 함께 빠진다. 호출 위치: 아래 실행부(해마다 한 번).
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


# 고른 칸들에서 예측 쪽 시간 수로 가중한 (예측 평균, 실제 평균, 예측 ÷ 실제)를 돌려준다. 같은
# 가중치를 양쪽에 써서 대여소·요일·시각 구성 차이가 비율에 끼어들지 않게 한다(docs/experiments.md
# 진단 절 '예측 쪽 시간 수로 가중'). keys가 비면 나눌 수가 없어 ValueError를 낸다. 해마다의 비교는
# 아래 compare_year가 칸이 없는 해를 먼저 걸러 이 오류가 나지 않는다.
def weighted_ratio(predicted: dict, weights: dict, actual: dict, keys: list) -> tuple:
    if not keys:
        raise ValueError("비교할 칸이 없음")
    n = sum(weights[k] for k in keys)
    p = sum(predicted[k] * weights[k] for k in keys)
    a = sum(actual[k] * weights[k] for k in keys)
    return p / n, a / n, p / a


# 한 해의 결과 한 줄. 두 쪽에 다 있는 칸만 비교한다. 최근에 생긴 대여소는 지난 해 같은 시기 칸이
# 없어 빠진다(진단 한계에 적힌 점). 출력의 cells a/b가 쓰인 칸 수를 보여 준다. warehouse에 없는
# 해(예: --years 2022)처럼 공통 칸이 하나도 없으면 0으로 나누지 않고 '비교할 칸 없음'으로 적는다.
# 호출 위치: 아래 실행부(해마다 한 번), tests/test_check_prediction_level.py.
def compare_year(year: int, predicted: dict, weights: dict, actual: dict) -> str:
    keys = [key for key in predicted if key in actual]
    if not keys:
        return f"{year}: cells 0/{len(predicted)}  비교할 칸 없음"
    mean_p, mean_a, ratio = weighted_ratio(predicted, weights, actual, keys)
    line = f"{year}: cells {len(keys)}/{len(predicted)}  predicted {mean_p:.3f}"
    line += f"  actual {mean_a:.3f}  ratio {ratio:.3f}"
    # 평일 칸과 쉬는 날 칸의 비율을 따로도 적는다(문서 표의 '평일', '쉬는 날' 열).
    for offday, label in ((False, "weekday"), (True, "offday")):
        subset = [key for key in keys if key[1] == offday]
        if subset:
            line += f"  {label} {weighted_ratio(predicted, weights, actual, subset)[2]:.3f}"
    return line


if __name__ == "__main__":
    # 기본값은 문서 결과를 만든 범위로 고정돼 있다(커밋 4eb819c): 버전 접두어 v3-M3p-, 예측 시각
    # 2026-09-17 15시 ~ 09-23 19시(끝 포함), 09-21 20시 발표까지. 실제 쪽은 2023~2025년 9월 15~30일.
    parser = argparse.ArgumentParser(description="서비스 예측 수준 진단(읽기 전용)")
    parser.add_argument("--version-prefix", default="v3-M3p-")
    parser.add_argument("--since", default="2026-09-17T15:00", help="첫 예측 시각(KST)")
    parser.add_argument("--until", default="2026-09-23T19:00", help="마지막 예측 시각(KST, 포함)")
    parser.add_argument(
        "--last-issue", default="2026-09-21T20:00", help="이 발표(KST)까지로 만든 예측만"
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--start", default="09-15", help="비교할 구간 시작 MM-DD")
    parser.add_argument("--end", default="10-01", help="비교할 구간 끝 MM-DD (제외)")
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    args = parser.parse_args()

    # 명령줄 시각은 시간대 없이 받으므로 KST로 붙인다.
    since = datetime.fromisoformat(args.since).replace(tzinfo=KST)
    until = datetime.fromisoformat(args.until).replace(tzinfo=KST)
    last_issue = datetime.fromisoformat(args.last_issue).replace(tzinfo=KST)
    predicted, weights, hours = predicted_cells(
        make_engine(), args.version_prefix, since, until, last_issue
    )
    if not predicted:
        raise SystemExit("조건에 맞는 예측이 없음")
    first, last = hours[0].astimezone(KST), hours[-1].astimezone(KST)
    print(f"predictions: {sum(weights.values())} rows, {len(hours)} hours, {first} ~ {last}")

    # DuckDB 파일은 쓰기 연결을 한 프로세스만 열 수 있어, 읽기만 하는 진단은 읽기 전용으로 연다.
    con = duckdb.connect(str(args.warehouse), read_only=True)
    for year in args.years:
        actual = actual_cells(con, year, args.start, args.end)
        print(compare_year(year, predicted, weights, actual))
    # 해마다 상반기(1~6월) 대여소당 시간당 평균. 가장 최근에 공개된 반기(2026 상반기)의 수준이
    # 2025년과 비슷한지 보려는 값이다(문서: 2026 1.500, 2025 1.483).
    halves = con.execute(
        """
        select year(hour_start), avg(rentals) from int_station_hour_grid
        where month(hour_start) <= 6 group by 1 order by 1
        """
    ).fetchall()
    print("first-half hourly mean per station:", ", ".join(f"{y} {m:.3f}" for y, m in halves))
