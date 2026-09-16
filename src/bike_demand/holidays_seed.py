"""dbt 시드 `dbt/seeds/kr_holidays.csv`(한국 공휴일)를 만든다.

`holidays` 패키지의 KR 달력(대체공휴일·임시공휴일·선거일 포함)을 쓴다. 법이 바뀌거나 임시공휴일이
새로 정해지면 패키지를 올리고 다시 만든다. 하루에 이름이 둘이면 `; `로 이어 붙인 한 행이다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import holidays


def write_seed(path: Path, first_year: int, last_year: int) -> int:
    calendar = holidays.country_holidays("KR", years=range(first_year, last_year + 1))
    rows = sorted(calendar.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["holiday_date", "holiday_name"])
        writer.writerows((day.isoformat(), name) for day, name in rows)
    return len(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("dbt/seeds/kr_holidays.csv"))
    parser.add_argument("--first-year", type=int, default=2023)
    parser.add_argument("--last-year", type=int, default=2027)
    args = parser.parse_args()
    print(write_seed(args.out, args.first_year, args.last_year), "holidays")
