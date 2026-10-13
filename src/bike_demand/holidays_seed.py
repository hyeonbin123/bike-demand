"""dbt 시드 `dbt/seeds/kr_holidays.csv`(한국 공휴일)를 만든다.

`holidays` 패키지의 KR 달력(대체공휴일·임시공휴일·선거일 포함)을 쓴다. 법이 바뀌거나 임시공휴일이
새로 정해지면 패키지를 올리고 다시 만든다. 하루에 이름이 둘이면 `; `로 이어 붙인 한 행이다.
"""

# [파일 역할] dbt 시드 CSV를 만드는 도구. 만든 CSV는 저장소에 커밋되어 dbt가 kr_holidays 표로 읽고,
# dim_hours가 이 표로 is_holiday·is_offday(쉬는 날)를 정한다.
# [왜 시드인가] 대체·임시공휴일과 선거일은 SQL 규칙으로 만들기 어려워, holidays 패키지가 아는 날짜
# 목록을 파일로 저장해 둔다. 서비스 예측(model/predict.py)도 같은 패키지의 KR 달력으로 쉬는 날을
# 정하므로 학습과 서비스의 공휴일 정의가 같다.
# [실행] python -m bike_demand.holidays_seed (기본 2023~2027년). 테스트:
# tests/test_holidays_seed.py.

from __future__ import annotations

import csv
from pathlib import Path

# holidays: 나라별 공휴일 달력 패키지. 한국(KR) 달력에 대체·임시공휴일과 선거일이 들어
# 있다(docs/experiments.md v1 알려진 한계).
import holidays


# first_year~last_year(양끝 포함)의 한국 공휴일을 날짜순으로 CSV에 쓰고 행 수를 돌려준다. 한 날짜에
# 공휴일 이름이 여럿이면 패키지가 이름을 이어 붙인 한 항목으로 주므로 날짜당 한 행이다.
# 호출 위치: 아래 실행부, tests/test_holidays_seed.py.
def write_seed(path: Path, first_year: int, last_year: int) -> int:
    calendar = holidays.country_holidays("KR", years=range(first_year, last_year + 1))
    rows = sorted(calendar.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    # csv 모듈 권장대로 newline=""로 열고 줄 끝을 LF로 고정한다. 저장소는 .gitattributes로 LF를
    # 쓰는데, Windows에서 기본값으로 쓰면 CRLF가 되기 때문이다.
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["holiday_date", "holiday_name"])
        writer.writerows((day.isoformat(), name) for day, name in rows)
    return len(rows)


if __name__ == "__main__":
    import argparse

    # 기본 기간 2023~2027년. dim_hours 달력의 시작(2023-01-01)부터, 2026년 하반기 자료를 넣을
    # 2027년까지 덮는다.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("dbt/seeds/kr_holidays.csv"))
    parser.add_argument("--first-year", type=int, default=2023)
    parser.add_argument("--last-year", type=int, default=2027)
    args = parser.parse_args()
    print(write_seed(args.out, args.first_year, args.last_year), "holidays")
