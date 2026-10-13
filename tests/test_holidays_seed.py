# [무엇을 검증하나] dbt 시드 kr_holidays.csv를 만드는 holidays_seed.write_seed가 holidays 패키지의
# KR 달력에서 대체공휴일과 선거일까지 담는지 확인한다.
# [왜 중요한가] 공휴일 여부(is_holiday, is_offday)는 대여소 패턴을 '평일 / 쉬는 날'로 나누는 핵심
# 특징이다(docs/experiments.md v1: 대체·임시공휴일, 선거일 포함). 이런 날이 빠지면 쉬는 날이 평일로
# 분류된다.

import csv

from bike_demand.holidays_seed import write_seed


# 2024~2025년 시드를 임시 폴더에 써서(저장소의 시드 파일은 건드리지 않음) 2024-02-12 설날 대체 휴일,
# 2024-04-10 국회의원 선거일, 2025-06-03 대통령 선거일이 있는지 본다. 2024-02-13(평일)은 없어야
# 한다. 행 수가 30보다 많은지로 두 해가 모두 들어갔는지도 확인한다.
def test_seed_has_substitute_and_election_days(tmp_path):
    out = tmp_path / "kr_holidays.csv"
    assert write_seed(out, 2024, 2025) > 30
    with out.open(encoding="utf-8") as stream:
        rows = {r["holiday_date"]: r["holiday_name"] for r in csv.DictReader(stream)}
    assert "2024-02-12" in rows  # 설날 대체 휴일
    assert "2024-04-10" in rows  # 국회의원 선거일
    assert "2025-06-03" in rows  # 대통령 선거일
    assert "2024-02-13" not in rows
