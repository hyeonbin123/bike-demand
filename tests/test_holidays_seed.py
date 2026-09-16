import csv

from bike_demand.holidays_seed import write_seed


def test_seed_has_substitute_and_election_days(tmp_path):
    out = tmp_path / "kr_holidays.csv"
    assert write_seed(out, 2024, 2025) > 30
    with out.open(encoding="utf-8") as stream:
        rows = {r["holiday_date"]: r["holiday_name"] for r in csv.DictReader(stream)}
    assert "2024-02-12" in rows  # 설날 대체 휴일
    assert "2024-04-10" in rows  # 국회의원 선거일
    assert "2025-06-03" in rows  # 대통령 선거일
    assert "2024-02-13" not in rows
