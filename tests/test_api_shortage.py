from datetime import datetime, timedelta, timezone

import pytest

from bike_demand.api.shortage import expected_rentals, hours_overlapping

KST = timezone(timedelta(hours=9))


def at(hour, minute=0, second=0):
    return datetime(2026, 9, 17, hour, minute, second, tzinfo=KST)


def hourly(*starts, value=6.0):
    return {at(h): value for h in starts}


def test_on_the_hour_uses_whole_hours():
    assert hours_overlapping(at(8), 3) == [at(8), at(9), at(10)]
    assert expected_rentals(at(8), 3, hourly(8, 9, 10)) == pytest.approx(18.0)


def test_partial_hours_are_prorated():
    # 08:20 + 3시간 → 08시 40/60 + 09시 + 10시 + 11시 20/60 (docs/api.md 예시)
    assert hours_overlapping(at(8, 20), 3) == [at(8), at(9), at(10), at(11)]
    assert expected_rentals(at(8, 20), 3, hourly(8, 9, 10, 11)) == pytest.approx(6 * 3)


def test_half_hour():
    values = {at(8): 2.0, at(9): 4.0}
    assert expected_rentals(at(8, 30), 1, values) == pytest.approx(1.0 + 2.0)


def test_missing_needed_hour_gives_none():
    assert expected_rentals(at(8, 20), 3, hourly(8, 9, 10)) is None  # 11시 빠짐
    assert expected_rentals(at(8), 3, hourly(8, 9, 10, 11)) == pytest.approx(18.0)  # 11시 불필요


def test_seconds_matter_at_boundaries():
    # 08:59:59 + 1시간 → 08시는 1초, 09시는 59분 59초
    values = {at(8): 3600.0, at(9): 3600.0}
    assert expected_rentals(at(8, 59, 59), 1, values) == pytest.approx(3600.0)


def test_crosses_midnight():
    late = datetime(2026, 9, 17, 23, 30, tzinfo=KST)
    values = {late.replace(minute=0): 2.0, datetime(2026, 9, 18, 0, tzinfo=KST): 4.0}
    assert expected_rentals(late, 1, values) == pytest.approx(1.0 + 2.0)
