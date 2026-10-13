# [무엇을 검증하나] /shortage-risk의 '앞으로 N시간 예상 대여'를 계산하는 순수
# 함수(api/shortage.py)의 부분 시간 비례를 DB 없이 숫자로 확인한다. 구간 [as_of, as_of + N시간)과
# 겹치는 정시 시작 시간을 고르고, 일부만 걸친 시간은 겹친 비율만큼 곱한다(docs/api.md).
# [왜 중요한가] 스냅샷 시각은 정시가 아니므로(10분마다 수집) 비례 계산이 틀리면 부족 대수가 한 시간
# 분량까지 틀어진다. v5 측정(shortage_eval)도 같은 방식으로 비례한다(docs/experiments.md v5 입력 E).

from datetime import datetime, timedelta, timezone

import pytest

# hours_overlapping: 구간과 겹치는 정시 시간 목록. expected_rentals: 비례 합(필요한 시간이 없으면
# None).
from bike_demand.api.shortage import expected_rentals, hours_overlapping

KST = timezone(timedelta(hours=9))


# 2026-09-17 KST의 시각을 만드는 도우미.
def at(hour, minute=0, second=0):
    return datetime(2026, 9, 17, hour, minute, second, tzinfo=KST)


# 주어진 정시마다 같은 예측값(기본 6대)을 가진 사전.
def hourly(*starts, value=6.0):
    return {at(h): value for h in starts}


# 정시에서 시작하면 정확히 N개의 시간을 통째로 더한다(다음 시간은 겹침 0이라 포함하지 않음).
def test_on_the_hour_uses_whole_hours():
    assert hours_overlapping(at(8), 3) == [at(8), at(9), at(10)]
    assert expected_rentals(at(8), 3, hourly(8, 9, 10)) == pytest.approx(18.0)


# 계약서 예시 그대로: 08:20부터 3시간이면 네 시간에 걸치고, 양 끝 시간은 40분·20분만큼만 센다. 합은
# 6 × 3 = 18.
def test_partial_hours_are_prorated():
    # 08:20 + 3시간 → 08시 40/60 + 09시 + 10시 + 11시 20/60 (docs/api.md 예시)
    assert hours_overlapping(at(8, 20), 3) == [at(8), at(9), at(10), at(11)]
    assert expected_rentals(at(8, 20), 3, hourly(8, 9, 10, 11)) == pytest.approx(6 * 3)


# 08:30부터 1시간: 08시 값의 절반(1.0) + 09시 값의 절반(2.0).
def test_half_hour():
    values = {at(8): 2.0, at(9): 4.0}
    assert expected_rentals(at(8, 30), 1, values) == pytest.approx(1.0 + 2.0)


# 필요한 시간(11시)이 빠지면 0으로 보지 않고 None을 돌려줘 API가 그 대여소를 빼게 한다. 정시
# 시작이면 11시는 필요 없으므로 있어도 더하지 않는다.
def test_missing_needed_hour_gives_none():
    assert expected_rentals(at(8, 20), 3, hourly(8, 9, 10)) is None  # 11시 빠짐
    assert expected_rentals(at(8), 3, hourly(8, 9, 10, 11)) == pytest.approx(18.0)  # 11시 불필요


# 초 단위까지 겹침을 잰다. 08:59:59부터 1시간이면 08시는 1초, 09시는 3,599초라, 시간당 3,600이면
# 합이 정확히 3,600이다. 겹친 길이를 분 단위로 버림하면(1초 → 0분, 59분 59초 → 59분) 3,540이 되어
# 차이가 드러난다.
def test_seconds_matter_at_boundaries():
    # 08:59:59 + 1시간 → 08시는 1초, 09시는 59분 59초
    values = {at(8): 3600.0, at(9): 3600.0}
    assert expected_rentals(at(8, 59, 59), 1, values) == pytest.approx(3600.0)


# 23:30부터 1시간이 자정을 넘어 다음 날 0시 예측까지 이어지는지 확인한다(날짜가 바뀌어도 시각 산술로
# 처리).
def test_crosses_midnight():
    late = datetime(2026, 9, 17, 23, 30, tzinfo=KST)
    values = {late.replace(minute=0): 2.0, datetime(2026, 9, 18, 0, tzinfo=KST): 4.0}
    assert expected_rentals(late, 1, values) == pytest.approx(1.0 + 2.0)
