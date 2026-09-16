"""부족 위험 계산(docs/api.md `GET /shortage-risk`). DB와 무관한 순수 함수."""

from __future__ import annotations

from datetime import datetime, timedelta

HOUR = timedelta(hours=1)


def hours_overlapping(as_of: datetime, hours: int) -> list[datetime]:
    """[as_of, as_of + hours) 구간과 겹치는 정시 시작 시간들."""
    end = as_of + hours * HOUR
    current = as_of.replace(minute=0, second=0, microsecond=0)
    result = []
    while current < end:
        result.append(current)
        current += HOUR
    return result


def expected_rentals(as_of: datetime, hours: int, hourly: dict[datetime, float]) -> float | None:
    """구간과 겹치는 시간별 예측의 합. 일부만 걸친 시간은 겹친 비율만큼 곱한다.

    필요한 시간의 예측이 하나라도 없으면 None.
    """
    end = as_of + hours * HOUR
    total = 0.0
    for start in hours_overlapping(as_of, hours):
        overlap = min(start + HOUR, end) - max(start, as_of)
        if overlap <= timedelta(0):
            continue
        if start not in hourly:
            return None
        total += hourly[start] * (overlap / HOUR)
    return total
