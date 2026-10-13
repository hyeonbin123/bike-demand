"""부족 위험 계산(docs/api.md `GET /shortage-risk`). DB와 무관한 순수 함수."""

# [파일 역할] /shortage-risk의 핵심 계산인 '지금부터 N시간 동안의 예상 대여 수'를 시간별 예측에서
# 구하는 순수 함수 모음이다.
# [왜 따로 두나] DB 없이 단위 테스트할 수 있고(tests/test_api_shortage.py), 같은 계산을 측정 코드도
# 그대로 써서 서비스와 측정의 정의가 어긋나지 않게 한다. model/forecast_gap.py(v4)가 이 함수를
# 가져다 쓰고, model/shortage_eval.py(v5)는 같은 비율 계산을 자기 쪽에 다시 구현했다고 docstring에
# 적어 두었다.
# [핵심] 스냅샷 시각(as_of)이 정시가 아니면 첫 시간과 마지막 시간은 일부만 구간에 걸치므로, 시간별
# 예측에 겹친 분의 비율을 곱해 더한다(docs/api.md 계산 절).

from __future__ import annotations

from datetime import datetime, timedelta

# 한 시간 간격. 시간 계산을 timedelta로 해서 시간대(+09:00)가 붙은 datetime에도 그대로 쓴다.
HOUR = timedelta(hours=1)


# 예: as_of 08:20, hours 3 → 구간 [08:20, 11:20) → [08시, 09시, 10시, 11시]. as_of가 정시면 정확히
# hours개.
# 호출 위치: expected_rentals, model/forecast_gap.py(구간의 비 옴 여부 판단).
def hours_overlapping(as_of: datetime, hours: int) -> list[datetime]:
    """[as_of, as_of + hours) 구간과 겹치는 정시 시작 시간들."""
    end = as_of + hours * HOUR
    # as_of를 정시로 내린 시간부터 시작해 구간 끝 직전까지 한 시간씩 늘린다.
    current = as_of.replace(minute=0, second=0, microsecond=0)
    result = []
    while current < end:
        result.append(current)
        current += HOUR
    return result


# 입력: 기준 시각, 볼 시간 수, {정시 시작: 그 시간의 예측 대여 수}. 출력: 구간의 예상 대여 합(float)
# 또는 None.
# None을 돌려주는 이유: 필요한 시간의 예측이 빠졌을 때 0으로 채우면 부족 대수가 실제보다 작게
# 나오므로, 그 대여소를 목록에서 빼도록 호출하는 쪽에 알린다(docs/api.md).
# 호출 위치: api/main.py shortage_risk, model/forecast_gap.py.
def expected_rentals(as_of: datetime, hours: int, hourly: dict[datetime, float]) -> float | None:
    """구간과 겹치는 시간별 예측의 합. 일부만 걸친 시간은 겹친 비율만큼 곱한다.

    필요한 시간의 예측이 하나라도 없으면 None.
    """
    end = as_of + hours * HOUR
    total = 0.0
    for start in hours_overlapping(as_of, hours):
        # 이 시간 [start, start+1h)과 구간 [as_of, end)의 겹친 길이. 예: as_of 08:20이면 08시는
        # 40분, 11시는 20분이 겹친다.
        overlap = min(start + HOUR, end) - max(start, as_of)
        # 겹침이 없는 시간은 건너뛴다. hours_overlapping이 겹치는 시간만 주므로 실제로는 생기지 않는
        # 방어용 조건이다.
        if overlap <= timedelta(0):
            continue
        if start not in hourly:
            return None
        # timedelta끼리 나누면 float 비율(0~1)이 나온다. 예측을 그 비율만큼만 더한다(한 시간 안에서
        # 대여가 고르게 일어난다고 보는 가정).
        total += hourly[start] * (overlap / HOUR)
    return total
