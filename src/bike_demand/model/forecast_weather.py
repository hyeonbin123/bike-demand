"""기상청 단기예보 값을 학습 때 쓴 ASOS 날씨 특징과 같은 형태로 바꾼다.

학습(dbt `stg_weather_hourly`)은 관측 시각 t의 값을 hour_start = t - 1시간에 붙였다
(강수량이 직전 1시간 누적이라서). 예보도 같은 규칙으로
예보 시각 t를 hour_start = t - 1시간에 붙인다.

| 특징 | 예보 항목 | 바꾸는 방법 |
|---|---|---|
| temp_c | TMP (℃) | 숫자 |
| rain_mm | PCP (1시간 강수량) | 강수없음 0, 1mm 미만 0.5, a~bmm 가운데 값, amm 이상 a, 숫자+mm |
| wind_ms | WSD (m/s) | 숫자 |
| humidity_pct | REH (%) | 숫자 |
| is_snow | SNO (1시간 신적설) | 적설없음이면 0, 그 밖에 0보다 크면 1 |

학습 쪽 `is_snow`는 ASOS 3시간 신적설 > 0(`stg_weather_hourly.is_new_snow`)이다.
둘 다 "새로 내린 눈"이라 뜻이 같다(쌓인 눈 깊이가 아님, T24).

해석할 수 없는 값은 결측(NaN)으로 두고, LightGBM이 결측으로 처리한다.
"""

# [파일 역할] 서비스 예측에서 '관측 날씨 특징' 자리에 넣을 값을 단기예보 문자열에서 만든다. 학습과
# 서비스의 날씨 정의를 맞추는 다리 역할이다.
# [왜 필요한가] 모델은 ASOS 관측으로 학습했지만 서비스 시점에는 미래 관측이 없으므로 예보를 넣는다.
# 예보 값은 '강수없음', '1mm 미만', '30.0~50.0mm' 같은 글자라 숫자로 바꾸는 규칙이 필요하고, 시각
# 정렬(t - 1시간)도 학습과 같아야 한다.
# [연결] model/predict.py(서비스 예측), model/forecast_gap.py(v4 측정), model/shortage_eval.py(v5
# 개발 자료의 예측 재계산)가 hourly_weather를 쓴다. 테스트: tests/test_forecast_weather.py.
# [한계] 범위 값을 가운데 값으로 바꾸는 데서 오차가 생긴다(docs/experiments.md v4 측정 전 보충).

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import datetime, timedelta

# 예보 문자열 패턴: 범위(a~b mm), 미만(a mm 미만), 이상(a mm 이상), 단위가 붙거나 없는 숫자.
# 적설(cm)도 같은 모양이라 함께 받는다.
_NUMBER = r"(\d+(?:\.\d+)?)"
_RANGE = re.compile(rf"^{_NUMBER}\s*~\s*{_NUMBER}\s*(?:mm|cm)$")
_BELOW = re.compile(rf"^{_NUMBER}\s*(?:mm|cm)\s*미만$")
_ABOVE = re.compile(rf"^{_NUMBER}\s*(?:mm|cm)\s*이상$")
_PLAIN = re.compile(rf"^-?{_NUMBER[1:-1]}\s*(?:mm|cm|℃|m/s|%)?$")

# 숫자만 바꾸면 되는 예보 항목 → 모델 특징 이름. 강수(PCP)와 신적설(SNO)은 별도 규칙이라 아래에서
# 따로 처리한다.
CATEGORY_FEATURE = {"TMP": "temp_c", "PCP": "rain_mm", "WSD": "wind_ms", "REH": "humidity_pct"}


# 규칙: '…없음' → 0, 범위 → 가운데 값, 'a 미만' → a/2(예: 1mm 미만 → 0.5), 'a 이상' → a, 그 밖은
# 숫자로 해석. 모르는 형식은 0으로 가정하지 않고 NaN으로 두어 모델이 결측으로 처리하게 한다.
def parse_amount(value: str | None) -> float:
    """강수량·적설 문자열을 숫자로. 모르는 형식은 NaN."""
    if value is None:
        return math.nan
    text = value.strip()
    if not text:
        return math.nan
    if text.endswith("없음"):
        return 0.0
    if match := _RANGE.match(text):
        low, high = map(float, match.groups())
        return (low + high) / 2
    if match := _BELOW.match(text):
        return float(match.group(1)) / 2
    if match := _ABOVE.match(text):
        return float(match.group(1))
    return parse_number(text)


# 단위가 붙을 수 있는 숫자 문자열(예: 22, -3.5, 2.5m/s)을 float로. 패턴에 맞지 않으면 NaN. 단위
# 글자를 지운 뒤 float로 바꾼다.
def parse_number(value: str | None) -> float:
    if value is None:
        return math.nan
    text = value.strip()
    if not _PLAIN.match(text):
        return math.nan
    return float(re.sub(r"[^\d.\-]", "", text))


# 한 발표의 (예보 시각, 항목, 값) 행들을 hour_start별 날씨 특징 사전으로 모은다. 다섯 특징은 NaN으로
# 시작해, 해당 항목이 없는 시간은 결측으로 남는다.
def hourly_weather(rows: Iterable[tuple[datetime, str, str]]) -> dict[datetime, dict[str, float]]:
    """(예보 시각, 항목, 값) → {hour_start: {특징: 값}}. 같은 발표 하나의 행을 넣는다."""
    out: dict[datetime, dict[str, float]] = {}
    for fcst_at, category, value in rows:
        # 학습 쪽 stg_weather_hourly와 같은 정렬: 시각 t의 값은 hour_start = t - 1시간 행에 붙인다.
        hour_start = fcst_at - timedelta(hours=1)
        features = out.setdefault(
            hour_start,
            {"temp_c": math.nan, "rain_mm": math.nan, "wind_ms": math.nan,
             "humidity_pct": math.nan, "is_snow": math.nan},
        )  # fmt: skip
        if category == "PCP":
            features["rain_mm"] = parse_amount(value)
        # 신적설은 양이 아니라 '새로 눈이 옴' 여부(0/1)로 바꾼다. 학습 쪽 is_snow(3시간 신적설 >
        # 0)와 뜻을 맞춘 것이다(docs/experiments.md v3 측정 전 보충).
        elif category == "SNO":
            amount = parse_amount(value)
            features["is_snow"] = math.nan if math.isnan(amount) else float(amount > 0)
        elif category in CATEGORY_FEATURE:
            features[CATEGORY_FEATURE[category]] = parse_number(value)
    return out
