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

해석할 수 없는 값은 결측(NaN)으로 두고, LightGBM이 결측으로 처리한다.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import datetime, timedelta

_NUMBER = r"(\d+(?:\.\d+)?)"
_RANGE = re.compile(rf"^{_NUMBER}\s*~\s*{_NUMBER}\s*(?:mm|cm)$")
_BELOW = re.compile(rf"^{_NUMBER}\s*(?:mm|cm)\s*미만$")
_ABOVE = re.compile(rf"^{_NUMBER}\s*(?:mm|cm)\s*이상$")
_PLAIN = re.compile(rf"^-?{_NUMBER[1:-1]}\s*(?:mm|cm|℃|m/s|%)?$")

CATEGORY_FEATURE = {"TMP": "temp_c", "PCP": "rain_mm", "WSD": "wind_ms", "REH": "humidity_pct"}


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


def parse_number(value: str | None) -> float:
    if value is None:
        return math.nan
    text = value.strip()
    if not _PLAIN.match(text):
        return math.nan
    return float(re.sub(r"[^\d.\-]", "", text))


def hourly_weather(rows: Iterable[tuple[datetime, str, str]]) -> dict[datetime, dict[str, float]]:
    """(예보 시각, 항목, 값) → {hour_start: {특징: 값}}. 같은 발표 하나의 행을 넣는다."""
    out: dict[datetime, dict[str, float]] = {}
    for fcst_at, category, value in rows:
        hour_start = fcst_at - timedelta(hours=1)
        features = out.setdefault(
            hour_start,
            {"temp_c": math.nan, "rain_mm": math.nan, "wind_ms": math.nan,
             "humidity_pct": math.nan, "is_snow": math.nan},
        )  # fmt: skip
        if category == "PCP":
            features["rain_mm"] = parse_amount(value)
        elif category == "SNO":
            amount = parse_amount(value)
            features["is_snow"] = math.nan if math.isnan(amount) else float(amount > 0)
        elif category in CATEGORY_FEATURE:
            features[CATEGORY_FEATURE[category]] = parse_number(value)
    return out
