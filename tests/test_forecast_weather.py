import math
from datetime import datetime, timedelta, timezone

import pytest

from bike_demand.model.forecast_weather import hourly_weather, parse_amount, parse_number

KST = timezone(timedelta(hours=9))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("강수없음", 0.0),
        ("적설없음", 0.0),
        ("1mm 미만", 0.5),
        ("1.0mm 미만", 0.5),
        ("1cm 미만", 0.5),
        ("3.0mm", 3.0),
        ("7mm", 7.0),
        ("30.0~50.0mm", 40.0),
        ("50.0mm 이상", 50.0),
        ("5.0cm 이상", 5.0),
    ],
)
def test_parse_amount(text, expected):
    assert parse_amount(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", None, "약간", "-", "mm"])
def test_parse_amount_unknown_is_nan(text):
    assert math.isnan(parse_amount(text))


def test_parse_number():
    assert parse_number("22") == 22.0
    assert parse_number("-3.5") == -3.5
    assert parse_number("1.1") == 1.1
    assert math.isnan(parse_number("강함"))


def test_hourly_weather_aligns_with_training_convention():
    at = datetime(2026, 9, 17, 9, tzinfo=KST)
    rows = [
        (at, "TMP", "22"),
        (at, "PCP", "1mm 미만"),
        (at, "WSD", "1.1"),
        (at, "REH", "55"),
        (at, "SNO", "적설없음"),
        (at, "POP", "30"),  # 쓰지 않는 항목
    ]
    weather = hourly_weather(rows)
    assert list(weather) == [datetime(2026, 9, 17, 8, tzinfo=KST)]  # 09시 예보 → 08시 시작 시간
    assert weather[datetime(2026, 9, 17, 8, tzinfo=KST)] == {
        "temp_c": 22.0, "rain_mm": 0.5, "wind_ms": 1.1, "humidity_pct": 55.0, "is_snow": 0.0,
    }  # fmt: skip
