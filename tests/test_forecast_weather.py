# [무엇을 검증하나] 기상청 단기예보 문자열을 학습 때 쓴 ASOS 날씨 특징과 같은 숫자·시각 규칙으로
# 바꾸는 model/forecast_weather.py를 확인한다. 강수·적설 문자열 해석(parse_amount), 숫자
# 해석(parse_number), 예보 시각 → hour_start 정렬과 항목 → 특징 이름 매핑(hourly_weather).
# [왜 중요한가] 모델은 관측 날씨로 학습하고 서비스는 예보 날씨를 넣는다. 단위나 시각을 한 시간이라도
# 다르게 붙이면 학습과 서비스의 특징 뜻이 달라져 예측이 조용히 틀어진다. 모르는 형식은 0이 아니라
# NaN(결측)이어야 LightGBM이 결측으로 다룬다.

import math
from datetime import datetime, timedelta, timezone

import pytest

from bike_demand.model.forecast_weather import hourly_weather, parse_amount, parse_number

KST = timezone(timedelta(hours=9))


# 기상청 표기 규칙대로 바꾸는지 본다: '없음'은 0, 'a 미만'은 a의 절반(1mm 미만 → 0.5), 'a~b'는
# 가운데 값, 'a 이상'은 a, 단위가 붙은 숫자는 그 숫자. 강수(mm)와 적설(cm)이 같은 규칙을 쓴다.
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


# 빈 값, None, 처음 보는 단어, 숫자 없는 기호·단위는 추측하지 않고 NaN. 0으로 바꾸면 '비 안 옴'으로
# 잘못 들어간다.
@pytest.mark.parametrize("text", ["", None, "약간", "-", "mm"])
def test_parse_amount_unknown_is_nan(text):
    assert math.isnan(parse_amount(text))


# 기온(음수 포함)·풍속·습도 같은 숫자 항목. 숫자가 아닌 값은 NaN.
def test_parse_number():
    assert parse_number("22") == 22.0
    assert parse_number("-3.5") == -3.5
    assert parse_number("1.1") == 1.1
    assert math.isnan(parse_number("강함"))


# 09시 예보 값이 hour_start 08시(08:00~09:00 시간)에 붙는지 확인한다. 학습(dbt stg_weather_hourly)이
# 관측 시각 t를 t − 1시간에 붙였으므로(강수가 직전 1시간 누적) 예보도 같은 규칙을 따라야 한다. 쓰지
# 않는 항목(POP, 강수확률)은 무시되고, SNO '적설없음'은 is_snow 0이 된다.
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
