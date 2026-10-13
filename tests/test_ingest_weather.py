# [무엇을 검증하나] 기상청 ASOS 시간자료 수집(src/bike_demand/ingest/weather.py)을 네트워크 없이
# 확인한다.
# - 달 목록 만들기, '끝난 달'(마지막 날이 어제 이전)만 저장하기
# - 쪽 나눠 받기: 전체 건수(totalCount)까지 받고, 모자란 쪽이나 쪽마다 다른 전체 건수는 그 쪽만 다시
#   받으며, 끝내 실패하면 부분 달을 저장하지 않는다. 관측 자체가 빠진 시간은 API 전체 건수와 맞으면
#   허용한다
# - 인증키 보호: 오류 메시지·로그에 키(그대로, URL 인코딩)가 나오지 않고, 키를 반사한 응답(유니코드
#   이스케이프 포함)은 저장하지 않는다
# - 원본 JSON → Parquet 변환에서 빈 값·없는 필드는 null
# [왜 중요한가] 날씨는 학습 특징이다(gain 기여 기온 6%·강수 5% 정도, README 결과). 한 달이 일부만
# 저장되면 '이미 있음'으로 다시 받지 않아 빈 시간이 영구히 남는다. 공공데이터포털 키는 URL 쿼리에
# 들어가 httpx 오류·로그에 섞이기 쉽다.

import json
from datetime import date

# httpx.MockTransport로 응답을 만들어 쪽 번호·인자·실패를 흉내 낸다. pyarrow.parquet은 변환 결과
# 확인용.
import httpx
import pyarrow.parquet as pq
import pytest

from bike_demand.ingest import weather

# 가짜 인증키. 실제 키는 .env에만 있고 테스트는 쓰지 않는다.
SECRET = "secret-key-123"


# 정상 응답(resultCode 00) 봉투. totalCount를 따로 줄 수 있어 '전체 건수와 받은 수가 다른' 응답을
# 만든다.
def ok_body(items, total=None):
    return {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
            "body": {
                "dataType": "JSON",
                "items": {"item": items},
                "totalCount": len(items) if total is None else total,
            },
        }
    }


# 서울(108) 관측 hours개. 강수(rn)는 빈 문자열(비 안 온 시간은 빈 값이라는 원본 특성, docs/data.md).
def hourly_items(month: str, hours: int):
    return [
        {"tm": f"{month}-01 {h % 24:02d}:00", "stnId": "108", "ta": "1.5", "rn": "", "hm": "40"}
        for h in range(hours)
    ]


# 2023-11 ~ 2024-02처럼 해를 넘어도 양 끝을 포함해 달을 차례로 낸다.
def test_months_between_crosses_year():
    assert list(weather.months_between("2023-11", "2024-02")) == [
        "2023-11",
        "2023-12",
        "2024-01",
        "2024-02",
    ]


# API는 전날까지만 주므로 8월은 9월 1일부터 '끝난 달'이다. 8월 31일 당일에는 아직 아니다.
def test_is_complete_needs_last_day_before_today():
    assert weather.is_complete("2026-08", date(2026, 9, 1))
    assert not weather.is_complete("2026-08", date(2026, 8, 31))


# 쪽 번호대로 page_size개씩 잘라 주는 가짜 API. 요청한 쪽 크기(numOfRows)가 맞는지도 확인한다.
def paged_handler(items, seen_pages, page_size=500):
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        assert params["numOfRows"] == str(page_size)
        page = int(params["pageNo"])
        seen_pages.append(page)
        chunk = items[(page - 1) * page_size : page * page_size]
        return httpx.Response(200, json=ok_body(chunk, total=len(items)))

    return handler


# 쪽 크기를 500으로 줄여, 2023-02의 672시간이 두 쪽(500 + 172)으로 받아지고 전체 건수에 닿으면
# 멈추는지 본다.
def test_fetch_month_pages_until_total(monkeypatch):
    monkeypatch.setattr(weather, "PAGE_SIZE", 500)
    items = hourly_items("2023-02", 28 * 24)  # 672개 -> 500개씩 두 쪽
    seen_pages = []
    client = httpx.Client(transport=httpx.MockTransport(paged_handler(items, seen_pages)))
    result = weather.fetch_month(client, SECRET, "2023-02")
    assert len(result) == 672
    assert seen_pages == [1, 2]


# 2쪽이 처음에 비어서 오면(전체 건수로 보면 172개여야 함) 그 쪽만 다시 받아 회복한다. 호출은 [1, 2,
# 2].
def test_short_page_is_retried_then_recovers(monkeypatch):
    monkeypatch.setattr(weather, "PAGE_SIZE", 500)
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)
    items = hourly_items("2023-02", 672)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["pageNo"])
        calls.append(page)
        if page == 2 and calls.count(2) == 1:  # 두 번째 쪽이 처음에는 비어서 옴
            return httpx.Response(200, json=ok_body([], total=672))
        return httpx.Response(200, json=ok_body(items[(page - 1) * 500 : page * 500], total=672))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert len(weather.fetch_month(client, SECRET, "2023-02")) == 672
    assert calls == [1, 2, 2]


# 전체 744건이라면서 계속 0건이면 재시도를 다 쓰고 실패하고, 그달 원본 JSON을 만들지 않는다.
def test_short_page_that_never_recovers_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body([], total=744))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    raw = tmp_path / "raw"
    with pytest.raises(weather.ApiError, match="0개"):
        list(weather.download(raw, ["2023-01"], SECRET, date(2023, 2, 10), client))
    assert not weather.raw_path(raw, "2023-01").exists()


# 관측소가 기록하지 못한 시간이 있으면 API도 전체 건수를 그만큼 적게 알린다(744 대신 700). 이때는
# 오류가 아니라 받은 그대로 쓴다(빠진 시간은 dbt 테스트에서 드러남, download docstring).
def test_missing_observation_hours_are_allowed_when_total_matches():
    items = hourly_items("2023-01", 700)  # 관측 자체가 빠진 시간: API도 700이라고 알림

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body(items, total=700))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert len(weather.fetch_month(client, SECRET, "2023-01")) == 700


# 서버가 오류 메시지·XML 본문·코드 자리에 요청 URL이나 키를 넣어 보내는 세 경우. 키에 '+/='를 붙여,
# 그대로·URL 인코딩된 모양 어느 것도 ApiError 메시지에 나오지 않는지 본다. 자유 문자열은 보여 주지
# 않고 형식이 정해진 코드만 쓰는 방식(모듈 주석)을 확인하는 것이다.
@pytest.mark.parametrize(
    "response",
    [
        # resultMsg에 요청 URL을 되돌려 보내는 서버
        lambda url: httpx.Response(
            200,
            json={"response": {"header": {"resultCode": "99", "resultMsg": str(url)}}},
        ),
        # XML 오류 본문에 URL 인코딩된 키가 섞인 경우
        lambda url: httpx.Response(
            401,
            text=f"<OpenAPI_ServiceResponse><errMsg>{url}</errMsg></OpenAPI_ServiceResponse>",
        ),
        # 코드 자리에 자유 문자열
        lambda url: httpx.Response(
            200, json={"response": {"header": {"resultCode": f"bad {SECRET}"}}}
        ),
    ],
)
def test_error_responses_never_show_the_key(monkeypatch, response):
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)
    client = httpx.Client(transport=httpx.MockTransport(lambda request: response(request.url)))
    with pytest.raises(weather.ApiError) as info:
        weather.fetch_month(client, SECRET + "+/=", "2023-02")
    message = str(info.value)
    for form in (SECRET, "secret-key-123%2B%2F%3D", "secret-key-123+/="):
        assert form not in message


# 성공한 요청에서도 DEBUG 로그(httpx 요청 줄 등)에 키가 남지 않는다.
def test_http_logs_do_not_contain_the_key(caplog):
    items = hourly_items("2023-02", 672)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body(items))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with caplog.at_level("DEBUG"):
        weather.fetch_month(client, SECRET, "2023-02")
    assert SECRET not in caplog.text


# 인증키 오류는 JSON을 요청해도 XML로 온다. 이때 형식이 정해진 오류
# 이름(SERVICE_KEY_IS_NOT_REGISTERED_ERROR)은 메시지에 남겨 원인을 알 수 있게 하고, 키는 남기지
# 않는다.
def test_api_error_does_not_leak_key(monkeypatch):
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)
    xml = (
        "<OpenAPI_ServiceResponse><cmmMsgHeader><errMsg>SERVICE ERROR</errMsg>"
        "<returnAuthMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR</returnAuthMsg>"
        "<returnReasonCode>30</returnReasonCode></cmmMsgHeader></OpenAPI_ServiceResponse>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=xml)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(weather.ApiError) as info:
        weather.fetch_month(client, SECRET, "2023-02")
    assert "SERVICE_KEY_IS_NOT_REGISTERED_ERROR" in str(info.value)
    assert SECRET not in str(info.value)


# 연결 오류 예외 메시지에는 키가 든 URL이 들어가므로, ApiError에는 예외 종류 이름만 남는지 본다.
def test_transport_error_does_not_leak_key(monkeypatch):
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(weather.ApiError) as info:
        weather.fetch_month(client, SECRET, "2023-02")
    assert SECRET not in str(info.value)


# download의 세 상태: 원본이 있는 2022-12는 API 없이 'exists', 끝난 2023-01은 'saved 743/744'(받은
# 수/그달 시간 수, 한 시간 빠짐 허용), 오늘(2023-02-10) 기준 끝나지 않은 2023-02는 'incomplete'로
# 받지 않는다. API 호출은 2023-01 한 번뿐이다.
# 이어서 원본 두 달에서 Parquet을 만들면 2 + 743행이고, 빈 강수는 null, 응답에 없는 적설 필드도
# null이다.
def test_download_skips_existing_and_incomplete_then_parquet(tmp_path):
    raw = tmp_path / "raw"
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["startDt"])
        return httpx.Response(200, json=ok_body(hourly_items("2023-01", 31 * 24 - 1)))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    existing = weather.raw_path(raw, "2022-12")
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps(hourly_items("2022-12", 2)), encoding="utf-8")

    statuses = dict(
        weather.download(raw, ["2022-12", "2023-01", "2023-02"], SECRET, date(2023, 2, 10), client)
    )
    assert statuses == {"2022-12": "exists", "2023-01": "saved 743/744", "2023-02": "incomplete"}
    assert calls == ["20230101"]

    out = tmp_path / "bronze" / "asos.parquet"
    assert weather.to_parquet(raw, out) == 2 + 743
    row = pq.read_table(out).to_pylist()[0]
    assert row["temp_c"] == "1.5"
    assert row["rain_mm"] is None  # 빈 문자열 -> null
    assert row["snow_cm"] is None  # 응답에 없는 필드


# 응답 원문 바이트에는 키가 \uXXXX로 숨어 있어 원문 검사로는 안 보인다. JSON을 해석한 뒤 다시 검사해
# '반사'로 거부하고, 그달 원본을 저장하지 않는지 확인한다. chr(92)는 역슬래시다.
def test_key_hidden_by_json_unicode_escape_is_rejected(tmp_path, monkeypatch):
    """원문 검사로는 안 보이는 유니코드 이스케이프 키도 해석 뒤 검사해 저장하지 않는다(T25)."""
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)
    escaped = "".join(chr(92) + f"u{ord(c):04x}" for c in SECRET)
    item = '{"tm": "2023-01-01 01:00", "stnId": "108", "ta": "' + escaped + '"}'
    body = (
        '{"response": {"header": {"resultCode": "00"}, "body": {"totalCount": 1, '
        '"items": {"item": [' + item + "]}}}}"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "application/json"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    raw = tmp_path / "raw"
    assert SECRET not in body
    with pytest.raises(weather.ApiError, match="반사") as info:
        list(weather.download(raw, ["2023-01"], SECRET, date(2023, 2, 10), client))
    assert SECRET not in str(info.value)
    assert not weather.raw_path(raw, "2023-01").exists()


# 쪽 크기를 1로 줄여 두 쪽을 만든다. 2쪽이 전체 건수를 2 → 0으로 바꿔 알리면, 끝난 것으로 보지 않고
# 2쪽만 재시도하다 실패하며 1건짜리 부분 달을 저장하지 않는다.
def test_total_count_changing_between_pages_is_retried_then_fails(tmp_path, monkeypatch):
    """1쪽은 전체 2건이라 하고 2쪽은 0건이라 하면 부분 달을 저장하지 않는다(T26)."""
    monkeypatch.setattr(weather, "PAGE_SIZE", 1)
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)
    items = hourly_items("2023-01", 2)
    pages: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        pages.append(int(request.url.params["pageNo"]))
        if request.url.params["pageNo"] == "1":
            return httpx.Response(200, json=ok_body(items[:1], total=2))
        return httpx.Response(200, json=ok_body([], total=0))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    raw = tmp_path / "raw"
    with pytest.raises(weather.ApiError, match="전체 건수가 쪽마다 다름"):
        list(weather.download(raw, ["2023-01"], SECRET, date(2023, 2, 10), client))
    # 1쪽은 다시 받지 않고, 건수가 어긋난 2쪽만 세 번 시도한다(T40)
    assert pages == [1, 2, 2, 2]
    assert not weather.raw_path(raw, "2023-01").exists()


# v4 측정은 끝나지 않은 달(2026-09)의 17~25일 관측만 필요하다. fetch_range가 시작일 0시 ~ 끝날
# 23시로 요청 인자를 정확히 만드는지 본다(이 결과는 저장하지 않음).
def test_fetch_range_asks_for_the_given_days_only():
    """v4 측정은 끝나지 않은 달의 일부만 받는다(T12)."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=ok_body(hourly_items("2026-09", 3)))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        items = weather.fetch_range(client, SECRET, date(2026, 9, 17), date(2026, 9, 25))
    assert len(items) == 3
    params = seen[0]
    assert (params["startDt"], params["startHh"]) == ("20260917", "00")
    assert (params["endDt"], params["endHh"]) == ("20260925", "23")
