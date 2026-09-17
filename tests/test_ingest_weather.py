import json
from datetime import date

import httpx
import pyarrow.parquet as pq
import pytest

from bike_demand.ingest import weather

SECRET = "secret-key-123"


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


def hourly_items(month: str, hours: int):
    return [
        {"tm": f"{month}-01 {h % 24:02d}:00", "stnId": "108", "ta": "1.5", "rn": "", "hm": "40"}
        for h in range(hours)
    ]


def test_months_between_crosses_year():
    assert list(weather.months_between("2023-11", "2024-02")) == [
        "2023-11",
        "2023-12",
        "2024-01",
        "2024-02",
    ]


def test_is_complete_needs_last_day_before_today():
    assert weather.is_complete("2026-08", date(2026, 9, 1))
    assert not weather.is_complete("2026-08", date(2026, 8, 31))


def paged_handler(items, seen_pages, page_size=500):
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        assert params["numOfRows"] == str(page_size)
        page = int(params["pageNo"])
        seen_pages.append(page)
        chunk = items[(page - 1) * page_size : page * page_size]
        return httpx.Response(200, json=ok_body(chunk, total=len(items)))

    return handler


def test_fetch_month_pages_until_total(monkeypatch):
    monkeypatch.setattr(weather, "PAGE_SIZE", 500)
    items = hourly_items("2023-02", 28 * 24)  # 672개 -> 500개씩 두 쪽
    seen_pages = []
    client = httpx.Client(transport=httpx.MockTransport(paged_handler(items, seen_pages)))
    result = weather.fetch_month(client, SECRET, "2023-02")
    assert len(result) == 672
    assert seen_pages == [1, 2]


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


def test_short_page_that_never_recovers_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body([], total=744))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    raw = tmp_path / "raw"
    with pytest.raises(weather.ApiError, match="0개"):
        list(weather.download(raw, ["2023-01"], SECRET, date(2023, 2, 10), client))
    assert not weather.raw_path(raw, "2023-01").exists()


def test_missing_observation_hours_are_allowed_when_total_matches():
    items = hourly_items("2023-01", 700)  # 관측 자체가 빠진 시간: API도 700이라고 알림

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body(items, total=700))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert len(weather.fetch_month(client, SECRET, "2023-01")) == 700


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


def test_http_logs_do_not_contain_the_key(caplog):
    items = hourly_items("2023-02", 672)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body(items))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with caplog.at_level("DEBUG"):
        weather.fetch_month(client, SECRET, "2023-02")
    assert SECRET not in caplog.text


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


def test_transport_error_does_not_leak_key(monkeypatch):
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(weather.ApiError) as info:
        weather.fetch_month(client, SECRET, "2023-02")
    assert SECRET not in str(info.value)


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
