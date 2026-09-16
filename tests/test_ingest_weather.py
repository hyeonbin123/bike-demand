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


def test_fetch_month_pages_until_total():
    items = hourly_items("2023-02", 28 * 24)  # 서버가 500개씩 준다고 가정해 두 쪽을 받게 함
    seen_pages = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        assert params["startDt"] == "20230201" and params["endDt"] == "20230228"
        page = int(params["pageNo"])
        seen_pages.append(page)
        chunk = items[(page - 1) * 500 : page * 500]
        return httpx.Response(200, json=ok_body(chunk, total=len(items)))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = weather.fetch_month(client, SECRET, "2023-02")
    assert len(result) == 672
    assert seen_pages == [1, 2]


def test_api_error_does_not_leak_key(monkeypatch):
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)
    xml = "<OpenAPI_ServiceResponse><returnAuthMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR"

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
