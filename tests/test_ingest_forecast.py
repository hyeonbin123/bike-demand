import json
import logging
import traceback
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from httpcore import ConnectionPool
from httpcore._backends.mock import MockBackend

from bike_demand.ingest import forecast

SECRET = "fake-key+/for-tests"
BASE = datetime(2026, 9, 16, 2, tzinfo=forecast.KST)
NOW = BASE + timedelta(minutes=10)
CATEGORIES = ["TMP", "PCP", "POP", "WSD", "REH", "SKY", "PTY", "SNO"]


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(forecast.time, "sleep", lambda _: None)


def items(count, base=BASE, nx=60, ny=127):
    result = []
    for i in range(count):
        forecast_at = base + timedelta(hours=i // len(CATEGORIES) + 1)
        result.append(
            {
                "baseDate": base.strftime("%Y%m%d"),
                "baseTime": base.strftime("%H%M"),
                "category": CATEGORIES[i % len(CATEGORIES)],
                "fcstDate": forecast_at.strftime("%Y%m%d"),
                "fcstTime": forecast_at.strftime("%H%M"),
                "fcstValue": "강수없음" if i % len(CATEGORIES) == 1 else "19.5",
                "nx": nx,
                "ny": ny,
                "unknown_item": {"kept": True},
            }
        )
    return result


def ok_body(rows, total=None):
    return {
        "unknown_top": "보존",
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE", "unknown": 1},
            "body": {
                "dataType": "JSON",
                "items": {"item": rows},
                "totalCount": len(rows) if total is None else total,
                "unknown_body": "유지",
            },
        },
    }


@pytest.mark.parametrize(
    "now, expected",
    [
        ("2026-09-16T02:09:59+09:00", "2026-09-15T23:00:00+09:00"),
        ("2026-09-16T02:10:00+09:00", "2026-09-16T02:00:00+09:00"),
        ("2026-09-16T05:09:59+09:00", "2026-09-16T02:00:00+09:00"),
        ("2026-09-16T05:10:00+09:00", "2026-09-16T05:00:00+09:00"),
        ("2026-09-16T23:09:59+09:00", "2026-09-16T20:00:00+09:00"),
        ("2026-09-16T23:10:00+09:00", "2026-09-16T23:00:00+09:00"),
        ("2026-09-16T00:00:00+09:00", "2026-09-15T23:00:00+09:00"),
        ("2026-01-01T00:05:00+09:00", "2025-12-31T23:00:00+09:00"),
        ("2024-03-01T00:00:00+09:00", "2024-02-29T23:00:00+09:00"),
    ],
)
def test_latest_base_boundaries(now, expected):
    assert forecast.latest_base(datetime.fromisoformat(now)).isoformat() == expected


def test_latest_base_normalizes_utc_and_naive_kst():
    assert forecast.latest_base(NOW.astimezone(UTC)) == BASE
    assert forecast.latest_base(NOW.replace(tzinfo=None)) == BASE


def test_fetch_forecast_all_pages_preserves_unknown_fields():
    rows = items(1001)
    seen = []

    def handler(request):
        params = request.url.params
        assert dict(params) == {
            "serviceKey": SECRET,
            "pageNo": str(len(seen) + 1),
            "numOfRows": "1000",
            "dataType": "JSON",
            "base_date": "20260916",
            "base_time": "0200",
            "nx": "60",
            "ny": "127",
        }
        page = int(params["pageNo"])
        seen.append(page)
        return httpx.Response(200, json=ok_body(rows[(page - 1) * 1000 : page * 1000], 1001))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = forecast.fetch_forecast(client, SECRET, BASE)
    assert seen == [1, 2]
    assert result["items"] == rows
    assert result["pages"][0]["unknown_top"] == "보존"
    assert result["pages"][1]["response"]["body"]["unknown_body"] == "유지"
    assert result["pages"][1]["response"]["header"]["unknown"] == 1


def test_exact_full_page_does_not_request_extra_page():
    calls = []

    def handler(request):
        calls.append(request.url.params["pageNo"])
        return httpx.Response(200, json=ok_body(items(1000)))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert len(forecast.fetch_forecast(client, SECRET, BASE)["items"]) == 1000
    assert calls == ["1"]


@pytest.mark.parametrize("failure", ["empty", "short", "changed_total", "transport"])
def test_retry_only_failed_page(failure):
    rows = items(1002)
    calls = []

    def handler(request):
        page = int(request.url.params["pageNo"])
        calls.append(page)
        if len(calls) == 2:
            if failure == "transport":
                raise httpx.ConnectError(f"failed {request.url}")
            chunk = [] if failure == "empty" else rows[1000:1001]
            total = 1003 if failure == "changed_total" else 1002
            return httpx.Response(200, json=ok_body(chunk, total))
        return httpx.Response(200, json=ok_body(rows[(page - 1) * 1000 : page * 1000], 1002))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert len(forecast.fetch_forecast(client, SECRET, BASE)["items"]) == 1002
    assert calls == [1, 2, 2]


@pytest.mark.parametrize(
    "kind",
    [
        "xml",
        "message",
        "encoded_key",
        "transport",
        "http",
        "invalid_json",
        "wrong_structure",
    ],
)
def test_failures_hide_keys_in_errors_tracebacks_and_logs(kind, caplog):
    caplog.set_level(logging.DEBUG)
    calls = []

    def handler(request):
        calls.append(request)
        if kind == "transport":
            raise httpx.ConnectError(f"failed {request.url}")
        if kind == "xml":
            return httpx.Response(200, text=f"<error>{request.url}</error>")
        if kind == "http":
            return httpx.Response(503, text=str(request.url))
        if kind == "invalid_json":
            return httpx.Response(200, content=b"\xff")
        if kind == "wrong_structure":
            return httpx.Response(200, json=[])
        message = quote(SECRET, safe="") if kind == "encoded_key" else str(request.url)
        return httpx.Response(
            200,
            json={
                "response": {
                    "header": {
                        "resultCode": "30",
                        "resultMsg": message,
                    }
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(forecast.ApiError) as info:
            forecast.fetch_forecast(client, SECRET, BASE, retries=2)
    assert len(calls) == 2
    displayed = "".join(traceback.format_exception(info.value))
    for output in [str(info.value), displayed, caplog.text]:
        assert SECRET not in output
        assert quote(SECRET, safe="") not in output
        assert forecast.ENDPOINT not in output


def test_unicode_escaped_key_in_decoded_payload_is_not_saved(tmp_path, caplog):
    payload = ok_body(items(1))
    payload["unknown_top"] = {"nested": [{"echo": SECRET}]}
    escaped = "".join(f"\\u{ord(char):04x}" for char in SECRET)
    body = json.dumps(payload).replace(SECRET, escaped)
    assert SECRET not in body
    assert json.loads(body)["unknown_top"]["nested"][0]["echo"] == SECRET
    calls = []

    def handler(request):
        calls.append(request.url.params["pageNo"])
        return httpx.Response(200, text=body)

    with (
        caplog.at_level(logging.DEBUG),
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
    ):
        with pytest.raises(forecast.ApiError, match="반사") as info:
            forecast.download(tmp_path, SECRET, now=NOW, client=client, retries=2)
    assert calls == ["1", "1"]
    rendered = "".join(traceback.format_exception(info.value)) + caplog.text
    assert SECRET not in rendered and escaped not in rendered
    assert quote(SECRET, safe="") not in rendered and forecast.ENDPOINT not in rendered
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_successful_request_logs_hidden_without_changing_logger_settings(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("httpx")
    old_filters = list(logger.filters)
    old_level, old_propagate = logger.level, logger.propagate
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=ok_body(items(1))))
    ) as client:
        forecast.fetch_forecast(client, SECRET, BASE)
    assert SECRET not in caplog.text and forecast.ENDPOINT not in caplog.text
    assert logger.filters == old_filters
    assert (logger.level, logger.propagate) == (old_level, old_propagate)
    logger.info("일반 로그는 유지")
    assert "일반 로그는 유지" in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        {"response": {"header": {"resultCode": "03", "resultMsg": "NO_DATA"}}},
        {"response": {"header": {"resultCode": "30", "resultMsg": "ERROR"}}},
        ok_body([], 0),
        ok_body([], 1),
        ok_body(items(1), 2),
        ok_body(items(1), "bad"),
        ok_body(items(1), True),
    ],
)
def test_error_or_partial_response_leaves_no_complete_raw(tmp_path, payload):
    raw = tmp_path / "raw"
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(forecast.ApiError):
            forecast.download(raw, SECRET, now=NOW, client=client, retries=1)
    assert not list(raw.rglob("*.json"))
    assert not list(raw.rglob("*.tmp"))
    assert not list(raw.rglob("*.lock"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseTime", "0500"),
        ("nx", 61),
        ("fcstDate", "20260931"),
        ("fcstTime", "bad"),
        ("category", None),
        ("fcstValue", {}),
    ],
)
def test_invalid_item_is_rejected(field, value):
    rows = items(1)
    rows[0][field] = value
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=ok_body(rows)))
    ) as client:
        with pytest.raises(forecast.ApiError):
            forecast.fetch_forecast(client, SECRET, BASE, retries=1)


def test_repeated_items_are_rejected():
    row = items(1)[0]
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=ok_body([row, row])))
    ) as client:
        with pytest.raises(forecast.ApiError, match="중복"):
            forecast.fetch_forecast(client, SECRET, BASE, retries=1)


def test_download_skips_existing_and_rebuilds_string_parquet(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "bronze" / "forecast.parquet"
    calls = []
    rows = items(3)
    rows[0]["fcstValue"] = 0

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=ok_body(rows))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        target, status = forecast.download(raw, SECRET, now=NOW, client=client)
        assert status == "saved 3"
        original = target.read_bytes()
        assert forecast.download(raw, SECRET, BASE, now=NOW, client=client) == (target, "exists")
        assert target.read_bytes() == original
    assert len(calls) == 1
    assert target == raw / "nx=60_ny=127" / "base=202609160200.json"
    payload = json.loads(original)
    assert payload["fetched_at"] == "2026-09-16T02:10:00+09:00"
    assert payload["base_datetime"] == "2026-09-16T02:00:00+09:00"
    assert payload["items"][0]["unknown_item"] == {"kept": True}
    assert SECRET not in original.decode("utf-8")
    assert forecast.to_parquet(raw, out) == 3
    assert forecast.to_parquet(raw, out) == 3
    table = pq.read_table(out)
    assert table.schema == forecast.SCHEMA
    assert all(pa.types.is_string(field.type) for field in table.schema)
    result = table.to_pylist()
    assert result[0]["value"] == "0"
    assert result[1]["category"] == "PCP" and result[1]["value"] == "강수없음"
    assert result[0]["fcst_datetime"] == "2026-09-16T03:00:00+09:00"
    assert result[0]["nx"] == "60" and result[0]["ny"] == "127"
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.lock"))


def test_parquet_keeps_multiple_publications_and_grids(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "forecast.parquet"

    def handler(request):
        params = request.url.params
        base = datetime.strptime(params["base_date"] + params["base_time"], "%Y%m%d%H%M")
        return httpx.Response(
            200, json=ok_body(items(2, base=base, nx=int(params["nx"]), ny=int(params["ny"])))
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        forecast.download(raw, SECRET, BASE, now=NOW, client=client)
        forecast.download(raw, SECRET, BASE - timedelta(hours=3), now=NOW, client=client)
        forecast.download(raw, SECRET, BASE, now=NOW, client=client, nx=61)
    assert forecast.to_parquet(raw, out) == 6
    result = pq.read_table(out).to_pylist()
    assert len({row["base_datetime"] for row in result}) == 2
    assert {row["nx"] for row in result} == {"60", "61"}


def test_concurrent_download_does_not_overwrite_or_remove_other_lock(tmp_path):
    target = forecast.raw_path(tmp_path, BASE)
    target.parent.mkdir(parents=True)
    lock = target.with_suffix(".json.lock")
    lock.write_text("owned by another collector", encoding="utf-8")
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("API 호출 금지"))
    ) as client:
        with pytest.raises(forecast.ApiError, match="다른 수집기"):
            forecast.download(tmp_path, SECRET, now=NOW, client=client)
    assert lock.read_text(encoding="utf-8") == "owned by another collector"
    assert not target.exists()


def test_atomic_replace_failure_preserves_prior_parquet_and_cleans_temporary(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "forecast.parquet"
    out.write_bytes(b"previous complete file")

    def fail_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(forecast.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        forecast.to_parquet(raw, out)
    assert out.read_bytes() == b"previous complete file"
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize(
    "base",
    [
        BASE.replace(hour=3),
        BASE.replace(minute=1),
        BASE.replace(second=1),
        BASE.replace(microsecond=1),
        BASE + timedelta(hours=3),
    ],
)
def test_invalid_or_not_available_base_never_calls_api(tmp_path, base):
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("API 호출 금지"))
    ) as client:
        with pytest.raises(ValueError):
            forecast.download(tmp_path, SECRET, base, now=NOW, client=client)


@pytest.mark.parametrize("kwargs", [{"retries": 0}, {"nx": 0}, {"ny": 254}])
def test_invalid_parameters_never_call_api(kwargs):
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("API 호출 금지"))
    ) as client:
        with pytest.raises(ValueError):
            forecast.fetch_forecast(client, SECRET, BASE, **kwargs)


@pytest.mark.parametrize("secret", [SECRET, 'fake\\key"for-tests'])
def test_success_payload_reflecting_key_is_not_saved(tmp_path, secret):
    payload = ok_body(items(1))
    payload["unknown_top"] = secret
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(forecast.ApiError, match="인증정보"):
            forecast.download(tmp_path, secret, now=NOW, client=client, retries=1)
    assert not list(tmp_path.rglob("*.json"))


def test_second_page_exhaustion_leaves_no_partial_raw(tmp_path):
    rows = items(1001)
    calls = []

    def handler(request):
        page = int(request.url.params["pageNo"])
        calls.append(page)
        return httpx.Response(200, json=ok_body(rows[:1000] if page == 1 else [], 1001))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(forecast.ApiError, match="2쪽 수집 실패"):
            forecast.download(tmp_path, SECRET, now=NOW, client=client, retries=2)
    assert calls == [1, 2, 2]
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.lock"))


def test_atomic_raw_replace_failure_cleans_lock_and_temporary(tmp_path, monkeypatch):
    def fail_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(forecast.os, "replace", fail_replace)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=ok_body(items(1))))
    ) as client:
        with pytest.raises(OSError, match="simulated"):
            forecast.download(tmp_path, SECRET, now=NOW, client=client)
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.lock"))


@pytest.mark.parametrize(
    "date,clock",
    [
        ("2026091", "60200"),
        ("202609160", "200"),
        ("20260916", "2:00"),
    ],
)
def test_date_and_time_need_separate_fixed_width(date, clock):
    with pytest.raises(forecast.ApiError, match="시각 형식"):
        forecast._datetime(date, clock)


def test_httpcore_debug_headers_cannot_leak_request_key(caplog):
    # 실제 네트워크 대신 MockBackend로 httpcore의 DEBUG 응답 헤더 경로를 통과한다.
    fake_url = f"{forecast.ENDPOINT}?serviceKey={SECRET}"
    wire = (f"HTTP/1.1 302 Found\r\nLocation: {fake_url}\r\nContent-Length: 0\r\n\r\n").encode()
    transport = httpx.HTTPTransport()
    transport._pool.close()
    transport._pool = ConnectionPool(network_backend=MockBackend([wire]))
    with caplog.at_level(logging.DEBUG), httpx.Client(transport=transport) as client:
        with pytest.raises(forecast.ApiError, match="302"):
            forecast.fetch_forecast(client, SECRET, BASE, retries=1)
    assert SECRET not in caplog.text
    assert forecast.ENDPOINT not in caplog.text


def test_recent_bases_cover_the_last_day_oldest_first():
    """PC가 밤새 꺼졌다 켜져도 하루 안쪽의 빠진 발표를 채운다."""
    now = datetime(2026, 9, 18, 8, 20, tzinfo=forecast.KST)
    bases = forecast.recent_bases(now)
    assert [f"{b:%d %H}" for b in bases] == [
        "17 11", "17 14", "17 17", "17 20", "17 23", "18 02", "18 05", "18 08",
    ]  # fmt: skip
    # 제공 10분 전이면 최신은 직전 발표
    assert forecast.recent_bases(datetime(2026, 9, 18, 8, 5, tzinfo=forecast.KST))[-1].hour == 5


def test_download_recent_skips_existing_and_only_fails_on_the_latest(tmp_path, monkeypatch):
    now = datetime(2026, 9, 18, 8, 20, tzinfo=forecast.KST)
    seen = []

    def fake_download(raw_dir, key, base, *, now, client, nx, ny):
        seen.append(base.hour)
        if base.hour == 2:
            raise forecast.ApiError("지난 발표 실패")
        return raw_dir / "x.json", "exists" if base.day == 17 else "saved 900"

    monkeypatch.setattr(forecast, "download", fake_download)
    results = forecast.download_recent(tmp_path, SECRET, now=now)
    assert len(seen) == 8 and seen[-1] == 8
    assert dict((b.hour, s) for b, s in results)[2].startswith("failed")
    assert results[-1][1] == "saved 900"

    def latest_fails(raw_dir, key, base, *, now, client, nx, ny):
        if base.hour == 8:
            raise forecast.ApiError("최신 발표 실패")
        return raw_dir / "x.json", "exists"

    monkeypatch.setattr(forecast, "download", latest_fails)
    with pytest.raises(forecast.ApiError, match="최신"):
        forecast.download_recent(tmp_path, SECRET, now=now)
