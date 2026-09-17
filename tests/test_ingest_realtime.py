import json
import logging
import traceback
from datetime import UTC, date, datetime

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from httpcore import ConnectionPool
from httpcore._backends.mock import MockBackend

from bike_demand.ingest import realtime

SECRET = "fake-key/with+symbols="
NOW = datetime(2026, 9, 16, 23, 59, 59, tzinfo=realtime.KST)


def row(number=1):
    return {
        "rackTotCnt": "15",
        "stationName": "101. 대여소",
        "parkingBikeTotCnt": "0",
        "shared": "0",
        "stationLatitude": "37.5",
        "stationLongitude": "127.0",
        "stationId": f"ST-{number}",
        "unknown": {"nested": "보존"},
    }


def ok_body(rows, total=None):
    return {
        "rentBikeStatus": {
            "list_total_count": len(rows) if total is None else total,
            "RESULT": {"CODE": "INFO-000", "MESSAGE": "정상"},
            "row": rows,
            "unknown_page": True,
        },
        "unknown_top": "보존",
    }


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(realtime.time, "sleep", lambda _: None)


def test_multiple_pages_preserve_complete_responses():
    calls = []

    def handler(request):
        start, end = map(int, request.url.path.rstrip("/").split("/")[-2:])
        calls.append((start, end))
        rows = [row(n) for n in range(start, min(end, 2001) + 1)]
        return httpx.Response(200, json=ok_body(rows, 2001))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = realtime.fetch_snapshot(client, SECRET)
    assert calls == [(1, 1000), (1001, 2000), (2001, 3000)]
    assert len(result["rentBikeStatus"]["row"]) == 2001
    assert result["rentBikeStatus"]["row"][-1] == row(2001)
    assert result["pages"][1]["unknown_top"] == "보존"
    assert result["pages"][2]["rentBikeStatus"]["unknown_page"] is True


@pytest.mark.parametrize("nested", [True, False])
def test_exact_page_ends_with_no_data(nested):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if len(calls) == 1:
            return httpx.Response(200, json=ok_body([row(n) for n in range(1000)]))
        body = {"RESULT": {"CODE": "INFO-200", "MESSAGE": "데이터 없음"}}
        return httpx.Response(200, json={"rentBikeStatus": body} if nested else body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert len(realtime.fetch_snapshot(client, SECRET)["rentBikeStatus"]["row"]) == 1000
    assert len(calls) == 2


def test_retries_only_failed_page():
    calls = []

    def handler(request):
        start = int(request.url.path.rstrip("/").split("/")[-2])
        calls.append(start)
        if start == 1:
            return httpx.Response(200, json=ok_body([row(n) for n in range(1000)], 1001))
        if calls.count(1001) == 1:
            raise httpx.ConnectError(f"failed {request.url}")
        return httpx.Response(200, json=ok_body([row(1000)], 1001))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert len(realtime.fetch_snapshot(client, SECRET)["rentBikeStatus"]["row"]) == 1001
    assert calls == [1, 1001, 1001]


@pytest.mark.parametrize(
    "kind", ["http", "transport", "decode", "xml", "nested", "top", "malformed"]
)
def test_errors_and_logs_hide_key_and_url(kind, caplog):
    calls = []

    def handler(request):
        calls.append(1)
        if kind == "transport":
            raise httpx.ReadTimeout(f"failed {request.url}")
        if kind == "decode":
            raise httpx.DecodingError(f"failed {request.url}")
        if kind in {"http", "xml"}:
            return httpx.Response(
                503 if kind == "http" else 200, text=f"<error>{request.url}</error>"
            )
        body = {"RESULT": {"CODE": "ERROR-300", "MESSAGE": f"invalid {request.url}"}}
        if kind == "malformed":
            body = [str(request.url)]
        return httpx.Response(200, json={"rentBikeStatus": body} if kind == "nested" else body)

    with (
        caplog.at_level(logging.DEBUG),
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
    ):
        with pytest.raises(realtime.ApiError) as info:
            realtime.fetch_snapshot(client, SECRET, retries=2)
    rendered = "".join(traceback.format_exception(info.value)) + caplog.text
    assert SECRET not in rendered
    assert realtime.ENDPOINT not in rendered
    assert len(calls) == 2


@pytest.mark.parametrize("nested", [True, False])
def test_error_code_is_handled(nested):
    body = {"RESULT": {"CODE": "ERROR-300", "MESSAGE": "실패"}}
    response = {"rentBikeStatus": body} if nested else body
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        with pytest.raises(realtime.ApiError, match="ERROR-300"):
            realtime.fetch_snapshot(client, SECRET, retries=1)


@pytest.mark.parametrize(
    "body",
    [
        {"RESULT": {"CODE": "INFO-200", "MESSAGE": "없음"}},
        ok_body([], 0),
        ok_body([], 2),
        ok_body([row()], 2),
        [],
        {"rentBikeStatus": {"RESULT": {"CODE": "INFO-000"}, "row": "bad"}},
        ok_body([row(), row()]),
        ok_body([{"stationName": "ID 없음"}]),
    ],
)
def test_bad_or_empty_snapshot_never_published(tmp_path, body):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        with pytest.raises(realtime.ApiError):
            realtime.download(tmp_path, SECRET, NOW, client, retries=1)
    assert list(tmp_path.rglob("*.json")) == []
    assert list(tmp_path.rglob("*.tmp")) == []
    assert list(tmp_path.rglob("*.lock")) == []


def test_midstream_no_data_retries_without_saving_partial(tmp_path):
    calls = []

    def handler(request):
        start = int(request.url.path.rstrip("/").split("/")[-2])
        calls.append(start)
        body = (
            ok_body([row(n) for n in range(1000)], 2001)
            if start == 1
            else {"RESULT": {"CODE": "INFO-200"}}
        )
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(realtime.ApiError, match="전체 건수"):
            realtime.download(tmp_path, SECRET, NOW, client, retries=2)
    assert calls == [1, 1001, 1001]
    assert not list(tmp_path.rglob("*.json"))


def test_download_skip_and_daily_parquet_are_idempotent(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=ok_body([row()]))

    raw = tmp_path / "raw"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        target, status = realtime.download(raw, SECRET, NOW, client)
        before = target.read_bytes()
        assert status == "saved 1"
        assert realtime.download(raw, SECRET, NOW, client) == (target, "exists")
        assert target.read_bytes() == before
        earlier = NOW.replace(hour=10)
        realtime.download(raw, SECRET, earlier, client)
        realtime.download(raw, SECRET, NOW.replace(day=17), client)
    assert len(calls) == 3
    assert target == raw / "date=2026-09-16" / "235959.json"
    saved = json.loads(before)
    assert saved["fetched_at"] == NOW.isoformat()
    assert saved["pages"][0] == ok_body([row()])
    assert SECRET not in before.decode("utf-8")
    assert realtime.ENDPOINT not in before.decode("utf-8")
    out = tmp_path / "bronze" / "snapshots.parquet"
    for _ in range(2):
        assert realtime.to_parquet(raw, out, date(2026, 9, 16)) == 2
        table = pq.read_table(out)
        assert table.num_rows == 2
        assert all(pa.types.is_string(field.type) for field in table.schema)
        assert table.to_pylist()[0]["bike_count"] == "0"
        assert table.to_pylist()[1]["fetched_at"] == NOW.isoformat()


def test_raw_path_uses_kst_and_crosses_date(tmp_path):
    utc = datetime(2026, 9, 16, 15, 0, tzinfo=UTC)
    assert realtime.raw_path(tmp_path, utc) == tmp_path / "date=2026-09-17" / "000000.json"
    assert realtime.raw_path(tmp_path, NOW.replace(tzinfo=None)) == realtime.raw_path(tmp_path, NOW)


def test_concurrent_snapshot_lock_preserves_existing_lock(tmp_path):
    target = realtime.raw_path(tmp_path, NOW)
    target.parent.mkdir(parents=True)
    lock = target.with_suffix(".json.lock")
    lock.touch()
    with pytest.raises(realtime.ApiError, match="진행 중"):
        realtime.download(tmp_path, SECRET, NOW)
    assert lock.exists()
    assert not target.exists()


def test_atomic_write_failure_cleans_temp_and_lock(tmp_path, monkeypatch):
    def fail_replace(src, dst):
        assert src.exists()
        raise OSError("가짜 저장 실패")

    monkeypatch.setattr(realtime.os, "replace", fail_replace)
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=ok_body([row()])))
    ) as client:
        with pytest.raises(OSError, match="가짜 저장 실패"):
            realtime.download(tmp_path, SECRET, NOW, client)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


@pytest.mark.parametrize("key", [SECRET, 'fake-key\\quote"'])
@pytest.mark.parametrize("echo_url", [True, False])
def test_success_response_reflecting_key_is_not_saved(tmp_path, key, echo_url):
    def handler(request):
        payload = ok_body([row()])
        payload["request_url"] = str(request.url) if echo_url else key
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(realtime.ApiError, match="인증정보"):
            realtime.download(tmp_path, key, NOW, client, retries=1)
    assert not list(tmp_path.rglob("*.json"))


def test_unicode_escaped_key_in_decoded_payload_is_not_saved(tmp_path, caplog):
    payload = ok_body([row()])
    payload["unknown_top"] = {"nested": [{"echo": SECRET}]}
    escaped = "".join(f"\\u{ord(char):04x}" for char in SECRET)
    body = json.dumps(payload).replace(SECRET, escaped)
    assert SECRET not in body
    assert json.loads(body)["unknown_top"]["nested"][0]["echo"] == SECRET
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, text=body)

    with (
        caplog.at_level(logging.DEBUG),
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
    ):
        with pytest.raises(realtime.ApiError, match="인증정보") as info:
            realtime.download(tmp_path, SECRET, NOW, client, retries=2)
    assert calls == [1, 1]
    rendered = "".join(traceback.format_exception(info.value)) + caplog.text
    assert SECRET not in rendered and escaped not in rendered
    assert realtime.ENDPOINT not in rendered
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_failed_parquet_replace_keeps_previous_file(tmp_path, monkeypatch):
    out = tmp_path / "snapshots.parquet"
    out.write_bytes(b"previous")

    def fail_replace(src, dst):
        raise OSError("가짜 저장 실패")

    monkeypatch.setattr(realtime.os, "replace", fail_replace)
    with pytest.raises(OSError, match="가짜 저장 실패"):
        realtime.to_parquet(tmp_path, out, NOW.date())
    assert out.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.tmp"))


def test_httpcore_debug_headers_cannot_leak_request_key(caplog):
    # MockTransport가 우회하는 httpcore 헤더 로그도 오프라인 바이트 응답으로 검증한다.
    fake_url = f"{realtime.ENDPOINT}/{SECRET}/json/bikeList/1/1000/"
    wire = (f"HTTP/1.1 302 Found\r\nLocation: {fake_url}\r\nContent-Length: 0\r\n\r\n").encode()
    transport = httpx.HTTPTransport()
    transport._pool.close()
    transport._pool = ConnectionPool(network_backend=MockBackend([wire]))
    logger = logging.getLogger("httpcore.http11")
    original_filters = list(logger.filters)
    with caplog.at_level(logging.DEBUG), httpx.Client(transport=transport) as client:
        with pytest.raises(realtime.ApiError, match="302"):
            realtime.fetch_snapshot(client, SECRET, retries=1)
        logger.debug("일반 헤더 로그는 유지")
    assert logger.filters == original_filters
    assert "일반 헤더 로그는 유지" in caplog.text
    assert SECRET not in caplog.text
    assert realtime.ENDPOINT not in caplog.text


@pytest.mark.parametrize("key,retries", [("", 3), (SECRET, 0), (SECRET, True)])
def test_invalid_arguments_do_not_make_request(key, retries):
    def handler(request):
        pytest.fail("잘못된 입력에서 API 호출")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            realtime.fetch_snapshot(client, key, retries)
