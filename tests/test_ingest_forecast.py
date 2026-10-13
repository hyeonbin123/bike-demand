# [무엇을 검증하나] 기상청 단기예보 수집(src/bike_demand/ingest/forecast.py)을 실제 네트워크 없이
# 확인한다.
# - 발표 시각 고르기: 발표 10분 뒤부터 받을 수 있다는 가정, 날짜·연도·윤년 경계, 하루 안쪽 빠진 발표
#   채우기
# - 여러 쪽 받기: 쪽마다 따로 재시도, 전체 건수와 쪽 크기가 맞는지, 중복·형식 오류 항목 거부, 응답의
#   모르는 필드 보존
# - 저장의 원자성: 실패하면 완성 JSON·임시 파일·잠금 파일을 남기지 않고, 다른 수집기의 잠금을 지우지
#   않는다
# - 인증키 보호: 오류 메시지·traceback·로그·저장 파일 어디에도 키(그대로, URL 인코딩, JSON
#   이스케이프)와 키가 붙는 API 주소가 나오지 않는다
# [왜 중요한가] 공공 API 키는 URL 쿼리에 들어가므로 httpx 로그나 오류 메시지에 쉽게 섞인다(README
# 설계 판단 '인증키가 새지 않게'). 또 예보 API는 지난 24시간 발표만 주므로, 반쯤 저장된 파일이 '이미
# 있음'으로 건너뛰어지면 그 발표는 되살릴 수 없다.
# [방식] httpx.MockTransport로 응답을 만들어 요청 인자와 실패를 흉내 내고, httpcore 로그 경로는
# MockBackend로 실제 HTTP 바이트를 흘려 확인한다.

import json
import logging
import traceback
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

# MockTransport: 가짜 HTTP 응답. httpcore ConnectionPool·MockBackend: httpx 아래 계층에 미리 정한
# 응답 바이트를 넣어 httpcore의 DEBUG 로그까지 실제와 같이 지나가게 한다. pyarrow: 만들어진 Parquet
# 확인.
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from httpcore import ConnectionPool
from httpcore._backends.mock import MockBackend

from bike_demand.ingest import forecast

# 가짜 키에 '+'와 '/'를 넣어, URL 인코딩되면 모양이 달라지는 경우(%2B, %2F)까지 숨기는지 보게 한다.
# BASE는 02시 발표, NOW는 그 발표를 막 받을 수 있게 된 02:10. CATEGORIES는 단기예보 항목 8개.
SECRET = "fake-key+/for-tests"
BASE = datetime(2026, 9, 16, 2, tzinfo=forecast.KST)
NOW = BASE + timedelta(minutes=10)
CATEGORIES = ["TMP", "PCP", "POP", "WSD", "REH", "SKY", "PTY", "SNO"]


# 재시도 사이 대기(2초, 4초…)를 없애 테스트가 빨리 끝나게 한다. 이 파일의 모든 테스트에 자동 적용.
@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(forecast.time, "sleep", lambda _: None)


# count개의 예보 항목을 만든다. 예보 시각 하나에 항목 8개씩 채우고, PCP는 '강수없음'(문자열 값),
# 나머지는 '19.5'. API 문서에 없는 필드(unknown_item)도 넣어 원본에 그대로 남는지 본다.
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


# 정상 응답(resultCode 00) 봉투. 맨 위·header·body에 모르는 필드를 하나씩 넣어 원본 보존을 확인하고,
# totalCount를 따로 줄 수 있어 '전체 건수와 실제 항목 수가 다른' 응답도 만든다.
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


# 발표(02·05·…·23시)는 10분 뒤부터 받을 수 있다고 본다. 02:09:59에는 아직 전날 23시, 02:10:00부터
# 02시 발표. 자정 직후에는 전날 23시, 새해 첫날은 전년 12-31, 윤년 3월 1일 0시는 2월 29일 23시.
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


# UTC 시각은 KST로 바꿔 계산하고, 시간대 없는 시각은 서울 현지 시각으로 해석한다(Airflow·테스트에서
# 들어오는 시각 형식이 달라도 같은 발표를 고르게).
def test_latest_base_normalizes_utc_and_naive_kst():
    assert forecast.latest_base(NOW.astimezone(UTC)) == BASE
    assert forecast.latest_base(NOW.replace(tzinfo=None)) == BASE


# 1,001건 발표는 1,000건씩 두 쪽으로 받는다. 요청 인자가 정확히 계약대로(서비스키, 쪽 번호, 쪽 크기
# 1000, JSON, 발표 날짜·시각, 격자)인지, 항목이 순서대로 모두 합쳐지는지, 쪽별 응답 전체가 pages에
# 남아 모르는 필드도 보존되는지 확인한다.
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


# 전체가 정확히 1,000건이면 2쪽을 요청하지 않는다(빈 쪽을 받으면 '불완전한 쪽'으로 실패하게 된다).
def test_exact_full_page_does_not_request_extra_page():
    calls = []

    def handler(request):
        calls.append(request.url.params["pageNo"])
        return httpx.Response(200, json=ok_body(items(1000)))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert len(forecast.fetch_forecast(client, SECRET, BASE)["items"]) == 1000
    assert calls == ["1"]


# 2쪽 첫 요청만 실패(빈 쪽, 모자란 쪽, 쪽 사이 전체 건수 변경, 연결 오류)하게 만들면 그 쪽만 다시
# 요청해 성공한다. 1쪽부터 다시 받지 않으므로 호출은 [1, 2, 2].
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


# 실패 종류마다 서버가 요청 URL(키 포함)이나 인코딩된 키를 응답·예외에 되돌려 보내게 만든다: XML
# 오류 본문, 오류 메시지, 인코딩된 키, 연결 오류, HTTP 503, JSON 아님, 구조 틀림. 재시도
# 횟수만큼(2번) 호출한 뒤 ApiError를 내고, 그 메시지·traceback 전체·DEBUG 로그 어디에도 키와 API
# 주소가 없어야 한다. ApiError는 원인 예외를 끊고(from None) 본문·URL을 담지 않아 이것이 가능하다.
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


# 응답 본문에 키가 \uXXXX로 이스케이프돼 있으면 원문 문자열에는 키가 보이지 않지만 JSON을 풀면 키가
# 된다. 수집기가 푼 결과를 다시 직렬화해 검사하므로 이 경우도 '반사'로 거부하고, 아무 파일도
# 저장하지 않으며, 오류·로그에도 어느 형태의 키도 남지 않는지 확인한다.
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


# 성공한 요청에서도 httpx가 INFO로 남기는 요청 줄(키가 든 URL)이 보이지 않아야 한다. 동시에
# private_request가 로거의 필터·레벨·전파 설정을 끝난 뒤 원래대로 돌려놓아, 요청 밖의 일반 httpx
# 로그는 그대로 남는지 본다(전역 로그 설정을 바꾸지 않는 방식).
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


# 오류·불완전 응답: 자료 없음(03), 기타 오류 코드(30), 전체 0건, 전체 1건인데 항목 0개, 전체 2건인데
# 1개, 전체 건수가 숫자가 아님, 불리언(JSON true는 파이썬에서 정수처럼 보여 따로 막는다). 모두
# ApiError이고 원본 폴더에 완성 JSON·임시 파일·잠금 파일이 하나도 남지 않아야 한다. 반쯤 받은 원본이
# 남으면 다음 실행이 '이미 있음'으로 건너뛴다.
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


# 항목 하나라도 요청과 다른 발표 시각·격자, 존재하지 않는 날짜(9월 31일), 형식이 틀린 시각, 빈 항목
# 이름, 값이 객체이면 그 발표 전체를 거부한다. 잘못된 항목이 섞인 채 저장되면 예측에 엉뚱한 날씨가
# 들어간다.
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


# 같은 (예보 시각, 항목)이 두 번 오면 '중복'으로 거부한다. API가 같은 쪽을 반복해 보낸 경우를
# 잡는다.
def test_repeated_items_are_rejected():
    row = items(1)[0]
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=ok_body([row, row])))
    ) as client:
        with pytest.raises(forecast.ApiError, match="중복"):
            forecast.fetch_forecast(client, SECRET, BASE, retries=1)


# 저장과 멱등성: 첫 다운로드는 'saved 3', 같은 발표를 다시 부르면 API를 부르지 않고 'exists'이며
# 파일 바이트도 그대로다. 저장 경로(nx=60_ny=127/base=YYYYMMDDHHMM.json), KST 수집·발표 시각, 모르는
# 필드 보존, 파일에 키가 없음을 확인한다.
# Parquet은 두 번 만들어도 같은 3행이고(통째로 다시 만듦), 모든 열이 문자열이다. 숫자 0으로 온 값도
# '0' 문자열로 남고, '강수없음' 같은 값은 해석하지 않고 그대로 둔다(해석은
# model/forecast_weather.py).
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


# 발표 두 개(02시, 전날 23시)와 격자 두 개(nx 60, 61)를 받으면 Parquet에 6행이 모두 남는다. 발표나
# 격자를 덮어쓰지 않아야 v4 측정처럼 발표별 예보 오차를 잴 수 있다.
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


# 다른 수집기가 같은 발표의 잠금 파일을 들고 있으면 API를 부르지 않고 '다른 수집기' 오류를 낸다.
# 이때 남의 잠금 파일을 지우거나 내용을 바꾸지 않는다(finally에서 잠금을 지우는 것은 자기가 만든
# 경우뿐).
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


# Parquet 교체(os.replace)가 실패해도 이전 완성 파일은 그대로 남고 임시 파일은 지워진다. 서비스
# 적재가 반쯤 쓴 Parquet을 읽는 일을 막는다.
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


# 발표 시각이 아닌 값(03시, 분·초·마이크로초가 붙은 값)과 아직 제공되지 않은 발표(02:10에 05시
# 발표)는 API를 부르기 전에 ValueError.
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


# 재시도 0번, 격자 범위 밖(nx 1~149, ny 1~253)은 요청 전에 ValueError.
@pytest.mark.parametrize("kwargs", [{"retries": 0}, {"nx": 0}, {"ny": 254}])
def test_invalid_parameters_never_call_api(kwargs):
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("API 호출 금지"))
    ) as client:
        with pytest.raises(ValueError):
            forecast.fetch_forecast(client, SECRET, BASE, **kwargs)


# 정상 응답이라도 어딘가에 키를 되돌려 보내면 저장하지 않는다('인증정보'). 역슬래시·큰따옴표가 든
# 키는 JSON 안에서 이스케이프된 모양으로 나타나므로 그 형태도 찾는지 확인한다.
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


# 1쪽은 성공하고 2쪽이 재시도를 다 써도 빈 쪽이면 '2쪽 수집 실패'로 끝나고, 1쪽만 담은 부분 원본을
# 남기지 않는다.
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


# 원본 JSON을 정식 이름으로 바꾸는 단계가 실패해도 임시 파일과 잠금 파일을 정리한다. 잠금이 남으면
# 다음 수집이 '다른 수집기가 처리 중'으로 계속 막힌다.
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


# 날짜 8자리·시각 4자리를 따로 검사하는지 본다. 세 경우 모두 이어 붙이면 '202609160200'으로 같아
# 보이지만, 칸이 어긋난 값은 '시각 형식' 오류여야 한다.
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


# httpcore는 DEBUG에서 응답 헤더를 로그로 남긴다. 서버가 Location 헤더에 키가 든 URL을 돌려주는 302
# 응답을 MockBackend로 흘려 그 로그 경로를 실제로 지나가게 하고, 로그에 키·주소가 없는지와 302를
# 오류로 처리하는지 확인한다. transport의 기본 풀을 닫고 가짜 백엔드 풀로 바꿔 끼운다.
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


# 08:20이면 최신은 08시 발표이고, 24시간 안쪽(전날 08시는 정확히 24시간이라 제외)의 발표 8개를
# 오래된 것부터 돌려준다. 08:05에는 08시 발표가 아직 아니라 최신이 05시.
def test_recent_bases_cover_the_last_day_oldest_first():
    """PC가 밤새 꺼졌다 켜져도 하루 안쪽의 빠진 발표를 채운다."""
    now = datetime(2026, 9, 18, 8, 20, tzinfo=forecast.KST)
    bases = forecast.recent_bases(now)
    assert [f"{b:%d %H}" for b in bases] == [
        "17 11", "17 14", "17 17", "17 20", "17 23", "18 02", "18 05", "18 08",
    ]  # fmt: skip
    # 제공 10분 전이면 최신은 직전 발표
    assert forecast.recent_bases(datetime(2026, 9, 18, 8, 5, tzinfo=forecast.KST))[-1].hour == 5


# download를 가짜로 바꿔 순서와 실패 처리만 본다. 지난 발표(02시)가 실패하면 'failed …'로 적고
# 넘어가 나머지와 최신(08시)을 받는다. 최신 발표가 실패하면 예외를 그대로 낸다(최신 예보는 서비스
# 예측에 필요하므로, download_recent docstring).
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
