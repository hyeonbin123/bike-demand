# [무엇을 검증하나] 따릉이 실시간 대여정보 수집(src/bike_demand/ingest/realtime.py)을 네트워크 없이
# 확인한다.
# - bikeList를 1~1000, 1001~2000… 순서로 끝까지 받아 한 시점의 스냅샷으로 합치고, 실패한 쪽만
#   재시도한다
# - 오류·빈 응답·쪽이 알린 건수(list_total_count)와 행 수가 다른 응답·중복 대여소는 원본으로
#   저장하지 않는다(완성 JSON·임시·잠금 파일 모두 남지 않음)
# - 잘린 스냅샷(짧은 쪽이나 INFO-200이 중간에 와서 일찍 끝남)은 쪽 단위로는 정상 끝과 구분되지
#   않으므로, 직전 60분 안에 저장한 원본보다 대여소가 5% 넘게 적으면 스냅샷 전체를 다시 받고 끝내
#   모자라면 저장하지 않는다. 60분 안 원본이 없으면 비교하지 않는다
# - 원본 경로는 수집 시각의 KST 날짜·초로 정하고, 같은 초를 다시 받으면 덮어쓰지 않는다. 날짜별
#   Parquet은 원본 전체로 다시 만들어 멱등이다
# - 인증키는 URL 경로에 들어가므로 오류·traceback·로그·저장 파일 어디에도 키와 API 주소가 나오지
#   않는다
# [왜 중요한가] 이 스냅샷이 부족 위험의 '지금 자전거 수'이고, 10분마다 쌓여 v5 측정의 정답(0대
# 도달)도 된다. 일부 쪽만 받은 스냅샷이 저장되면 그 시점 대여소 일부가 조용히 빠진다.

import json
import logging
import traceback
from datetime import UTC, date, datetime, timedelta

# httpx.MockTransport로 응답을 만들고, httpcore MockBackend로 실제 HTTP 바이트를 흘려 httpcore 헤더
# 로그 경로까지 확인한다.
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from httpcore import ConnectionPool
from httpcore._backends.mock import MockBackend

from bike_demand.ingest import realtime

# 가짜 키에 '/', '+', '='를 넣어 URL 인코딩되면 모양이 달라지는 경우까지 숨기는지 본다. NOW는 자정
# 1초 전(23:59:59)이라 날짜 경계를 같이 확인한다.
SECRET = "fake-key/with+symbols="
NOW = datetime(2026, 9, 16, 23, 59, 59, tzinfo=realtime.KST)


# API 행 하나. 값은 모두 문자열(API 그대로), 자전거 0대, 문서에 없는 필드(unknown)를 넣어 원본
# 보존을 본다.
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


# 정상 응답(INFO-000) 봉투. 기본은 실제 API 형식이다: list_total_count는 전체 대여소 수가 아니라
# 그 쪽의 행 수다. 2026-09-16~10-03에 저장한 원본 570개의 모든 쪽이 이 형식이었고, 가장 흔한 쪽
# 모양은 (list_total_count, 행 수) = (1000, 1000), (1000, 1000), (734, 734)였다. total로 다른 값을
# 주면 쪽이 알린 건수와 행 수가 다른(잘렸거나 깨진) 쪽을 만든다.
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


# 마지막 다음 쪽에 오는 '해당 데이터 없음' 응답(맨 위에 RESULT만 있는 형태).
NO_DATA = {"RESULT": {"CODE": "INFO-200", "MESSAGE": "해당 데이터가 없습니다."}}


# 대여소 count곳을 실제 형식으로 쪽마다 나눠 주는 가짜 bikeList. 요청 범위(경로 끝 두 숫자)의 행만
# 주고, 대여소가 다 끝난 범위에는 INFO-200을 준다. calls에 요청한 시작 번호를 모은다.
def station_pages(count, calls=None):
    def handler(request):
        start, end = map(int, request.url.path.rstrip("/").split("/")[-2:])
        if calls is not None:
            calls.append(start)
        if start > count:
            return httpx.Response(200, json=NO_DATA)
        return httpx.Response(
            200, json=ok_body([row(n) for n in range(start, min(end, count) + 1)])
        )

    return handler


# raw 폴더에 at 시각의 대여소 count곳 원본을 실제 수집 경로(download)로 저장해, 잘림 검사의 기준이
# 될 '직전 원본'을 만든다.
def save_snapshot(raw, at, count):
    with httpx.Client(transport=httpx.MockTransport(station_pages(count))) as client:
        path, status = realtime.download(raw, SECRET, at, client)
    assert status == f"saved {count}"
    return path


# 재시도 사이 대기를 없앤다(모든 테스트에 자동 적용).
@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(realtime.time, "sleep", lambda _: None)


# 2,001곳이면 세 쪽(1~1000, 1001~2000, 2001~3000)을 요청하고, 마지막 짧은 쪽에서 멈춘다. 쪽은 실제
# 형식(list_total_count = 그 쪽 행 수: 1000, 1000, 1)이다. 행은 순서대로 모두 합쳐지고, 쪽마다 응답
# 전체가 pages에 남아 모르는 필드도 보존된다. 요청 범위는 URL 경로의 끝 두 숫자로 읽는다.
def test_multiple_pages_preserve_complete_responses():
    calls = []

    def handler(request):
        start, end = map(int, request.url.path.rstrip("/").split("/")[-2:])
        calls.append((start, end))
        rows = [row(n) for n in range(start, min(end, 2001) + 1)]
        return httpx.Response(200, json=ok_body(rows))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = realtime.fetch_snapshot(client, SECRET)
    assert calls == [(1, 1000), (1001, 2000), (2001, 3000)]
    assert len(result["rentBikeStatus"]["row"]) == 2001
    assert result["rentBikeStatus"]["row"][-1] == row(2001)
    assert [p["rentBikeStatus"]["list_total_count"] for p in result["pages"]] == [1000, 1000, 1]
    assert result["pages"][1]["unknown_top"] == "보존"
    assert result["pages"][2]["rentBikeStatus"]["unknown_page"] is True


# 정확히 1,000곳이면 다음 쪽에서 INFO-200(데이터 없음)이 와야 끝을 안다. INFO-200은 rentBikeStatus
# 안에 오기도, 맨 위에 오기도 하므로 두 형태 모두 '정상 끝'으로 처리하는지 본다.
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


# 1001행부터의 쪽이 한 번 연결 오류를 내면 그 쪽만 다시 받는다(1쪽을 다시 받지 않음). 호출은 [1,
# 1001, 1001]. 쪽은 실제 형식(1000곳 쪽은 1000, 마지막 1곳 쪽은 1)이다.
def test_retries_only_failed_page():
    calls = []

    def handler(request):
        start = int(request.url.path.rstrip("/").split("/")[-2])
        calls.append(start)
        if start == 1:
            return httpx.Response(200, json=ok_body([row(n) for n in range(1000)]))
        if calls.count(1001) == 1:
            raise httpx.ConnectError(f"failed {request.url}")
        return httpx.Response(200, json=ok_body([row(1000)]))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert len(realtime.fetch_snapshot(client, SECRET)["rentBikeStatus"]["row"]) == 1001
    assert calls == [1, 1001, 1001]


# 실패 종류마다(HTTP 503, 시간 초과, 디코딩 오류, XML 본문, 오류 코드가 안쪽·바깥에 있는 JSON,
# 구조가 틀린 JSON) 서버가 요청 URL을 돌려보내게 만든다. 재시도 2번 뒤 ApiError가 나고, traceback
# 전체와 DEBUG 로그에 키·API 주소가 없어야 한다.
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


# API 오류 코드(ERROR-300)는 위치와 상관없이 ApiError가 되고, 형식이 맞는 코드는 메시지에 남겨
# 원인을 알 수 있게 한다.
@pytest.mark.parametrize("nested", [True, False])
def test_error_code_is_handled(nested):
    body = {"RESULT": {"CODE": "ERROR-300", "MESSAGE": "실패"}}
    response = {"rentBikeStatus": body} if nested else body
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        with pytest.raises(realtime.ApiError, match="ERROR-300"):
            realtime.fetch_snapshot(client, SECRET, retries=1)


# 저장하면 안 되는 응답들: 첫 쪽부터 데이터 없음, 0건, 쪽이 2건이라고 알렸는데 행은 0개·1개,
# 최상위가 목록, row가 목록이 아님, 같은 대여소ID 두 번, 대여소ID 없음. 모두 ApiError이고 원본
# 폴더에 아무 파일도(.json, .tmp, .lock) 남지 않는다.
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


# 쪽이 알린 건수와 실제 행 수가 다르면 그 쪽을 다시 받고, 끝내 다르면 저장하지 않는다. (1) 한 쪽뿐인
# 응답이 행 5개를 주면서 3이라고 알림 (2) 2쪽이 1000이라고 알리면서 734행만 줌(전송 중 잘린 쪽).
# 고치기 전 코드는 이 값을 전체 건수로 읽어 두 경우 모두 그대로 저장했다.
@pytest.mark.parametrize(
    "pages, expected_calls",
    [
        ({1: ok_body([row(n) for n in range(1, 6)], 3)}, [1, 1]),
        (
            {
                1: ok_body([row(n) for n in range(1, 1001)]),
                1001: ok_body([row(n) for n in range(1001, 1735)], 1000),
            },
            [1, 1001, 1001],
        ),
    ],
)
def test_page_whose_count_differs_from_its_rows_is_retried_then_not_saved(
    tmp_path, pages, expected_calls
):
    calls = []

    def handler(request):
        start = int(request.url.path.rstrip("/").split("/")[-2])
        calls.append(start)
        return httpx.Response(200, json=pages[start])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(realtime.ApiError, match="건수 불일치"):
            realtime.download(tmp_path, SECRET, NOW, client, retries=2)
    assert calls == expected_calls
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


# 잘린 스냅샷: 직전 원본은 2,001곳인데 이번에는 2쪽이 짧게(600행) 오거나 2쪽에서 INFO-200이 와서
# 1,600곳·1,000곳에서 끝난다. 실제 형식에서는 짧은 쪽·INFO-200이 정상 끝과 똑같이 생겨 쪽
# 단위로는 알 수 없으므로, 직전 60분 안 원본 대여소 수의 95% 미만이면 스냅샷 전체를 retries번 다시
# 받고 끝내 모자라면 저장하지 않는다(새 원본·임시·잠금 파일 없음, 직전 원본은 그대로). 기준 원본이
# 10분 전·정확히 60분 전·자정 너머 전날 폴더(00:05 수집, 기준 23:55)에 있는 세 경우를 본다. 고치기
# 전 코드는 잘린 스냅샷을 '완료'로 저장했다.
@pytest.mark.parametrize("cut", ["short_page", "no_data"])
@pytest.mark.parametrize(
    "at, before",
    [
        (NOW, timedelta(minutes=10)),
        (NOW, timedelta(minutes=60)),
        (datetime(2026, 9, 17, 0, 5, tzinfo=realtime.KST), timedelta(minutes=10)),
    ],
)
def test_truncated_snapshot_is_refetched_then_not_saved(tmp_path, cut, at, before):
    previous = save_snapshot(tmp_path, at - before, 2001)
    calls = []

    def handler(request):
        start = int(request.url.path.rstrip("/").split("/")[-2])
        calls.append(start)
        if start == 1:
            return httpx.Response(200, json=ok_body([row(n) for n in range(1, 1001)]))
        if cut == "short_page":
            return httpx.Response(200, json=ok_body([row(n) for n in range(1001, 1601)]))
        return httpx.Response(200, json=NO_DATA)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(realtime.ApiError, match="직전"):
            realtime.download(tmp_path, SECRET, at, client, retries=2)
    assert calls == [1, 1001, 1, 1001]
    assert sorted(tmp_path.rglob("*.json")) == [previous]
    assert not list(tmp_path.rglob("*.lock")) and not list(tmp_path.rglob("*.tmp"))


# 첫 시도만 잘리고(2쪽 600행) 다시 받은 스냅샷이 온전하면 그것을 저장한다(2,001곳, 호출은 첫 시도 두
# 쪽 + 다시 받은 세 쪽).
def test_truncated_snapshot_recovers_when_refetched(tmp_path):
    save_snapshot(tmp_path, NOW - timedelta(minutes=10), 2001)
    calls = []
    full = station_pages(2001)

    def handler(request):
        start = int(request.url.path.rstrip("/").split("/")[-2])
        calls.append(start)
        if start == 1001 and calls.count(1001) == 1:
            return httpx.Response(200, json=ok_body([row(n) for n in range(1001, 1601)]))
        return full(request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        target, status = realtime.download(tmp_path, SECRET, NOW, client, retries=2)
    assert status == "saved 2001"
    assert calls == [1, 1001, 1, 1001, 2001]
    assert len(json.loads(target.read_text(encoding="utf-8"))["rentBikeStatus"]["row"]) == 2001


# 95% 경계: 직전 원본 2,001곳의 95%는 1,900.95곳이라 1,901곳은 저장하고 1,900곳은 거부한다.
@pytest.mark.parametrize("count, saved", [(1901, True), (1900, False)])
def test_truncation_threshold_is_95_percent_of_previous_snapshot(tmp_path, count, saved):
    save_snapshot(tmp_path, NOW - timedelta(minutes=10), 2001)
    with httpx.Client(transport=httpx.MockTransport(station_pages(count))) as client:
        if saved:
            assert (
                realtime.download(tmp_path, SECRET, NOW, client, retries=1)[1] == f"saved {count}"
            )
        else:
            with pytest.raises(realtime.ApiError, match="95%"):
                realtime.download(tmp_path, SECRET, NOW, client, retries=1)
    assert realtime.raw_path(tmp_path, NOW).exists() == saved


# 비교할 기준이 없으면 대여소 수와 상관없이 저장한다: 첫 실행(원본 없음), 61분 전 원본만
# 있음(60분 창 밖), 수집 시각보다 뒤의 원본만 있음(시각을 지정해 지난 시각을 손으로 받는 경우).
# 정당하게 대여소가 크게 줄어도 직전 원본이 60분 창을 벗어나면 다시 저장되기 시작한다는 뜻이기도
# 하다.
@pytest.mark.parametrize("offset", [None, timedelta(minutes=-61), timedelta(minutes=5)])
def test_snapshot_without_recent_previous_snapshot_is_saved(tmp_path, offset):
    if offset is not None:
        save_snapshot(tmp_path, NOW + offset, 2001)
    with httpx.Client(transport=httpx.MockTransport(station_pages(1600))) as client:
        assert realtime.download(tmp_path, SECRET, NOW, client)[1] == "saved 1600"


# 대여소가 정확히 1,000곳이면 1쪽이 꽉 차고 2쪽 INFO-200으로 끝난다. 직전 원본도 1,000곳이면 줄지
# 않았으므로 그대로 저장한다(호출 [1, 1001], 다시 받지 않음).
def test_exact_page_snapshot_matching_previous_count_is_saved(tmp_path):
    save_snapshot(tmp_path, NOW - timedelta(minutes=10), 1000)
    calls = []
    with httpx.Client(transport=httpx.MockTransport(station_pages(1000, calls))) as client:
        assert realtime.download(tmp_path, SECRET, NOW, client)[1] == "saved 1000"
    assert calls == [1, 1001]


# 기준을 고를 때 읽을 수 없는 원본(JSON이 아님)은 건너뛰고 그보다 이른 60분 안 원본을 쓴다. 20분 전
# 2,001곳 원본 + 10분 전 깨진 파일이면 기준은 2,001곳이라 1,600곳 스냅샷은 저장하지 않는다.
def test_unreadable_recent_snapshot_is_skipped_when_choosing_the_baseline(tmp_path):
    save_snapshot(tmp_path, NOW - timedelta(minutes=20), 2001)
    broken = realtime.raw_path(tmp_path, NOW - timedelta(minutes=10))
    broken.write_text("{잘린 파일", encoding="utf-8")
    with httpx.Client(transport=httpx.MockTransport(station_pages(1600))) as client:
        with pytest.raises(realtime.ApiError, match="2001곳"):
            realtime.download(tmp_path, SECRET, NOW, client, retries=1)
    assert not realtime.raw_path(tmp_path, NOW).exists()


# 저장과 멱등성. 같은 초(NOW)를 다시 받으면 API 호출 없이 'exists'이고 파일도 그대로다. 같은 날 다른
# 시각(10:59:59)과 다음 날 스냅샷을 더 받으면 호출은 모두 3번.
# 원본 경로는 date=2026-09-16/235959.json, fetched_at은 수집 시각 그대로, pages에 응답 전체가 남고
# 키·주소는 없다. 09-16의 Parquet을 두 번 만들어도 2행(다음 날 파일은 섞이지 않음)이고 모든 열이
# 문자열이다(파일 이름 순서라 10시 스냅샷이 첫 행).
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


# UTC 15:00은 KST 다음 날 0시라 date=2026-09-17로 간다. 시간대 없는 시각은 KST로 본다. 적재 날짜를
# 수집 시각의 KST 날짜로 정하는 규칙의 바탕이다.
def test_raw_path_uses_kst_and_crosses_date(tmp_path):
    utc = datetime(2026, 9, 16, 15, 0, tzinfo=UTC)
    assert realtime.raw_path(tmp_path, utc) == tmp_path / "date=2026-09-17" / "000000.json"
    assert realtime.raw_path(tmp_path, NOW.replace(tzinfo=None)) == realtime.raw_path(tmp_path, NOW)


# 같은 초의 잠금 파일이 이미 있으면 '진행 중' 오류를 내고, 남의 잠금 파일을 지우지 않는다. client
# 없이 불러도 API 호출 전에 멈춘다.
def test_concurrent_snapshot_lock_preserves_existing_lock(tmp_path):
    target = realtime.raw_path(tmp_path, NOW)
    target.parent.mkdir(parents=True)
    lock = target.with_suffix(".json.lock")
    lock.touch()
    with pytest.raises(realtime.ApiError, match="진행 중"):
        realtime.download(tmp_path, SECRET, NOW)
    assert lock.exists()
    assert not target.exists()


# 임시 파일을 정식 이름으로 바꾸는 단계가 실패하면(임시 파일은 존재함을 확인) 임시 파일과 잠금
# 파일을 모두 지워 폴더에 아무 파일도 남지 않는다.
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


# 정상 응답이라도 키나 키가 든 요청 URL을 되돌려 보내면 '인증정보'로 거부하고 저장하지 않는다.
# 역슬래시·큰따옴표가 든 키(JSON 이스케이프 모양)까지 두 가지 키 × URL 반사 여부 네 조합.
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


# 응답 원문에는 \uXXXX 이스케이프라 키가 안 보여도, JSON을 푼 뒤 다시 직렬화해 검사하므로 반사로
# 잡는다. 재시도 2번 뒤 실패하고 파일·오류·로그 어디에도 어떤 형태의 키도 남지 않는다.
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


# 날짜 Parquet 교체가 실패해도 이전 파일은 그대로 남고 임시 파일은 지워진다. 적재가 반쯤 쓴
# Parquet을 읽지 않게 한다.
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


# MockTransport는 httpcore를 거치지 않으므로, 키가 든 Location 헤더를 담은 302 응답 바이트를
# MockBackend로 흘려 httpcore.http11의 DEBUG 헤더 로그 경로를 실제로 지나가게 한다. 로그에 키·주소가
# 없고, 수집이 끝난 뒤 로거 필터가 원래대로 돌아와 요청 밖의 일반 DEBUG 로그는 남는지 확인한다.
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


# 빈 키, 재시도 0번, 불리언 재시도(True는 파이썬에서 정수 1처럼 보여 따로 막음)는 API를 부르기 전에
# ValueError.
@pytest.mark.parametrize("key,retries", [("", 3), (SECRET, 0), (SECRET, True)])
def test_invalid_arguments_do_not_make_request(key, retries):
    def handler(request):
        pytest.fail("잘못된 입력에서 API 호출")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            realtime.fetch_snapshot(client, key, retries)
