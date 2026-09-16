"""따릉이 bikeList의 모든 페이지를 한 시점의 스냅샷으로 보관한다.

원본: raw_dir/date=YYYY-MM-DD/HHMMSS.json. fetched_at은 수집 시작 시각(KST,
ISO 8601)이고 rentBikeStatus.row는 모든 페이지의 행을 합친 목록이다. pages에는
각 응답 전체를 그대로 남겨 알려지지 않은 필드도 보존한다. 페이지 수집은 순차적이므로
각 대여소의 관측 시각이 완전히 같지는 않다. 같은 초의 파일은 다시 받거나 덮어쓰지 않는다.
bronze: 날짜별 snapshots.parquet. 알려진 필드와 fetched_at을 문자열로 저장하며,
원본만 읽어 통째로 다시 만들므로 중복 적재되지 않는다. 날짜와 시각은 모두 KST이다.

오류에는 응답 본문/요청 URL을 넣지 않으며, httpx의 요청 로그도 수집 중에는 숨긴다.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, quote_plus

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

ENDPOINT = "http://openapi.seoul.go.kr:8088"
PAGE_SIZE = 1000
KST = timezone(timedelta(hours=9))
_REQUEST_ACTIVE: ContextVar[bool] = ContextVar("realtime_request_active", default=False)
FIELDS = {
    "rackTotCnt": "rack_count",
    "stationName": "station_name",
    "parkingBikeTotCnt": "bike_count",
    "shared": "shared",
    "stationLatitude": "lat",
    "stationLongitude": "lon",
    "stationId": "station_id",
}
SCHEMA = pa.schema([("fetched_at", pa.string()), *[(v, pa.string()) for v in FIELDS.values()]])


class ApiError(RuntimeError):
    """인증키나 요청 URL을 포함하지 않는 수집 오류."""


class _HideRequestLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _REQUEST_ACTIVE.get()


@contextmanager
def _private_request() -> Iterator[None]:
    """현재 요청의 HTTP 로그만 숨긴다. 하위 로거의 응답 헤더에도 키가 섞일 수 있다."""
    names = {
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.http2",
        "httpcore.proxy",
        "httpcore.socks",
    }
    names.update(
        name
        for name in list(logging.Logger.manager.loggerDict)
        if name.startswith(("httpx.", "httpcore."))
    )
    loggers = [logging.getLogger(name) for name in names]
    hide = _HideRequestLog()
    token = _REQUEST_ACTIVE.set(True)
    for logger in loggers:
        logger.addFilter(hide)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(hide)
        _REQUEST_ACTIVE.reset(token)


def _kst(value: datetime) -> datetime:
    """시간대 없는 입력은 KST로 해석하고, 있는 입력은 KST로 변환한다."""
    return value.replace(tzinfo=KST) if value.tzinfo is None else value.astimezone(KST)


def _parse_page(response: httpx.Response, service_key: str) -> tuple[dict, list[dict], bool]:
    if not 200 <= response.status_code < 300:
        raise ApiError(f"HTTP 오류 {response.status_code}")
    try:
        payload = response.json()
    except ValueError:
        raise ApiError("JSON이 아닌 응답") from None
    if not isinstance(payload, dict):
        raise ApiError("응답 구조 오류")
    # 잘못된 서버가 인증키/요청 URL을 반사해도 원본 파일에 저장하지 않는다.
    serialized = json.dumps(payload, ensure_ascii=False)
    variants = {service_key, quote(service_key, safe=""), quote_plus(service_key)}
    variants.update(json.dumps(token, ensure_ascii=False)[1:-1] for token in tuple(variants))
    if any(token in serialized for token in variants) or ENDPOINT in serialized:
        raise ApiError("응답에 요청 인증정보가 포함됨")
    body = payload.get("rentBikeStatus", payload)
    if not isinstance(body, dict) or not isinstance(body.get("RESULT"), dict):
        raise ApiError("응답 구조 오류")
    code = body["RESULT"].get("CODE")
    if code == "INFO-200":
        return payload, [], True
    if code != "INFO-000":
        safe_code = (
            code if isinstance(code, str) and re.fullmatch(r"ERROR-\d{3}", code) else "알 수 없음"
        )
        raise ApiError(f"API 오류 {safe_code}")
    if "rentBikeStatus" not in payload:
        raise ApiError("응답 rentBikeStatus 없음")
    rows = body.get("row")
    total = body.get("list_total_count")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ApiError("응답 row 구조 오류")
    if isinstance(total, bool) or not re.fullmatch(r"[0-9]+", str(total)) or len(rows) > PAGE_SIZE:
        raise ApiError("응답 건수 오류")
    return payload, rows, False


def _get_page(
    client: httpx.Client, service_key: str, start: int, retries: int, previous_total: int
) -> tuple[dict, list[dict], bool]:
    url = f"{ENDPOINT}/{quote(service_key, safe='')}/json/bikeList/{start}/{start + PAGE_SIZE - 1}/"
    for attempt in range(1, retries + 1):
        try:
            with _private_request():
                response = client.get(url)
            payload, rows, no_data = _parse_page(response, service_key)
            if no_data and start - 1 < previous_total:
                raise ApiError("전체 건수보다 일찍 끝난 응답 (INFO-200)")
            # 전체 건수 안쪽의 짧은 페이지는 전송 누락일 수 있어 완료로 저장하지 않는다.
            if not no_data and len(rows) < PAGE_SIZE:
                if start - 1 + len(rows) < int(payload["rentBikeStatus"]["list_total_count"]):
                    raise ApiError("전체 건수보다 일찍 끝난 페이지")
            return payload, rows, no_data
        except (httpx.HTTPError, httpx.InvalidURL, ApiError) as exc:
            if attempt == retries:
                detail = str(exc) if isinstance(exc, ApiError) else type(exc).__name__
                raise ApiError(f"bikeList {start}행부터 수집 실패: {detail}") from None
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def fetch_snapshot(client: httpx.Client, service_key: str, retries: int = 3) -> dict:
    """1~1000, 1001~2000 순으로 짧은 페이지/INFO-200까지 받아 원본을 합친다."""
    if not isinstance(service_key, str) or not service_key.strip():
        raise ValueError("인증키가 필요함")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 1:
        raise ValueError("재시도 횟수는 1 이상이어야 함")
    pages: list[dict] = []
    rows: list[dict] = []
    start = 1
    total = 0
    seen: set[str] = set()
    while True:
        payload, chunk, no_data = _get_page(client, service_key, start, retries, total)
        if no_data and not rows:
            raise ApiError("bikeList 데이터 없음 (INFO-200)")
        for row in chunk:
            station_id = row.get("stationId")
            if not isinstance(station_id, str) or not station_id.strip():
                raise ApiError("응답 대여소 ID 누락")
            if station_id in seen:
                raise ApiError("응답 대여소 ID 중복")
            seen.add(station_id)
        if not no_data:
            total = int(payload["rentBikeStatus"]["list_total_count"])
        pages.append(payload)
        rows.extend(chunk)
        if no_data or len(chunk) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    if not rows:
        raise ApiError("bikeList 데이터 없음")
    merged = {**pages[0]["rentBikeStatus"], "row": rows}
    return {"rentBikeStatus": merged, "pages": pages}


def raw_path(raw_dir: Path, fetched_at: datetime) -> Path:
    """수집 시작 시각의 KST 날짜/초로 원본 경로를 정한다."""
    local = _kst(fetched_at)
    return raw_dir / f"date={local:%Y-%m-%d}" / f"{local:%H%M%S}.json"


def download(
    raw_dir: Path,
    service_key: str,
    fetched_at: datetime | None = None,
    client: httpx.Client | None = None,
    retries: int = 3,
) -> tuple[Path, str]:
    """한 번 수집해 (원본 경로, saved 행수 / exists)를 반환한다.

    같은 초에 실행되면 기존 원본을 보존한다. 동시 작성은 배타적 잠금으로 막는다.
    중단으로 .lock이 남으면 실행 중인 수집기가 없는지 확인한 뒤 운영자가 제거한다.
    """
    fetched_at = _kst(fetched_at or datetime.now(KST))
    target = raw_path(raw_dir, fetched_at)
    if target.exists():
        return target, "exists"
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(".json.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ApiError("같은 시각의 스냅샷 수집이 이미 진행 중") from None
    os.close(fd)
    http = None
    tmp = None
    try:
        if target.exists():
            return target, "exists"
        http = client or httpx.Client(timeout=30)
        snapshot = fetch_snapshot(http, service_key, retries)
        snapshot["fetched_at"] = fetched_at.isoformat()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".tmp", dir=target.parent, delete=False
        ) as stream:
            tmp = Path(stream.name)
            json.dump(snapshot, stream, ensure_ascii=False)
        if target.exists():
            return target, "exists"
        os.replace(tmp, target)
        return target, f"saved {len(snapshot['rentBikeStatus']['row'])}"
    finally:
        if http is not None and client is None:
            http.close()
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def to_parquet(raw_dir: Path, out_path: Path, day: date) -> int:
    """KST 날짜 하나의 모든 원본을 읽어 문자열 Parquet을 원자적으로 교체한다."""
    records: list[dict] = []
    for path in sorted((raw_dir / f"date={day:%Y-%m-%d}").glob("*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        for row in snapshot["rentBikeStatus"]["row"]:
            record = {"fetched_at": snapshot["fetched_at"]}
            record.update(
                {
                    dst: None if row.get(src) in (None, "") else str(row[src])
                    for src, dst in FIELDS.items()
                }
            )
            records.append(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tmp", dir=out_path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), tmp, compression="zstd")
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return len(records)


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="따릉이 실시간 대여정보 스냅샷 하나를 받는다")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/realtime/bikelist"))
    parser.add_argument("--bronze", type=Path, default=Path("data/bronze/realtime/bikelist"))
    args = parser.parse_args()
    load_dotenv()
    key = os.environ.get("SEOUL_OPEN_API_KEY")
    if not key:
        raise SystemExit(".env에 SEOUL_OPEN_API_KEY가 없음 (.env.example 참고)")
    collected_at = datetime.now(KST)
    try:
        path, status = download(args.raw, key, collected_at)
        print(path, status, flush=True)
        output = args.bronze / f"date={collected_at:%Y-%m-%d}" / "snapshots.parquet"
        print("parquet rows", to_parquet(args.raw, output, collected_at.date()))
    except ApiError as exc:
        raise SystemExit(str(exc)) from None
