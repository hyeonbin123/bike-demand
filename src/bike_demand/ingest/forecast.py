"""기상청 단기예보 한 발표를 모든 쪽과 함께 보관하고 긴 형식 Parquet로 옮긴다.

원본은 ``nx=60_ny=127/base=YYYYMMDDHHMM.json``에 저장한다. JSON의 ``base_datetime``과
``fetched_at``은 KST ISO 시각, ``nx``·``ny``는 문자열이다. ``pages``는 응답 전체를 쪽
순서대로 담아 알 수 없는 응답 필드도 보존하고, ``items``는 모든 쪽의 예보 항목을 합친다.
기존 발표는 건너뛰며, 같은 발표를 처리하는 수집기끼리는 잠금 파일로 덮어쓰기를 막는다.
실패하면 완료 JSON을 남기지 않는다. 중단으로 남은 .lock 파일은 실행 상태 확인 후 정리한다.

발표 후 10분부터 받을 수 있다고 가정한다. API 오류의 본문·메시지·URL은 출력하지 않으며,
httpx/httpcore 요청 로그도 해당 요청 범위에서 숨긴다. 원본에 인증키가 반사되면 저장을 거부한다.
Parquet은 모든 격자의 원본 전체에서 다시 만들고, 모든 컬럼을 문자열로 저장한다.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from bike_demand.ingest._http import private_request, reflects_secret

ENDPOINT = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
KST = timezone(timedelta(hours=9), name="KST")
BASE_HOURS = (2, 5, 8, 11, 14, 17, 20, 23)
PAGE_SIZE = 1000
SEOUL_NX, SEOUL_NY = 60, 127
SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "base_datetime",
            "fcst_datetime",
            "category",
            "value",
            "nx",
            "ny",
            "fetched_at",
        )
    ]
)


class ApiError(RuntimeError):
    """요청 URL이나 응답 본문을 포함하지 않는 수집 오류."""


def _kst(value: datetime) -> datetime:
    """시간대 없는 입력은 서울 현지 시각으로 해석한다."""
    return value.replace(tzinfo=KST) if value.tzinfo is None else value.astimezone(KST)


def latest_base(now: datetime) -> datetime:
    """현재 시각에서 발표 후 10분이 지난 최신 발표를 고르는 순수 함수."""
    available = _kst(now) - timedelta(minutes=10)
    for hour in reversed(BASE_HOURS):
        candidate = available.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate <= available:
            return candidate
    return (available - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)


def recent_bases(now: datetime, hours: int = 24) -> list[datetime]:
    """지금 받을 수 있는 최신 발표와 그 전 `hours`시간 안의 발표들(오래된 것부터).

    API는 지난 발표도 하루 안쪽이면 준다. PC나 Docker가 꺼져 빠진 발표를 다음 실행이 채운다.
    """
    latest = latest_base(now)
    bases = []
    candidate = latest
    while latest - candidate < timedelta(hours=hours):
        bases.append(candidate)
        previous = candidate - timedelta(hours=1)
        candidate = latest_base(previous + timedelta(minutes=10))
    return bases[::-1]


def download_recent(
    raw_dir: Path,
    service_key: str,
    *,
    now: datetime | None = None,
    client: httpx.Client | None = None,
    nx: int = SEOUL_NX,
    ny: int = SEOUL_NY,
) -> list[tuple[datetime, str]]:
    """최근 24시간 발표 중 없는 것을 받는다. 지난 발표가 실패하면 적어 두고 넘어가고,
    최신 발표가 실패할 때만 오류를 낸다(최신 예보가 서비스에 필요하므로)."""
    fetched_at = _kst(now if now is not None else datetime.now(KST))
    bases = recent_bases(fetched_at)
    results = []
    for base in bases:
        try:
            _, status = download(
                raw_dir, service_key, base, now=fetched_at, client=client, nx=nx, ny=ny
            )
        except ApiError as exc:
            if base == bases[-1]:
                raise
            status = f"failed: {exc}"
        results.append((base, status))
    return results


def _base(value: datetime) -> datetime:
    base = _kst(value)
    if base.hour not in BASE_HOURS or base.minute or base.second or base.microsecond:
        raise ValueError("기준 시각은 0200, 0500, 0800, 1100, 1400, 1700, 2000, 2300 중 하나")
    return base


def _grid(nx: int, ny: int) -> None:
    if type(nx) is not int or type(ny) is not int or not (1 <= nx <= 149 and 1 <= ny <= 253):
        raise ValueError("격자는 정수 nx=1~149, ny=1~253 범위여야 함")


def _datetime(date: object, clock: object) -> datetime:
    if not isinstance(date, str) or not isinstance(clock, str):
        raise ApiError("예보 시각 형식 오류")
    if not re.fullmatch(r"[0-9]{8}", date) or not re.fullmatch(r"[0-9]{4}", clock):
        raise ApiError("예보 시각 형식 오류")
    try:
        return datetime.strptime(date + clock, "%Y%m%d%H%M").replace(tzinfo=KST)
    except ValueError:
        raise ApiError("예보 시각 형식 오류") from None


def _integer(value: object) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        raise ApiError("응답 건수 또는 격자 형식 오류")
    return int(str(value))


def _item_key(item: dict, base: datetime, nx: int, ny: int) -> tuple:
    if _datetime(item.get("baseDate"), item.get("baseTime")) != base:
        raise ApiError("요청과 다른 발표 시각의 응답")
    if _integer(item.get("nx")) != nx or _integer(item.get("ny")) != ny:
        raise ApiError("요청과 다른 격자의 응답")
    forecast_at = _datetime(item.get("fcstDate"), item.get("fcstTime"))
    category, value = item.get("category"), item.get("fcstValue")
    if not isinstance(category, str) or not category or not isinstance(value, (str, int, float)):
        raise ApiError("예보 항목 형식 오류")
    return forecast_at, category


def _parse_page(
    response: httpx.Response,
    service_key: str,
    base: datetime,
    nx: int,
    ny: int,
    count: int,
    total: int | None,
    seen: set[tuple],
) -> tuple[dict, list[dict], int]:
    if response.status_code != 200:
        raise ApiError(f"HTTP 응답 오류 ({response.status_code})")
    try:
        payload = response.json()
    except ValueError:
        raise ApiError("JSON이 아닌 응답") from None
    serialized = json.dumps(payload, ensure_ascii=False)
    if reflects_secret(serialized, service_key):
        raise ApiError("인증정보가 반사된 응답은 저장하지 않음")
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
        raise ApiError("API 응답 구조 오류")
    response_body = payload["response"]
    header = response_body.get("header")
    if not isinstance(header, dict):
        raise ApiError("API 헤더 구조 오류")
    if header.get("resultCode") == "03":
        raise ApiError("예보 데이터 없음 (03)")
    if header.get("resultCode") != "00":
        raise ApiError("API 응답 오류 (정상 코드 00 아님)")
    body = response_body.get("body")
    if not isinstance(body, dict):
        raise ApiError("API 본문 구조 오류")
    page_total = _integer(body.get("totalCount"))
    if page_total == 0:
        raise ApiError("예보 데이터 없음")
    if total is not None and page_total != total:
        raise ApiError("쪽 사이 전체 건수가 달라짐")
    container = body.get("items")
    items = container.get("item") if isinstance(container, dict) else None
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ApiError("예보 목록 구조 오류")
    if len(items) != min(PAGE_SIZE, page_total - count):
        raise ApiError("전체 건수에 비해 불완전한 예보 페이지")
    page_seen = set()
    for item in items:
        key = _item_key(item, base, nx, ny)
        if key in seen or key in page_seen:
            raise ApiError("중복 예보 항목 또는 반복된 페이지")
        page_seen.add(key)
    return payload, items, page_total


def _get_page(
    client: httpx.Client,
    params: dict[str, str],
    retries: int,
    base: datetime,
    nx: int,
    ny: int,
    count: int,
    total: int | None,
    seen: set[tuple],
) -> tuple[dict, list[dict], int]:
    for attempt in range(1, retries + 1):
        try:
            with private_request():
                response = client.get(ENDPOINT, params=params)
            return _parse_page(response, params["serviceKey"], base, nx, ny, count, total, seen)
        except (httpx.HTTPError, ApiError) as exc:
            if attempt == retries:
                detail = str(exc) if isinstance(exc, ApiError) else "HTTP 전송 오류"
                raise ApiError(f"{params['pageNo']}쪽 수집 실패: {detail}") from None
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def fetch_forecast(
    client: httpx.Client,
    service_key: str,
    base: datetime,
    nx: int = SEOUL_NX,
    ny: int = SEOUL_NY,
    retries: int = 3,
) -> dict:
    """쪽마다 재시도하여 한 발표 전체를 받고 원본 envelope를 돌려준다."""
    base = _base(base)
    _grid(nx, ny)
    if not isinstance(service_key, str) or not service_key.strip():
        raise ValueError("인증키가 비어 있음")
    if type(retries) is not int or retries < 1:
        raise ValueError("재시도 횟수는 양의 정수여야 함")
    params = {
        "serviceKey": service_key,
        "pageNo": "1",
        "numOfRows": str(PAGE_SIZE),
        "dataType": "JSON",
        "base_date": base.strftime("%Y%m%d"),
        "base_time": base.strftime("%H%M"),
        "nx": str(nx),
        "ny": str(ny),
    }
    pages, items = [], []
    seen: set[tuple] = set()
    total = None
    while total is None or len(items) < total:
        params["pageNo"] = str(len(pages) + 1)
        payload, chunk, total = _get_page(
            client, params, retries, base, nx, ny, len(items), total, seen
        )
        pages.append(payload)
        items.extend(chunk)
        seen.update(_item_key(item, base, nx, ny) for item in chunk)
    return {
        "base_datetime": base.isoformat(),
        "nx": str(nx),
        "ny": str(ny),
        "pages": pages,
        "items": items,
    }


def raw_path(raw_dir: Path, base: datetime, nx: int = SEOUL_NX, ny: int = SEOUL_NY) -> Path:
    """발표 시각과 격자로 저장할 경로를 만든다."""
    base = _base(base)
    _grid(nx, ny)
    return raw_dir / f"nx={nx}_ny={ny}" / f"base={base:%Y%m%d%H%M}.json"


def download(
    raw_dir: Path,
    service_key: str,
    base: datetime | None = None,
    *,
    now: datetime | None = None,
    client: httpx.Client | None = None,
    nx: int = SEOUL_NX,
    ny: int = SEOUL_NY,
    retries: int = 3,
) -> tuple[Path, str]:
    """한 발표를 저장해 (경로, exists 또는 saved 건수)를 돌려준다.

    now는 수집 시작 시각이며 테스트에서 주입할 수 있다. 명시한 발표도 제공 지연과
    발표 시각을 검증한다. 기존 파일은 API 호출 없이 건너뛴다.
    """
    fetched_at = _kst(now if now is not None else datetime.now(KST))
    available = latest_base(fetched_at)
    base = available if base is None else _base(base)
    if base > available:
        raise ValueError("아직 제공 시각이 되지 않은 발표")
    target = raw_path(raw_dir, base, nx, ny)
    if target.exists():
        return target, "exists"
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(".json.lock")
    try:
        lock_file = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise ApiError("같은 발표를 다른 수집기가 처리 중") from None
    temporary = None
    http = None
    try:
        lock_file.close()
        if target.exists():
            return target, "exists"
        http = client or httpx.Client(timeout=30)
        payload = fetch_forecast(http, service_key, base, nx, ny, retries)
        payload["fetched_at"] = fetched_at.isoformat()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=target.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            return target, "exists"
        os.replace(temporary, target)
        return target, f"saved {len(payload['items'])}"
    finally:
        if client is None and http is not None:
            http.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def to_parquet(raw_dir: Path, out_path: Path) -> int:
    """모든 격자·발표 원본에서 문자열 긴 형식 Parquet을 통째로 다시 만든다."""
    rows = []
    for path in sorted(raw_dir.glob("nx=*_ny=*/base=*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            base = _base(datetime.fromisoformat(payload["base_datetime"]))
            fetched_at = _kst(datetime.fromisoformat(payload["fetched_at"])).isoformat()
            nx, ny = _integer(payload["nx"]), _integer(payload["ny"])
            _grid(nx, ny)
            items = payload["items"]
            if not isinstance(items, list):
                raise ApiError("원본의 예보 목록 구조 오류")
            for item in items:
                forecast_at, category = _item_key(item, base, nx, ny)
                rows.append(
                    {
                        "base_datetime": base.isoformat(),
                        "fcst_datetime": forecast_at.isoformat(),
                        "category": category,
                        "value": str(item["fcstValue"]),
                        "nx": str(nx),
                        "ny": str(ny),
                        "fetched_at": fetched_at,
                    }
                )
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ApiError("저장된 예보 JSON 형식 오류") from None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out_path.parent, prefix=out_path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temporary, compression="zstd")
        os.replace(temporary, out_path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rows)


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="기상청 단기예보 한 발표를 모든 쪽과 함께 수집")
    parser.add_argument("--base-date", help="YYYYMMDD (--base-time과 함께 지정)")
    parser.add_argument("--base-time", help="HHMM (--base-date와 함께 지정)")
    parser.add_argument(
        "--catch-up", action="store_true", help="최신 발표와 지난 24시간의 빠진 발표를 받는다"
    )
    parser.add_argument("--nx", type=int, default=SEOUL_NX)
    parser.add_argument("--ny", type=int, default=SEOUL_NY)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/weather/vilage_fcst"))
    parser.add_argument("--out", type=Path, default=Path("data/bronze/weather/vilage_fcst.parquet"))
    args = parser.parse_args()
    if bool(args.base_date) != bool(args.base_time):
        parser.error("--base-date와 --base-time은 함께 지정해야 함")
    load_dotenv()
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY")
    if not key:
        raise SystemExit(".env에 DATA_GO_KR_SERVICE_KEY가 없음 (.env.example 참고)")
    if args.catch_up and args.base_date:
        parser.error("--catch-up은 --base-date와 함께 쓰지 않는다")
    try:
        if args.catch_up:
            for base, status in download_recent(args.raw, key, nx=args.nx, ny=args.ny):
                print(f"{base:%Y%m%d%H%M}", status, flush=True)
        else:
            chosen = _datetime(args.base_date, args.base_time) if args.base_date else None
            path, status = download(args.raw, key, chosen, nx=args.nx, ny=args.ny)
            print(path.name, status, flush=True)
        print("parquet rows", to_parquet(args.raw, args.out))
    except (ApiError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
