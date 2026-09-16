"""기상청 ASOS 시간자료(공공데이터포털 AsosHourlyInfoService)를 월 단위로 받는다.

받은 응답은 월마다 JSON 원본(`data/raw/weather/asos_hourly/stn=108/YYYY-MM.json`)으로
남기고 다시 받지 않는다. API는 전날(D-1)까지만 주므로 끝난 달만 저장한다.
Parquet 변환은 원본 JSON에서 하므로 API를 다시 부르지 않고 몇 번이든 할 수 있다.

인증키는 URL 쿼리에 들어가므로 오류 메시지나 로그에 요청 URL·응답 본문을 남기지 않는다.
전체 건수(totalCount)보다 덜 받은 달은 저장하지 않는다(관측 자체가 빠진 시간은 허용).
"""

from __future__ import annotations

import calendar
import json
import os
import re
import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from bike_demand.ingest._http import private_request, reflects_secret

ENDPOINT = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
SEOUL_STATION = "108"
PAGE_SIZE = 999


class ApiError(RuntimeError):
    pass


def months_between(start: str, end: str) -> Iterator[str]:
    """YYYY-MM 형식의 두 달 사이(양 끝 포함)의 달."""
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    while (year, month) <= (end_year, end_month):
        yield f"{year}-{month:02d}"
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def is_complete(month: str, today: date) -> bool:
    """그달 마지막 날이 어제 이전이면 API가 그달을 다 준다."""
    year, mon = map(int, month.split("-"))
    last_day = date(year, mon, calendar.monthrange(year, mon)[1])
    return last_day < today


# 인증키 오류 같은 경우 JSON을 요청해도 XML이 온다. 본문은 보여 주지 않고, 형식이 정해진
# 오류 이름(대문자·밑줄)과 코드(숫자)만 뽑아 쓴다. 자유 문자열(resultMsg 등)에는 서버가
# 요청 URL을 되돌려 넣을 수 있기 때문이다.
_XML_AUTH_MSG = re.compile(r"<returnAuthMsg>([A-Z_]{1,80})</returnAuthMsg>")
_XML_REASON = re.compile(r"<returnReasonCode>(\d{1,4})</returnReasonCode>")


def _parse_page(response: httpx.Response, service_key: str) -> tuple[list[dict], int]:
    """(이번 쪽 항목, 전체 건수)를 돌려준다."""
    text = response.text
    if reflects_secret(text, service_key):
        raise ApiError(f"인증키가 반사된 응답 (HTTP {response.status_code})")
    try:
        payload = response.json()
    except ValueError:
        auth, reason = _XML_AUTH_MSG.search(text), _XML_REASON.search(text)
        detail = " ".join(m.group(1) for m in (auth, reason) if m) or "형식 알 수 없음"
        raise ApiError(f"JSON이 아닌 응답 (HTTP {response.status_code}): {detail}") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
        raise ApiError("응답 구조 오류")
    header = payload["response"].get("header")
    code = header.get("resultCode") if isinstance(header, dict) else None
    if code != "00":
        safe = code if isinstance(code, str) and re.fullmatch(r"\d{1,3}", code) else "알 수 없음"
        raise ApiError(f"API 오류 코드 {safe}")
    body = payload["response"].get("body")
    if not isinstance(body, dict):
        raise ApiError("응답 본문 구조 오류")
    total = str(body.get("totalCount", ""))
    if not total.isdigit():
        raise ApiError("응답 전체 건수 오류")
    container = body.get("items")
    items = container.get("item", []) if isinstance(container, dict) else []
    if not isinstance(items, list) or any(not isinstance(i, dict) for i in items):
        raise ApiError("응답 항목 구조 오류")
    return items, int(total)


def _get_page(
    client: httpx.Client,
    params: dict[str, str],
    retries: int,
    label: str,
    received: int,
) -> tuple[list[dict], int]:
    """한 쪽을 받는다. 앞서 받은 수(received)로 이 쪽에 와야 할 개수를 확인해 다르면 다시 받는다."""
    for attempt in range(1, retries + 1):
        try:
            with private_request():
                response = client.get(ENDPOINT, params=params)
            items, total = _parse_page(response, params["serviceKey"])
            expected = max(0, min(PAGE_SIZE, total - received))
            if len(items) != expected:
                raise ApiError(f"항목 {len(items)}개, 전체 건수로 보면 {expected}개여야 함")
            return items, total
        except (httpx.HTTPError, ApiError) as exc:
            if attempt == retries:
                # httpx 예외 메시지에는 인증키가 든 URL이 들어갈 수 있어 예외 종류만 남긴다.
                detail = str(exc) if isinstance(exc, ApiError) else type(exc).__name__
                raise ApiError(f"{label} 실패: {detail}") from None
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def fetch_month(
    client: httpx.Client,
    service_key: str,
    month: str,
    station: str = SEOUL_STATION,
    retries: int = 3,
) -> list[dict]:
    year, mon = map(int, month.split("-"))
    last = calendar.monthrange(year, mon)[1]
    params = {
        "serviceKey": service_key,
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "stnIds": station,
        "startDt": f"{year}{mon:02d}01",
        "startHh": "00",
        "endDt": f"{year}{mon:02d}{last:02d}",
        "endHh": "23",
        "numOfRows": str(PAGE_SIZE),
    }
    items: list[dict] = []
    page = 1
    while True:
        page_items, total = _get_page(
            client, {**params, "pageNo": str(page)}, retries, f"{month} {page}쪽", len(items)
        )
        items.extend(page_items)
        if len(items) >= total:
            break
        page += 1
    return items


def raw_path(raw_dir: Path, month: str, station: str = SEOUL_STATION) -> Path:
    return raw_dir / f"stn={station}" / f"{month}.json"


def download(
    raw_dir: Path,
    months: list[str],
    service_key: str,
    today: date,
    client: httpx.Client | None = None,
    station: str = SEOUL_STATION,
) -> Iterator[tuple[str, str]]:
    """달마다 (month, 상태)를 돌려준다. 상태: exists / incomplete / saved 받은수/그달시간수.

    관측이 빠진 시간이 있어도 받은 그대로 저장한다. 빠진 시간은 dbt 테스트에서 드러난다.
    """
    http = client or httpx.Client(timeout=30)
    try:
        for month in months:
            target = raw_path(raw_dir, month, station)
            if target.exists():
                yield month, "exists"
                continue
            if not is_complete(month, today):
                yield month, "incomplete"
                continue
            items = fetch_month(http, service_key, month, station)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, target)
            hours = calendar.monthrange(*map(int, month.split("-")))[1] * 24
            yield month, f"saved {len(items)}/{hours}"
    finally:
        if client is None:
            http.close()


# 원본 응답 필드 중 저장할 것. 값은 문자열 그대로 두고(빈 문자열은 null) 해석은 dbt에서 한다.
FIELDS = {
    "tm": "observed_at",  # "YYYY-MM-DD HH:MM", KST
    "stnId": "station_id",
    "ta": "temp_c",
    "rn": "rain_mm",  # 비가 안 오면 빈 값
    "ws": "wind_ms",
    "wd": "wind_dir_deg",
    "hm": "humidity_pct",
    "td": "dew_point_c",
    "pa": "pressure_hpa",
    "ss": "sunshine_hr",
    "dsnw": "snow_cm",
    "hr3Fhsc": "new_snow_3h_cm",
    "dc10Tca": "cloud_total",
    "vs": "visibility_10m",
    "ts": "ground_temp_c",
}
SCHEMA = pa.schema([(name, pa.string()) for name in FIELDS.values()])


def to_parquet(raw_dir: Path, out_path: Path, station: str = SEOUL_STATION) -> int:
    """저장된 월별 JSON 전체를 Parquet 파일 하나로 다시 만든다."""
    rows: list[dict] = []
    for path in sorted((raw_dir / f"stn={station}").glob("*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            rows.append({dst: (item.get(src) or None) for src, dst in FIELDS.items()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp, compression="zstd")
    os.replace(tmp, out_path)
    return len(rows)


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="기상청 ASOS 시간자료를 월 단위로 받는다")
    parser.add_argument("--start", default="2023-01")
    parser.add_argument("--end", default="2026-06")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/weather/asos_hourly"))
    parser.add_argument("--out", type=Path, default=Path("data/bronze/weather/asos_hourly.parquet"))
    args = parser.parse_args()

    load_dotenv()
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY")
    if not key:
        raise SystemExit(".env에 DATA_GO_KR_SERVICE_KEY가 없음 (.env.example 참고)")
    for month, status in download(
        args.raw, list(months_between(args.start, args.end)), key, date.today()
    ):
        print(month, status, flush=True)
    print("parquet rows", to_parquet(args.raw, args.out))
