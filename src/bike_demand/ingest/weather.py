"""기상청 ASOS 시간자료(공공데이터포털 AsosHourlyInfoService)를 월 단위로 받는다.

받은 응답은 월마다 JSON 원본(`data/raw/weather/asos_hourly/stn=108/YYYY-MM.json`)으로
남기고 다시 받지 않는다. API는 전날(D-1)까지만 주므로 끝난 달만 저장한다.
Parquet 변환은 원본 JSON에서 하므로 API를 다시 부르지 않고 몇 번이든 할 수 있다.

인증키는 URL 쿼리에 들어가므로 오류 메시지나 로그에 요청 URL·응답 본문을 남기지 않는다.
전체 건수(totalCount)보다 덜 받은 달은 저장하지 않는다(관측 자체가 빠진 시간은 허용).
"""

# [파일 역할] 학습용 관측 날씨를 받는 수집기. 기상청 ASOS 서울(108) 시간자료를 달마다 받아 월별 JSON
# 원본으로 두고, 전체를 문자열 Parquet 하나(bronze)로 다시 만든다. 2023-01 ~ 2026-06의 30,648시간을
# 빠짐없이 받았다(docs/data.md).
# [왜 이렇게] 모델 특징의 날씨(기온·강수·풍속·습도·신적설)는 이 관측값으로
# 학습·평가한다(docs/experiments.md v1). 원본을 달 단위로 남기면 이미 받은 달은 API를 다시 부르지
# 않고, 변환 규칙을 바꿔도 Parquet만 다시 만들면 된다.
# [안전 장치] Codex 검토(라운드 1·3)에서 나온 문제를 고친 것들이다: 요청 로그 숨김과 오류 메시지에서
# 서버 문자열 제외(커밋 7ae19b3), 덜 받은 쪽은 재시도하고 달 전체를 저장하지 않음(7ae19b3), JSON
# 해석 뒤의 키 반사 검사와 쪽마다 다른 전체 건수 거부(커밋 d021516).
# [연결] refresh_history DAG의 fetch_asos 작업(--end 원본 최신 달). 결과는 dbt source
# bronze.asos_hourly → stg_weather_hourly. model/forecast_gap.py(v4 측정)는 fetch_range로 끝나지
# 않은 달의 관측도 받는다.

from __future__ import annotations

import calendar
import json
import os
import re
import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path

# httpx: API 호출. pyarrow: 문자열 컬럼 Parquet 쓰기.
import httpx
import pyarrow as pa
import pyarrow.parquet as pq

# 인증키 보호 공통 도구(요청 로그 숨김, 응답의 키 반사 검사).
from bike_demand.ingest._http import private_request, reflects_secret

# ENDPOINT: 공공데이터포털 '기상청 지상(종관, ASOS) 시간자료 조회서비스'. SEOUL_STATION: 서울 관측소
# 번호 108(모든 대여소에 이 한 지점을 씀, README 한계). PAGE_SIZE: 한 쪽 행 수 999. 한 달은 최대
# 744시간이라 보통 한 쪽에 다 들어온다.
ENDPOINT = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
SEOUL_STATION = "108"
PAGE_SIZE = 999


# 수집 실패 예외. 메시지에는 이 모듈이 정한 문장과 형식이 정해진 코드만 넣는다.
class ApiError(RuntimeError):
    pass


# 호출 위치: 아래 실행부(--start ~ --end의 달 목록), tests/test_ingest_weather.py.
def months_between(start: str, end: str) -> Iterator[str]:
    """YYYY-MM 형식의 두 달 사이(양 끝 포함)의 달."""
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    while (year, month) <= (end_year, end_month):
        yield f"{year}-{month:02d}"
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


# API는 전날까지의 관측만 주므로, 그달 말일이 오늘보다 앞서야 한 달이 다 들어 있다. 끝나지 않은 달을
# 저장하면 이미 있는 파일은 다시 받지 않는 규칙 때문에 덜 찬 달이 영영 남으므로 건너뛴다.
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


# 검사 순서: (1) 원문에 키가 보이면 거부 → (2) JSON이 아니면(인증 오류는 XML로 옴) 정해진 형식의
# 오류 이름·코드만 뽑아 보고 → (3) 해석한 값에 키가 있으면 거부 → (4) 결과 코드 00만 정상 → (5) 전체
# 건수와 항목 구조 확인. 호출 위치: _get_page.
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
    # 원문에서 안 보여도(유니코드 이스케이프 등) 해석한 값에 키가 들어 있을 수 있다(T25)
    if reflects_secret(json.dumps(payload, ensure_ascii=False), service_key):
        raise ApiError(f"인증키가 반사된 응답 (HTTP {response.status_code})")
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
        raise ApiError("응답 구조 오류")
    # 결과 코드는 숫자 세 자리 이하일 때만 메시지에 넣는다. 서버 메시지(resultMsg)는 요청 URL을 담을
    # 수 있어 쓰지 않는다.
    header = payload["response"].get("header")
    code = header.get("resultCode") if isinstance(header, dict) else None
    if code != "00":
        safe = code if isinstance(code, str) and re.fullmatch(r"\d{1,3}", code) else "알 수 없음"
        raise ApiError(f"API 오류 코드 {safe}")
    body = payload["response"].get("body")
    if not isinstance(body, dict):
        raise ApiError("응답 본문 구조 오류")
    # 전체 건수는 숫자여야 한다. 항목이 없으면 items가 사전이 아닐 수 있어 그때는 빈 목록으로 보고,
    # 개수 검사는 _get_page가 한다.
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
    first_total: int | None = None,
) -> tuple[list[dict], int]:
    """한 쪽을 받는다. 앞서 받은 수(received)로 이 쪽에 와야 할 개수를 확인해 다르면 다시 받는다.

    first_total: 첫 쪽이 알린 전체 건수. 뒤쪽이 다른 값을 알리면 다시 받는다(T26).
    """
    # 키가 든 요청 URL이 로그에 남지 않게 private_request 안에서 요청하고, 실패하면 2초·4초…로 늘려
    # 기다리며 다시 받는다.
    for attempt in range(1, retries + 1):
        try:
            with private_request():
                response = client.get(ENDPOINT, params=params)
            items, total = _parse_page(response, params["serviceKey"])
            if first_total is not None and total != first_total:
                raise ApiError(f"전체 건수가 쪽마다 다름 ({first_total} -> {total})")
            # 이 쪽에 와야 할 개수 = min(쪽 크기, 남은 건수). 짧은 쪽을 그대로 받으면 덜 찬 달이
            # 완료로 저장되므로 다시 받는다(7ae19b3).
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


# 한 달(1일 0시 ~ 말일 23시)의 관측을 받는다. 호출 위치: download, tests/test_ingest_weather.py.
def fetch_month(
    client: httpx.Client,
    service_key: str,
    month: str,
    station: str = SEOUL_STATION,
    retries: int = 3,
) -> list[dict]:
    year, mon = map(int, month.split("-"))
    last = calendar.monthrange(year, mon)[1]
    start, end = date(year, mon, 1), date(year, mon, last)
    return fetch_range(client, service_key, start, end, station, retries, label=month)


# 호출 위치: fetch_month, model/forecast_gap.py(v4 측정의 관측 날씨, 결과는 bronze에 넣지 않음,
# docs/experiments.md v4 기간과 자료).
def fetch_range(
    client: httpx.Client,
    service_key: str,
    start: date,
    end: date,
    station: str = SEOUL_STATION,
    retries: int = 3,
    label: str | None = None,
) -> list[dict]:
    """start 0시 ~ end 23시 관측. 끝나지 않은 달 일부도 받는다(v4 측정용, 저장 안 함)."""
    label = label or f"{start:%Y-%m-%d}~{end:%Y-%m-%d}"
    # 요청 조건: ASOS 자료(dataCd), 시간 단위(dateCd=HR), 관측소 하나, 시작일 00시 ~ 끝일 23시.
    # 인증키는 쿼리 serviceKey로 들어간다.
    params = {
        "serviceKey": service_key,
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "stnIds": station,
        "startDt": f"{start:%Y%m%d}",
        "startHh": "00",
        "endDt": f"{end:%Y%m%d}",
        "endHh": "23",
        "numOfRows": str(PAGE_SIZE),
    }
    items: list[dict] = []
    page = 1
    first_total: int | None = None
    # 쪽을 차례로 받아 전체 건수에 닿으면 멈춘다. 전체 건수는 첫 쪽 값으로 고정해, 뒤쪽이 다른 값을
    # 알리면 _get_page가 거부한다(커밋 d021516: 뒤쪽이 0을 알려 달이 일찍 끝나며 덜 찬 파일이
    # 저장되던 문제).
    while True:
        page_items, total = _get_page(
            client,
            {**params, "pageNo": str(page)},
            retries,
            f"{label} {page}쪽",
            len(items),
            first_total,
        )
        first_total = total if first_total is None else first_total
        items.extend(page_items)
        if len(items) >= total:
            break
        page += 1
    return items


# 원본 경로: raw_dir/stn=108/YYYY-MM.json. 파일 존재가 곧 '그달은 다 받았음'이다.
def raw_path(raw_dir: Path, month: str, station: str = SEOUL_STATION) -> Path:
    return raw_dir / f"stn={station}" / f"{month}.json"


# 호출 위치: 아래 실행부, tests/test_ingest_weather.py. 생성기라 달마다 상태를 바로 출력할 수 있다.
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
            # 이미 받은 달은 API를 부르지 않는다. 다시 받으려면 그달 파일을 지운다.
            if target.exists():
                yield month, "exists"
                continue
            if not is_complete(month, today):
                yield month, "incomplete"
                continue
            items = fetch_month(http, service_key, month, station)
            target.parent.mkdir(parents=True, exist_ok=True)
            # 임시 파일에 쓰고 os.replace로 바꿔, 중간에 멈춰도 반쯤 쓴 파일이 '다 받은 달'로 남지
            # 않게 한다. 상태의 '받은 수/그달 시간 수'로 관측이 빠진 시간이 있는지 바로 보인다.
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, target)
            hours = calendar.monthrange(*map(int, month.split("-")))[1] * 24
            yield month, f"saved {len(items)}/{hours}"
    finally:
        if client is None:
            http.close()


# 이 중 dbt stg_weather_hourly가 쓰는 것은 tm, ta, rn, ws, hm, dsnw, hr3Fhsc이고, 나머지(풍향,
# 이슬점, 기압, 일조, 운량, 시정, 지면온도)는 원본 모양 그대로 보존만 한다.
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


# 모든 월별 원본을 읽어 FIELDS 컬럼만 뽑고 빈 문자열은 null로 바꿔 Parquet 하나를 통째로 다시
# 만든다(멱등). 임시 파일 + os.replace로 교체한다. 호출 위치: 아래 실행부.
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

    # 기본 기간 2023-01 ~ 2026-06(대여이력 기간). 새 반기를 넣을 때 refresh_history가 --end를 원본
    # 최신 달로 넘긴다. 인증키는 .env의 DATA_GO_KR_SERVICE_KEY.
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
