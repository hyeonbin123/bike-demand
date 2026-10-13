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

# [파일 역할] 서비스 예측의 날씨 입력을 공급하는 수집기. 기상청 단기예보(공공데이터포털
# VilageFcstInfoService_2.0)의 한 발표를 서울 격자(60·127)에 대해 모든 쪽까지 받아 원본 JSON으로
# 보관하고, 모든 발표를 긴 형식 문자열 Parquet(bronze) 하나로 다시 만든다.
# [왜 이렇게] 학습은 관측 날씨(ASOS)로 했지만 서비스 시점에는 관측이 없으므로 같은 특징 자리에
# 예보를 넣는다(docs/experiments.md v1·v4). 원본은 응답 그대로(모르는 필드까지) 남기고 Parquet은
# 원본에서 언제든 다시 만든다. API는 지난 하루 안쪽 발표만 주므로 원본을 잃으면 다시 받을 수 없다.
# [안전 장치] 부분·중복 쪽 거부, 키가 반사된 응답 저장 거부, 요청 로그 숨김, 잠금 파일 + 임시 파일
# 교체로 원자적 저장(커밋 18f2bdd, 23b1093).
# [연결] forecast_and_predict DAG의 fetch_forecast 작업이 --catch-up으로 부른다. 만든 Parquet은
# serving/load.py의 load_forecasts가 weather_forecasts 표에 넣고, model/predict.py가 그 표의 예보로
# 예측한다.

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# httpx: API 호출(동기 클라이언트, 제한 시간 30초). pyarrow: 수집 결과를 문자열 컬럼 Parquet으로
# 쓴다.
import httpx
import pyarrow as pa
import pyarrow.parquet as pq

# 인증키가 로그·저장 파일로 새지 않게 하는 공통 도구(ingest/_http.py).
from bike_demand.ingest._http import private_request, reflects_secret

# ENDPOINT: 단기예보 조회 API 주소. 인증키는 쿼리 serviceKey로 붙는다.
# BASE_HOURS: 단기예보 발표 시각(하루 8번). PAGE_SIZE: 한 쪽의 최대 행 수(한 발표는 약 1,000여
# 항목이라 보통 두 쪽, 커밋 18f2bdd 확인 값 1,016).
# SEOUL_NX/NY: 기상청 격자 좌표의 서울 지점. 모든 대여소에 이 한 격자의 예보를 쓴다(README 한계
# '날씨는 한 지점').
ENDPOINT = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
KST = timezone(timedelta(hours=9), name="KST")
BASE_HOURS = (2, 5, 8, 11, 14, 17, 20, 23)
PAGE_SIZE = 1000
SEOUL_NX, SEOUL_NY = 60, 127
# bronze 규칙대로 모든 컬럼을 문자열로 저장한다. 값(value)에는 '강수없음' 같은 글자가 섞여 있어
# 해석은 적재·예측 쪽에서 한다.
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


# 수집 실패를 나타내는 예외. 메시지에 요청 URL이나 응답 본문을 넣지 않는다(서버가 URL·키를 응답에
# 되돌려 보낼 수 있으므로). 실행부가 이 메시지만 출력하고 종료한다.
class ApiError(RuntimeError):
    """요청 URL이나 응답 본문을 포함하지 않는 수집 오류."""


# 모든 시각을 KST 기준으로 맞추는 도우미.
def _kst(value: datetime) -> datetime:
    """시간대 없는 입력은 서울 현지 시각으로 해석한다."""
    return value.replace(tzinfo=KST) if value.tzinfo is None else value.astimezone(KST)


# now에서 10분을 뺀 시각 이전의 가장 늦은 발표 시각을 고른다(발표 뒤 10분부터 받을 수 있다는 가정).
# 그날 02:10 이전이면 전날 23시 발표다.
# 호출 위치: recent_bases, download(요청 발표가 아직 제공 전인지 검사).
def latest_base(now: datetime) -> datetime:
    """현재 시각에서 발표 후 10분이 지난 최신 발표를 고르는 순수 함수."""
    available = _kst(now) - timedelta(minutes=10)
    for hour in reversed(BASE_HOURS):
        candidate = available.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate <= available:
            return candidate
    return (available - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)


# 기본 24시간이면 최신 발표 + 그 앞 7개, 모두 8개 발표다(하루 8번 발표).
def recent_bases(now: datetime, hours: int = 24) -> list[datetime]:
    """지금 받을 수 있는 최신 발표와 그 전 `hours`시간 안의 발표들(오래된 것부터).

    API는 지난 발표도 하루 안쪽이면 준다. PC나 Docker가 꺼져 빠진 발표를 다음 실행이 채운다.
    """
    latest = latest_base(now)
    bases = []
    candidate = latest
    while latest - candidate < timedelta(hours=hours):
        bases.append(candidate)
        # 한 시간 전 시각에서 다시 latest_base를 쓰면 바로 앞 발표가 나온다(10분을 더하는 것은
        # latest_base가 10분을 빼기 때문).
        previous = candidate - timedelta(hours=1)
        candidate = latest_base(previous + timedelta(minutes=10))
    return bases[::-1]


# --catch-up 모드의 본체. 2026-09-17 밤 PC가 꺼져 발표 세 개를 못 받은 뒤 더했다(커밋 e3c788a). 이미
# 있는 발표는 download가 API를 부르지 않고 'exists'로 넘어가므로 매 실행이 사실상 최신 발표 하나만
# 받는다.
# 반환: [(발표 시각, 상태 문자열)] 오래된 순. 상태는 exists / saved N / failed: 사유.
# 호출 위치: 아래 실행부(--catch-up).
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
            # 최신 발표 실패만 예외로 올려 Airflow 작업을 실패시킨다(재시도·예측 중단). 지난 발표
            # 실패는 상태에 적고 계속한다.
            if base == bases[-1]:
                raise
            status = f"failed: {exc}"
        results.append((base, status))
    return results


# 발표 시각이 하루 8번 발표 중 하나(정각)인지 검사하고 KST로 돌려준다. 잘못된 시각으로 API를
# 부르거나 파일 이름을 만들지 않게 한다.
def _base(value: datetime) -> datetime:
    base = _kst(value)
    if base.hour not in BASE_HOURS or base.minute or base.second or base.microsecond:
        raise ValueError("기준 시각은 0200, 0500, 0800, 1100, 1400, 1700, 2000, 2300 중 하나")
    return base


# 격자 좌표가 기상청 격자 범위(nx 1~149, ny 1~253)의 정수인지 검사. type(...) is int로 비교해
# bool(True는 int의 하위형)도 막는다.
def _grid(nx: int, ny: int) -> None:
    if type(nx) is not int or type(ny) is not int or not (1 <= nx <= 149 and 1 <= ny <= 253):
        raise ValueError("격자는 정수 nx=1~149, ny=1~253 범위여야 함")


# API의 날짜(YYYYMMDD)와 시각(HHMM) 문자열을 KST datetime으로 엄격히 바꾼다. 형식이 다르면 값을
# 메시지에 넣지 않고 ApiError.
def _datetime(date: object, clock: object) -> datetime:
    if not isinstance(date, str) or not isinstance(clock, str):
        raise ApiError("예보 시각 형식 오류")
    if not re.fullmatch(r"[0-9]{8}", date) or not re.fullmatch(r"[0-9]{4}", clock):
        raise ApiError("예보 시각 형식 오류")
    try:
        return datetime.strptime(date + clock, "%Y%m%d%H%M").replace(tzinfo=KST)
    except ValueError:
        raise ApiError("예보 시각 형식 오류") from None


# 숫자로만 된 값(문자열 또는 정수)을 int로. bool이나 부호·소수점이 있는 값은 거부한다.
def _integer(value: object) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        raise ApiError("응답 건수 또는 격자 형식 오류")
    return int(str(value))


# 예보 항목 하나가 요청한 발표·격자의 것인지 확인하고 (예보 시각, 항목 이름) 키를 돌려준다. 이 키로
# 쪽 사이 중복을 찾고, Parquet 변환 때도 같은 검사를 한다.
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


# 한 쪽 응답을 검증해 (원본 payload, 예보 항목들, 전체 건수)를 돌려준다. 실패는 모두 ApiError라
# _get_page가 재시도한다.
# count: 앞 쪽까지 모은 항목 수, total: 첫 쪽에서 본 전체 건수(첫 쪽이면 None), seen: 앞 쪽들의 항목
# 키.
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
    # 구조를 보기 전에 키 반사부터 검사한다. 이후 어떤 오류 메시지나 저장 파일에도 이 응답이
    # 들어가지 않게 하려는 순서다.
    serialized = json.dumps(payload, ensure_ascii=False)
    if reflects_secret(serialized, service_key):
        raise ApiError("인증정보가 반사된 응답은 저장하지 않음")
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
        raise ApiError("API 응답 구조 오류")
    response_body = payload["response"]
    header = response_body.get("header")
    if not isinstance(header, dict):
        raise ApiError("API 헤더 구조 오류")
    # 공공데이터포털 결과 코드: 00 정상, 03 데이터 없음. 나머지는 모두 오류로 본다. 서버
    # 메시지(resultMsg)는 URL을 담을 수 있어 옮기지 않는다.
    if header.get("resultCode") == "03":
        raise ApiError("예보 데이터 없음 (03)")
    if header.get("resultCode") != "00":
        raise ApiError("API 응답 오류 (정상 코드 00 아님)")
    body = response_body.get("body")
    if not isinstance(body, dict):
        raise ApiError("API 본문 구조 오류")
    # 전체 건수가 0이면 빈 발표, 쪽마다 전체 건수가 달라지면 발표가 바뀌는 중으로 보고 거부한다.
    page_total = _integer(body.get("totalCount"))
    if page_total == 0:
        raise ApiError("예보 데이터 없음")
    if total is not None and page_total != total:
        raise ApiError("쪽 사이 전체 건수가 달라짐")
    container = body.get("items")
    items = container.get("item") if isinstance(container, dict) else None
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ApiError("예보 목록 구조 오류")
    # 이 쪽에 있어야 할 항목 수 = min(쪽 크기, 남은 건수). 짧은 쪽을 그대로 받으면 일부만 담긴
    # 발표가 '완료'로 저장돼 다시 받지 않게 되므로 거부하고 재시도한다.
    if len(items) != min(PAGE_SIZE, page_total - count):
        raise ApiError("전체 건수에 비해 불완전한 예보 페이지")
    # 같은 쪽 안 또는 앞 쪽과 겹치는 항목이 있으면(서버가 같은 쪽을 다시 준 경우 등) 거부한다.
    page_seen = set()
    for item in items:
        key = _item_key(item, base, nx, ny)
        if key in seen or key in page_seen:
            raise ApiError("중복 예보 항목 또는 반복된 페이지")
        page_seen.add(key)
    return payload, items, page_total


# 한 쪽을 최대 retries번 받는다. 요청은 private_request 안에서 보내 키가 든 URL이 로그에 남지 않게
# 하고, 실패 사이에는 2초·4초…로 늘려 기다린다. 마지막 실패는 쪽 번호와 이 모듈이 만든 사유만 담아
# 올린다(httpx 예외 문자열에는 URL이 들어 있을 수 있음).
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


# 입력을 검증한 뒤 쪽을 차례로 받아 한 발표 전체를 모은다. 반환 envelope: base_datetime, nx, ny,
# pages(응답 원문 전체, 모르는 필드도 보존), items(모든 쪽의 항목 합). 호출 위치: download.
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
    # 인증키가 쿼리 문자열(serviceKey)에 들어가므로 요청 URL 자체가 비밀이다. 그래서 위 _get_page가
    # 로그를 숨긴다.
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
    # 첫 쪽에서 전체 건수를 알게 되고, 모은 항목이 그 수에 닿을 때까지 다음 쪽을 받는다.
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


# 원본 경로 규칙: raw_dir/nx=60_ny=127/base=YYYYMMDDHHMM.json. 파일 하나가 발표 하나라 '이미
# 받았는지'를 파일 존재로 판단한다.
def raw_path(raw_dir: Path, base: datetime, nx: int = SEOUL_NX, ny: int = SEOUL_NY) -> Path:
    """발표 시각과 격자로 저장할 경로를 만든다."""
    base = _base(base)
    _grid(nx, ny)
    return raw_dir / f"nx={nx}_ny={ny}" / f"base={base:%Y%m%d%H%M}.json"


# 호출 위치: download_recent, 실행부(발표를 직접 지정하거나 최신 하나만 받을 때).
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
    # 아직 제공되지 않았을 발표를 요청하면 거부한다(빈 응답을 완료로 저장하지 않게).
    if base > available:
        raise ValueError("아직 제공 시각이 되지 않은 발표")
    target = raw_path(raw_dir, base, nx, ny)
    # 이미 받은 발표는 API를 부르지 않는다. 한 발표의 값은 바뀌지 않으므로 다시 받을 필요가 없다.
    if target.exists():
        return target, "exists"
    target.parent.mkdir(parents=True, exist_ok=True)
    # 잠금 파일을 'x' 모드(이미 있으면 실패하는 배타 생성)로 만들어 같은 발표를 두 수집기가 동시에
    # 쓰지 못하게 한다. 비정상 종료로 남은 잠금은 자동으로 지우지 않으며, 실행 중인 수집이 없는지
    # 확인한 뒤 사람이 지운다(모듈 docstring).
    lock = target.with_suffix(".json.lock")
    try:
        lock_file = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise ApiError("같은 발표를 다른 수집기가 처리 중") from None
    temporary = None
    http = None
    try:
        lock_file.close()
        # 잠금을 얻는 사이에 다른 수집기가 끝냈을 수 있어 다시 확인한다.
        if target.exists():
            return target, "exists"
        http = client or httpx.Client(timeout=30)
        payload = fetch_forecast(http, service_key, base, nx, ny, retries)
        payload["fetched_at"] = fetched_at.isoformat()
        # 같은 폴더의 임시 파일에 다 쓰고 fsync로 디스크에 내린 뒤 os.replace로 이름을 바꾼다. 같은
        # 볼륨 안의 교체는 원자적이라, 읽는 쪽은 완성된 파일이나 없는 파일만 보고 반쯤 쓴 파일은
        # 보지 않는다.
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
        # 여기서 만든 클라이언트만 닫는다(테스트가 넘긴 클라이언트는 호출한 쪽 소유). 실패했으면
        # 임시 파일을 지우고, 성공·실패와 상관없이 잠금을 푼다.
        if client is None and http is not None:
            http.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


# 저장된 모든 발표 원본을 다시 검증하며 (발표 시각, 예보 시각, 항목, 값, 격자, 수집 시각) 긴 형식
# 행으로 펴서 Parquet 하나를 새로 만든다. 원본이 정답이고 Parquet은 언제든 다시 만들 수 있는
# 파생물이다.
# 주의: 발표가 쌓일수록 매번 전체를 다시 만든다.
# 호출 위치: 아래 실행부(수집 뒤 매번).
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
        # 원본이 손상됐거나 형식이 다르면 어느 파일인지와 상관없이 같은 짧은 메시지로 멈춘다(원본
        # 내용을 메시지에 옮기지 않음).
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ApiError("저장된 예보 JSON 형식 오류") from None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Parquet도 임시 파일에 쓴 뒤 os.replace로 바꿔, 적재 작업이 반쯤 쓴 파일을 읽지 않게 한다.
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

    # 실행 방식: (1) --catch-up: 최신 + 빠진 지난 발표(Airflow가 쓰는 방식) (2)
    # --base-date/--base-time: 특정 발표 하나 (3) 둘 다 없으면 최신 발표 하나. 어느 경우든 끝에
    # Parquet을 다시 만든다.
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
    # 인증키는 .env의 DATA_GO_KR_SERVICE_KEY에서 읽는다(커밋하지 않는 파일, 예시는 .env.example).
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
    # 예상한 오류는 메시지만 남기고 종료 코드 1로 끝낸다. 추적 정보(traceback)를 찍지 않아 내부 값이
    # 로그에 나오지 않는다.
    except (ApiError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
