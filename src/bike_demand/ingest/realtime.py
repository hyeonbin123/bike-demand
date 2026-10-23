"""따릉이 bikeList의 모든 페이지를 한 시점의 스냅샷으로 보관한다.

원본: raw_dir/date=YYYY-MM-DD/HHMMSS.json. fetched_at은 수집 시작 시각(KST,
ISO 8601)이고 rentBikeStatus.row는 모든 페이지의 행을 합친 목록이다. pages에는
각 응답 전체를 그대로 남겨 알려지지 않은 필드도 보존한다. 페이지 수집은 순차적이므로
각 대여소의 관측 시각이 완전히 같지는 않다. 같은 초의 파일은 다시 받거나 덮어쓰지 않는다.
bronze: 날짜별 snapshots.parquet. 알려진 필드와 fetched_at을 문자열로 저장하며,
원본만 읽어 통째로 다시 만들므로 중복 적재되지 않는다. 날짜와 시각은 모두 KST이다.

건수: 응답의 list_total_count는 전체 대여소 수가 아니라 그 쪽의 행 수다(1,000곳 쪽은 1000,
마지막 쪽은 그 쪽 행 수). 그래서 쪽마다 list_total_count와 행 수가 같은지만 확인한다. 합친
rentBikeStatus의 list_total_count는 첫 쪽 값 그대로라 전체 대여소 수가 아니다(전체는 row의
길이). 쪽 단위로는 잘린 스냅샷(중간 쪽이 짧게 오거나 INFO-200이 일찍 옴)을 정상 끝과 구분할 수
없으므로, 직전 60분 안에 저장한 원본보다 대여소가 5% 넘게 적으면 스냅샷 전체를 다시 받고 끝내
모자라면 저장하지 않는다(download). 60분 안 원본이 없으면 비교하지 않는다.

오류에는 응답 본문/요청 URL을 넣지 않으며, httpx의 요청 로그도 수집 중에는 숨긴다.
"""

# [파일 역할] 서비스의 '지금 자전거 수'를 만드는 수집기. 서울 열린데이터광장 실시간
# 대여정보(bikeList, 약 2,700곳)를 1,000행씩 모든 쪽을 받아 한 시점의 스냅샷 원본 JSON으로 저장하고,
# 하루치 원본을 문자열 Parquet(bronze)으로 다시 만든다.
# [왜 원본 + 재생성인가] 원본을 지우지 않고 Parquet을 원본에서 통째로 다시 만들면 같은 날을 몇 번
# 처리해도 행이 중복되지 않는다. 실시간 API는 지금 상태만 주므로 원본이 곧 유일한 기록이다.
# [안전 장치] forecast.py와 같은 원칙: 부분·중복 쪽 거부, 키·요청 주소가 반사된 응답 저장 거부, 요청
# 로그 숨김, 배타 잠금 + 임시 파일 교체(커밋 18f2bdd, 23b1093). 부분 응답은 두 단계로 막는다: 쪽마다
# 알린 건수와 행 수가 같아야 하고, 스냅샷 전체 대여소 수가 직전 60분 안 원본의 95% 이상이어야 한다.
# [연결] pipelines.collect_realtime(Airflow collect_realtime DAG)이 download → to_parquet →
# serving.load 순으로 부른다. 이 파일의 실행부는 손으로 한 번 수집할 때 쓴다.

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

# httpx: API 호출. pyarrow: 하루치 스냅샷을 문자열 컬럼 Parquet으로 쓴다.
import httpx
import pyarrow as pa
import pyarrow.parquet as pq

# 인증키 보호 공통 도구(요청 로그 숨김, 응답의 키 반사 검사).
from bike_demand.ingest._http import private_request, reflects_secret

# ENDPOINT: 서울 열린데이터광장 OpenAPI 주소. 인증키가 쿼리가 아니라 URL 경로에 들어간다(_get_page).
# 이용 안내가 http 주소(포트 8088)만 제공하고 HTTPS 안내가 없어 http 그대로 둔다. 그래서 키가 암호화
# 없이 전송된다(README 한계).
# PAGE_SIZE: 한 번에 요청하는 행 수(시작~끝 번호 범위 1,000개).
# FIELDS: API 필드 이름 → 프로젝트 컬럼 이름. bronze에는 이 필드들만 옮기고, 원본 JSON에는 응답
# 전체가 남는다. parkingBikeTotCnt가 서비스의 '지금 자전거 수'(bike_count)다.
ENDPOINT = "http://openapi.seoul.go.kr:8088"
PAGE_SIZE = 1000
KST = timezone(timedelta(hours=9))
FIELDS = {
    "rackTotCnt": "rack_count",
    "stationName": "station_name",
    "parkingBikeTotCnt": "bike_count",
    "shared": "shared",
    "stationLatitude": "lat",
    "stationLongitude": "lon",
    "stationId": "station_id",
}
# bronze는 모든 값을 문자열로 둔다(해석은 적재 단계). fetched_at은 수집 시작 시각이라 한 스냅샷의
# 모든 행이 같은 값을 가진다.
SCHEMA = pa.schema([("fetched_at", pa.string()), *[(v, pa.string()) for v in FIELDS.values()]])

# 잘린 스냅샷 검사(download). 새 스냅샷의 대여소 수가 수집 시각 전 TRUNCATION_WINDOW 안에 저장된
# 가장 최근 원본의 MIN_STATION_RATIO배보다 적으면 잘린 것으로 본다.
# - 95%: 2026-09-16~10-03에 저장한 원본 570개는 스냅샷당 2,732~2,747곳(최소/최대 0.994)이었고,
#   60분 안 직전 원본이 있는 548개에서 직전 대비 가장 낮은 비는 0.9993이었다. 그래서 정상 증감으로
#   거부되는 일은 없고, 1,000행 쪽 하나나 마지막 쪽(약 730행)이 빠지는 잘림(약 27~36% 감소)은
#   잡는다. 5%보다 적게 잘린 스냅샷(약 140곳 미만이 빠짐)은 잡지 못한다.
# - 60분: 대여소가 정당하게 크게 줄어도(운영 변경 등) 마지막으로 저장한 큰 원본이 창 밖으로 나가는
#   최대 1시간 뒤에는 비교 기준이 없어져 다시 저장된다. 기준을 계속 이어 붙이면 이런 감소를 영영
#   거부한다.
# 비교는 원본 폴더를 읽기만 하고 보조 파일을 새로 쓰지 않는다.
TRUNCATION_WINDOW = timedelta(minutes=60)
MIN_STATION_RATIO = 0.95
# 원본 파일 이름(수집 시작 시각 HHMMSS.json). 잠금(.json.lock)·임시(.tmp) 파일은 이 모양이 아니다.
_RAW_NAME = re.compile(r"[0-9]{6}\.json")


# 수집 실패 예외. 메시지에 인증키·URL·응답 본문을 넣지 않는다. pipelines의 실행부가 이 예외를 잡아
# 메시지만 출력하고 끝낸다.
class ApiError(RuntimeError):
    """인증키나 요청 URL을 포함하지 않는 수집 오류."""


# 수집 시각을 KST로 맞추는 도우미(원본 경로의 날짜·시각이 KST 기준이 되게).
def _kst(value: datetime) -> datetime:
    """시간대 없는 입력은 KST로 해석하고, 있는 입력은 KST로 변환한다."""
    return value.replace(tzinfo=KST) if value.tzinfo is None else value.astimezone(KST)


# 한 쪽 응답을 검증해 (원본 payload, 행 목록, 데이터 없음 여부)를 돌려준다. INFO-200(해당 데이터
# 없음)은 오류가 아니라 '이 범위에 행이 없음'으로 돌려줘, 대여소 수가 1,000의 배수일 때 마지막 다음
# 쪽을 끝 신호로 쓸 수 있게 한다.
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
    if reflects_secret(serialized, service_key) or ENDPOINT in serialized:
        raise ApiError("응답에 요청 인증정보가 포함됨")
    # 정상 응답은 rentBikeStatus 아래에 RESULT와 row가 있고, 오류 응답은 RESULT가 최상위에 올 수
    # 있어 둘 다 받는다. 성공 코드인데 rentBikeStatus가 없으면 아래에서 따로 거부한다.
    body = payload.get("rentBikeStatus", payload)
    if not isinstance(body, dict) or not isinstance(body.get("RESULT"), dict):
        raise ApiError("응답 구조 오류")
    # INFO-000 정상, INFO-200 데이터 없음, 그 밖은 오류.
    code = body["RESULT"].get("CODE")
    if code == "INFO-200":
        return payload, [], True
    if code != "INFO-000":
        # 오류 코드는 'ERROR-숫자 세 자리' 모양일 때만 메시지에 넣는다. 다른 값은 서버가 넣은 임의
        # 문자열일 수 있어 '알 수 없음'으로 바꾼다.
        safe_code = (
            code if isinstance(code, str) and re.fullmatch(r"ERROR-\d{3}", code) else "알 수 없음"
        )
        raise ApiError(f"API 오류 {safe_code}")
    if "rentBikeStatus" not in payload:
        raise ApiError("응답 rentBikeStatus 없음")
    # 행 목록이 사전들의 목록인지, 쪽 건수(list_total_count)가 숫자인지, 한 쪽이 요청 크기를 넘지
    # 않는지 확인한다.
    rows = body.get("row")
    count = body.get("list_total_count")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ApiError("응답 row 구조 오류")
    if isinstance(count, bool) or not re.fullmatch(r"[0-9]+", str(count)) or len(rows) > PAGE_SIZE:
        raise ApiError("응답 건수 오류")
    # list_total_count는 그 쪽의 행 수다(전체 대여소 수가 아님). 2026-09-16~10-03에 저장한 원본
    # 570개의 모든 쪽에서 두 값이 같았다. 다르면 전송 중 잘렸거나 깨진 쪽이라 _get_page가 다시
    # 받는다. 서버가 준 숫자는 메시지에 넣지 않는다(이 모듈이 센 행 수만).
    if int(count) != len(rows):
        raise ApiError(f"응답 건수 불일치 (행 {len(rows)}개, list_total_count가 다름)")
    return payload, rows, False


# start번째 행부터 한 쪽을 최대 retries번 받는다. 쪽 하나만 보고는 잘림을 알 수 없으므로(짧은 쪽과
# INFO-200은 정상 끝과 같은 모양) 여기서는 응답 형식만 확인하고, 잘림은 download가 스냅샷 단위로
# 찾는다. 호출 위치: fetch_snapshot.
def _get_page(
    client: httpx.Client, service_key: str, start: int, retries: int
) -> tuple[dict, list[dict], bool]:
    # 주소 형식: /인증키/json/bikeList/시작/끝/. 키에 /·+ 같은 문자가 있어도 경로가 깨지지 않게
    # safe=''로 모두 인코딩한다. 이 URL 자체가 비밀이라 private_request 안에서만 요청한다.
    url = f"{ENDPOINT}/{quote(service_key, safe='')}/json/bikeList/{start}/{start + PAGE_SIZE - 1}/"
    for attempt in range(1, retries + 1):
        try:
            with private_request():
                response = client.get(url)
            return _parse_page(response, service_key)
        except (httpx.HTTPError, httpx.InvalidURL, ApiError) as exc:
            if attempt == retries:
                # httpx 예외의 문자열에는 요청 URL(키 포함)이 들어갈 수 있어 예외 종류 이름만
                # 남긴다. 2초·4초…로 늘려 기다리며 재시도한다.
                detail = str(exc) if isinstance(exc, ApiError) else type(exc).__name__
                raise ApiError(f"bikeList {start}행부터 수집 실패: {detail}") from None
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


# 모든 쪽을 차례로 받아 한 스냅샷으로 합친다. 쪽은 순서대로 받으므로 첫 쪽과 마지막 쪽 대여소의 실제
# 관측 시각은 몇 초 다를 수 있다(모듈 docstring). 호출 위치: download(_fetch_complete_snapshot).
def fetch_snapshot(client: httpx.Client, service_key: str, retries: int = 3) -> dict:
    """1~1000, 1001~2000 순으로 짧은 페이지/INFO-200까지 받아 원본을 합친다.

    쪽마다 형식(건수 = 행 수, 대여소ID)은 확인하지만, 스냅샷이 잘렸는지는 이 함수만으로 알 수 없다.
    잘림은 download가 직전 원본과 비교해 거른다.
    """
    if not isinstance(service_key, str) or not service_key.strip():
        raise ValueError("인증키가 필요함")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 1:
        raise ValueError("재시도 횟수는 1 이상이어야 함")
    pages: list[dict] = []
    rows: list[dict] = []
    start = 1
    # 대여소ID가 비었거나 쪽 사이에서 겹치면(같은 쪽을 다시 받은 경우 등) 스냅샷 전체를 거부한다.
    # realtime_snapshots의 기본 키가 (대여소, 수집 시각)이라 한 스냅샷 안에서 대여소는 한 번만
    # 나와야 한다.
    seen: set[str] = set()
    while True:
        payload, chunk, no_data = _get_page(client, service_key, start, retries)
        # 첫 쪽부터 데이터가 없으면 대여소가 하나도 없는 응답이라 실패로 본다.
        if no_data and not rows:
            raise ApiError("bikeList 데이터 없음 (INFO-200)")
        for row in chunk:
            station_id = row.get("stationId")
            if not isinstance(station_id, str) or not station_id.strip():
                raise ApiError("응답 대여소 ID 누락")
            if station_id in seen:
                raise ApiError("응답 대여소 ID 중복")
            seen.add(station_id)
        pages.append(payload)
        rows.extend(chunk)
        # 끝 조건: 짧은 쪽(마지막 쪽)이 오거나, 앞 쪽이 꽉 차서 다음 쪽을 요청했는데 데이터
        # 없음(INFO-200)이 오면 멈춘다. 중간 쪽이 잘려 와도 같은 모양이라 여기서는 구분하지 못한다.
        if no_data or len(chunk) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    if not rows:
        raise ApiError("bikeList 데이터 없음")
    # 첫 쪽의 메타 정보(결과 코드, 첫 쪽의 list_total_count)에 모든 쪽의 행을 합친 rentBikeStatus와,
    # 쪽별 응답 원문(pages)을 함께 저장한다. 합친 list_total_count는 전체 대여소 수가 아니다.
    merged = {**pages[0]["rentBikeStatus"], "row": rows}
    return {"rentBikeStatus": merged, "pages": pages}


# 수집 시각(fetched_at, KST) 전 TRUNCATION_WINDOW 안에 저장된 원본 중 가장 최근 것의 대여소 수(합친
# row의 길이)를 돌려준다. 그런 원본이 없으면 None(비교하지 않음).
# - 창이 자정을 넘으면 전날 날짜 폴더도 본다(00:05 수집의 기준은 전날 23:55 원본).
# - 파일 이름(HHMMSS.json)과 날짜 폴더로 시각을 읽어 수집 시각 뒤의 원본(지난 시각을 지정해 받는
#   경우)과 이번 수집 경로(exclude)는 기준에서 뺀다.
# - 읽을 수 없는 원본(JSON이 아니거나 모양이 다름)은 건너뛰고 그보다 이른 원본을 쓴다.
# 원본 폴더는 읽기만 한다. 호출 위치: download.
def _recent_station_count(raw_dir: Path, fetched_at: datetime, exclude: Path) -> int | None:
    window_start = fetched_at - TRUNCATION_WINDOW
    candidates: list[tuple[datetime, Path]] = []
    day = window_start.date()
    while day <= fetched_at.date():
        for path in (raw_dir / f"date={day:%Y-%m-%d}").glob("*.json"):
            if not _RAW_NAME.fullmatch(path.name) or path == exclude:
                continue
            try:
                saved_at = datetime.strptime(f"{day:%Y-%m-%d} {path.stem}", "%Y-%m-%d %H%M%S")
            except ValueError:
                continue
            saved_at = saved_at.replace(tzinfo=KST)
            if window_start <= saved_at < fetched_at:
                candidates.append((saved_at, path))
        day += timedelta(days=1)
    for _, path in sorted(candidates, reverse=True):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))["rentBikeStatus"]["row"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(rows, list):
            return len(rows)
    return None


# 스냅샷 하나를 받아, 비교 기준(baseline: 직전 60분 안 원본의 대여소 수)의 MIN_STATION_RATIO배보다
# 적으면 잘린 것으로 보고 스냅샷 전체를 처음부터 다시 받는다(최대 retries번, 2초·4초…로 늘려
# 기다림). 끝내 모자라면 ApiError를 내 download가 저장하지 않게 한다. baseline이 None이면 첫
# 스냅샷을 그대로 돌려준다. 다시 받은 스냅샷의 관측은 수집 시작 시각(fetched_at)보다 몇 초~몇십 초
# 늦지만, 쪽을 차례로 받는 것과 같은 정도의 차이라 fetched_at은 바꾸지 않는다. 호출 위치: download.
def _fetch_complete_snapshot(
    client: httpx.Client, service_key: str, retries: int, baseline: int | None
) -> dict:
    for attempt in range(1, retries + 1):
        snapshot = fetch_snapshot(client, service_key, retries)
        count = len(snapshot["rentBikeStatus"]["row"])
        if baseline is None or count >= MIN_STATION_RATIO * baseline:
            return snapshot
        if attempt == retries:
            minutes = int(TRUNCATION_WINDOW.total_seconds() // 60)
            raise ApiError(
                f"대여소 {count}곳: 직전 {minutes}분 안 스냅샷 {baseline}곳의 "
                f"{MIN_STATION_RATIO:.0%} 미만이라 잘린 응답으로 보고 저장하지 않음"
            )
        time.sleep(2 * attempt)
    raise AssertionError("unreachable")


# 경로 규칙: raw_dir/date=YYYY-MM-DD/HHMMSS.json(수집 시작 시각, KST). 날짜 폴더를 수집 시각으로
# 정하는 것이 자정 경계에서 적재가 빠지지 않게 하는 기준이다(pipelines.py 모듈 docstring, 커밋
# d19a3e8).
def raw_path(raw_dir: Path, fetched_at: datetime) -> Path:
    """수집 시작 시각의 KST 날짜/초로 원본 경로를 정한다."""
    local = _kst(fetched_at)
    return raw_dir / f"date={local:%Y-%m-%d}" / f"{local:%H%M%S}.json"


# 호출 위치: pipelines.collect_realtime, 아래 실행부.
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
    같은 raw_dir에 수집 시각 전 60분 안에 저장한 원본이 있으면, 새 스냅샷의 대여소 수가 그중 가장
    최근 원본의 95% 이상일 때만 저장한다. 모자라면 스냅샷 전체를 retries번까지 다시 받고, 끝내
    모자라면 ApiError를 내고 아무것도 저장하지 않는다(실시간 API는 지금 상태만 주므로 잘린 원본이
    그 시각의 유일한 기록이 되는 것을 막는다). 정당하게 대여소가 크게 줄면 최대 1시간 동안 수집이
    거부되다가, 마지막 큰 원본이 60분 창을 벗어나면 다시 저장된다.
    """
    fetched_at = _kst(fetched_at or datetime.now(KST))
    target = raw_path(raw_dir, fetched_at)
    # 같은 초의 원본이 이미 있으면 다시 받지 않고 기존 파일을 보존한다.
    if target.exists():
        return target, "exists"
    target.parent.mkdir(parents=True, exist_ok=True)
    # O_CREAT | O_EXCL: 잠금 파일이 이미 있으면 만들기가 실패하는 배타 생성. 같은 시각을 두 수집기가
    # 동시에 쓰지 못하게 한다. 잠금을 얻은 뒤 원본이 생겼는지 한 번 더 확인한다.
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
        # 잘림 비교 기준은 잠금을 얻은 뒤, 받기 전에 정한다(이번 수집 경로는 빼고).
        baseline = _recent_station_count(raw_dir, fetched_at, target)
        http = client or httpx.Client(timeout=30)
        snapshot = _fetch_complete_snapshot(http, service_key, retries, baseline)
        snapshot["fetched_at"] = fetched_at.isoformat()
        # 같은 폴더 임시 파일에 다 쓴 뒤 os.replace로 바꿔, Parquet 재생성이 반쯤 쓴 원본을 읽지
        # 않게 한다.
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
        # 직접 만든 클라이언트만 닫고, 남은 임시 파일과 잠금은 성공·실패와 상관없이 지운다.
        if http is not None and client is None:
            http.close()
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


# 그날 폴더의 모든 원본에서 FIELDS 컬럼만 뽑아 행으로 펴고, 빈 문자열은 None(결측)으로 바꾼다.
# 원본만 읽어 통째로 다시 만들므로 몇 번 돌려도 같은 결과다.
# 호출 위치: pipelines.collect_realtime(실행마다 자기만의 임시 경로에 만들어 그 파일을 적재), 아래
# 실행부.
# 주의: 10분마다 그날 전체를 다시 만들고 다시 적재한다.
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
    # 임시 파일에 쓰고 os.replace로 원자적으로 바꾼다. 실패하면 임시 파일만 지워지고 기존 Parquet은
    # 그대로다.
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

    # 손으로 스냅샷 하나를 받고 그날 Parquet을 다시 만든다(DB 적재는 하지 않음). Airflow는 이 실행부
    # 대신 pipelines collect-realtime을 쓴다. 인증키는 .env의 SEOUL_OPEN_API_KEY.
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
