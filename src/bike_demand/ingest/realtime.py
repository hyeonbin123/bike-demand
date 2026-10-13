"""따릉이 bikeList의 모든 페이지를 한 시점의 스냅샷으로 보관한다.

원본: raw_dir/date=YYYY-MM-DD/HHMMSS.json. fetched_at은 수집 시작 시각(KST,
ISO 8601)이고 rentBikeStatus.row는 모든 페이지의 행을 합친 목록이다. pages에는
각 응답 전체를 그대로 남겨 알려지지 않은 필드도 보존한다. 페이지 수집은 순차적이므로
각 대여소의 관측 시각이 완전히 같지는 않다. 같은 초의 파일은 다시 받거나 덮어쓰지 않는다.
bronze: 날짜별 snapshots.parquet. 알려진 필드와 fetched_at을 문자열로 저장하며,
원본만 읽어 통째로 다시 만들므로 중복 적재되지 않는다. 날짜와 시각은 모두 KST이다.

오류에는 응답 본문/요청 URL을 넣지 않으며, httpx의 요청 로그도 수집 중에는 숨긴다.
"""

# [파일 역할] 서비스의 '지금 자전거 수'를 만드는 수집기. 서울 열린데이터광장 실시간
# 대여정보(bikeList, 약 2,700곳)를 1,000행씩 모든 쪽을 받아 한 시점의 스냅샷 원본 JSON으로 저장하고,
# 하루치 원본을 문자열 Parquet(bronze)으로 다시 만든다.
# [왜 원본 + 재생성인가] 원본을 지우지 않고 Parquet을 원본에서 통째로 다시 만들면 같은 날을 몇 번
# 처리해도 행이 중복되지 않는다. 실시간 API는 지금 상태만 주므로 원본이 곧 유일한 기록이다.
# [안전 장치] forecast.py와 같은 원칙: 부분·중복 쪽 거부, 키·요청 주소가 반사된 응답 저장 거부, 요청
# 로그 숨김, 배타 잠금 + 임시 파일 교체(커밋 18f2bdd, 23b1093).
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
    # 행 목록이 사전들의 목록인지, 전체 건수(list_total_count)가 숫자인지, 한 쪽이 요청 크기를 넘지
    # 않는지 확인한다.
    rows = body.get("row")
    total = body.get("list_total_count")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ApiError("응답 row 구조 오류")
    if isinstance(total, bool) or not re.fullmatch(r"[0-9]+", str(total)) or len(rows) > PAGE_SIZE:
        raise ApiError("응답 건수 오류")
    return payload, rows, False


# start번째 행부터 한 쪽을 최대 retries번 받는다. previous_total은 앞 쪽에서 본 전체 건수로, 그보다
# 일찍 '데이터 없음'이 오면 잘린 응답으로 보고 다시 시도한다. 호출 위치: fetch_snapshot.
def _get_page(
    client: httpx.Client, service_key: str, start: int, retries: int, previous_total: int
) -> tuple[dict, list[dict], bool]:
    # 주소 형식: /인증키/json/bikeList/시작/끝/. 키에 /·+ 같은 문자가 있어도 경로가 깨지지 않게
    # safe=''로 모두 인코딩한다. 이 URL 자체가 비밀이라 private_request 안에서만 요청한다.
    url = f"{ENDPOINT}/{quote(service_key, safe='')}/json/bikeList/{start}/{start + PAGE_SIZE - 1}/"
    for attempt in range(1, retries + 1):
        try:
            with private_request():
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
                # httpx 예외의 문자열에는 요청 URL(키 포함)이 들어갈 수 있어 예외 종류 이름만
                # 남긴다. 2초·4초…로 늘려 기다리며 재시도한다.
                detail = str(exc) if isinstance(exc, ApiError) else type(exc).__name__
                raise ApiError(f"bikeList {start}행부터 수집 실패: {detail}") from None
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


# 모든 쪽을 차례로 받아 한 스냅샷으로 합친다. 쪽은 순서대로 받으므로 첫 쪽과 마지막 쪽 대여소의 실제
# 관측 시각은 몇 초 다를 수 있다(모듈 docstring). 호출 위치: download.
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
    # 대여소ID가 비었거나 쪽 사이에서 겹치면(같은 쪽을 다시 받은 경우 등) 스냅샷 전체를 거부한다.
    # realtime_snapshots의 기본 키가 (대여소, 수집 시각)이라 한 스냅샷 안에서 대여소는 한 번만
    # 나와야 한다.
    seen: set[str] = set()
    while True:
        payload, chunk, no_data = _get_page(client, service_key, start, retries, total)
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
        if not no_data:
            total = int(payload["rentBikeStatus"]["list_total_count"])
        pages.append(payload)
        rows.extend(chunk)
        # 끝 조건: 짧은 쪽(마지막 쪽)이 오거나, 앞 쪽이 꽉 차서 다음 쪽을 요청했는데 데이터
        # 없음(INFO-200)이 오면 멈춘다.
        if no_data or len(chunk) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    if not rows:
        raise ApiError("bikeList 데이터 없음")
    # 첫 쪽의 메타 정보(전체 건수, 결과 코드)에 모든 쪽의 행을 합친 rentBikeStatus와, 쪽별 응답
    # 원문(pages)을 함께 저장한다.
    merged = {**pages[0]["rentBikeStatus"], "row": rows}
    return {"rentBikeStatus": merged, "pages": pages}


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
        http = client or httpx.Client(timeout=30)
        snapshot = fetch_snapshot(http, service_key, retries)
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
