"""인증키를 URL에 담는 공공 API 호출에서 키가 새지 않게 하는 공통 도구.

- ``private_request``: httpx·httpcore 로거는 요청 URL(키 포함)과 응답 헤더를 남긴다.
  요청하는 동안 그 흐름의 로그 기록만 막는다(전역 로그 레벨은 바꾸지 않음).
- ``reflects_secret``: 서버가 요청 URL·키를 응답에 되돌려 보내는 경우를 찾는다.
  그런 응답은 오류 메시지에도, 원본 파일에도 넣지 않는다.
"""

# [파일 역할] 공공 API 세 곳(ASOS 날씨, 단기예보, 따릉이 실시간)을 부르는 수집 모듈이 함께 쓰는
# '인증키 보호' 도구.
# [왜 필요한가] 공공데이터포털·서울 열린데이터광장 API는 인증키를 URL 쿼리(serviceKey)나 경로에
# 넣는다. 그래서 (1) httpx가 INFO 수준에서 요청 URL을 로그로 남기면 키가 Airflow 로그에 찍히고, (2)
# 서버가 오류 응답에 요청 URL을 되돌려 보내면 오류 메시지나 저장한 원본 파일에 키가 남는다. Codex
# 검토 라운드 1에서 나온 문제이고(커밋 7ae19b3), README 설계 판단의 '인증키가 새지 않게' 항목이
# 이것이다.
# [연결] ingest/weather.py, ingest/forecast.py, ingest/realtime.py가 요청을 private_request 안에서
# 보내고, 응답을 저장하기 전에 reflects_secret으로 검사한다.

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import quote, quote_plus, unquote

# _REQUEST_ACTIVE: 지금 실행 흐름이 '키가 든 요청 중'인지 표시. ContextVar라서 스레드·비동기
# 작업마다 값이 따로 있어, 한 흐름의 표시가 다른 흐름의 로그를 막지 않는다.
# _BASE_LOGGERS: httpx와 httpcore가 요청·연결 정보를 남기는 로거 이름들.
_REQUEST_ACTIVE: ContextVar[bool] = ContextVar("private_request_active", default=False)
_BASE_LOGGERS = (
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
)


# 로그 필터: 현재 흐름이 키가 든 요청 중이면 기록을 버린다(False를 돌려줌). 그 밖의 기록은 그대로
# 통과한다.
class _HideWhileActive(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _REQUEST_ACTIVE.get()


# with 블록 안에서 보내는 요청의 httpx·httpcore 로그만 숨긴다. 전역 로그 레벨을 바꾸지 않으므로 다른
# 코드의 로그 설정에 영향이 없다.
# 호출 위치: weather.py, forecast.py, realtime.py의 실제 HTTP 호출.
@contextmanager
def private_request() -> Iterator[None]:
    # 로거에 붙인 필터는 그 로거에 직접 남긴 기록에만 적용되고 자식 로거에서 올라온 기록에는
    # 적용되지 않는다. 그래서 이미 만들어진 httpx.*·httpcore.* 자식 로거를 모두 찾아 각각에 필터를
    # 붙인다.
    names: set[str] = set(_BASE_LOGGERS)
    names.update(
        name
        for name in list(logging.Logger.manager.loggerDict)
        if name.startswith(("httpx.", "httpcore."))
    )
    loggers = [logging.getLogger(name) for name in names]
    hide = _HideWhileActive()
    # 표시를 켜고 필터를 붙인다. finally에서 필터를 떼고 표시를 이전 값으로 되돌려, 예외가 나도 숨김
    # 상태가 남지 않게 한다.
    token = _REQUEST_ACTIVE.set(True)
    for logger in loggers:
        logger.addFilter(hide)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(hide)
        _REQUEST_ACTIVE.reset(token)


# 응답 본문에 키가 어떤 형태로든 들어 있는지 검사한다. 키는 URL에 들어갈 때 인코딩될 수 있고(quote,
# quote_plus), 이미 인코딩된 키를 받았을 수도 있으며(unquote), JSON 응답 안에서는 이스케이프될 수
# 있어 이 형태들을 모두 만들어 비교한다.
# True면 호출하는 쪽은 그 응답을 원본 파일로 저장하지 않고, 오류 메시지에도 본문을 넣지 않는다.
# 호출 위치: weather.py, forecast.py, realtime.py.
def reflects_secret(text: str, secret: str) -> bool:
    """``text``에 키가 그대로·URL 인코딩·JSON 이스케이프된 형태로 들어 있으면 True."""
    if not secret:
        return False
    variants = {secret, quote(secret, safe=""), quote_plus(secret), unquote(secret)}
    # json.dumps 결과의 양끝 큰따옴표를 떼면 JSON 문자열 안에 들어갈 때의 이스케이프된 모양이 된다.
    variants.update(json.dumps(v, ensure_ascii=False)[1:-1] for v in tuple(variants))
    return any(v and v in text for v in variants)
