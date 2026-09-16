"""인증키를 URL에 담는 공공 API 호출에서 키가 새지 않게 하는 공통 도구.

- ``private_request``: httpx·httpcore 로거는 요청 URL(키 포함)과 응답 헤더를 남긴다.
  요청하는 동안 그 흐름의 로그 기록만 막는다(전역 로그 레벨은 바꾸지 않음).
- ``reflects_secret``: 서버가 요청 URL·키를 응답에 되돌려 보내는 경우를 찾는다.
  그런 응답은 오류 메시지에도, 원본 파일에도 넣지 않는다.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import quote, quote_plus, unquote

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


class _HideWhileActive(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _REQUEST_ACTIVE.get()


@contextmanager
def private_request() -> Iterator[None]:
    names: set[str] = set(_BASE_LOGGERS)
    names.update(
        name
        for name in list(logging.Logger.manager.loggerDict)
        if name.startswith(("httpx.", "httpcore."))
    )
    loggers = [logging.getLogger(name) for name in names]
    hide = _HideWhileActive()
    token = _REQUEST_ACTIVE.set(True)
    for logger in loggers:
        logger.addFilter(hide)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(hide)
        _REQUEST_ACTIVE.reset(token)


def reflects_secret(text: str, secret: str) -> bool:
    """``text``에 키가 그대로·URL 인코딩·JSON 이스케이프된 형태로 들어 있으면 True."""
    if not secret:
        return False
    variants = {secret, quote(secret, safe=""), quote_plus(secret), unquote(secret)}
    variants.update(json.dumps(v, ensure_ascii=False)[1:-1] for v in tuple(variants))
    return any(v and v in text for v in variants)
