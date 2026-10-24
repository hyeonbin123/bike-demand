# [무엇을 검증하나] API 서버가 함께 내주는 대시보드(GET /의 index.html과 /static의 CSS·JS)가 DB 없이
# 열리고, HTML이 가리키는 정적 파일이 모두 실제로 받아지는지, 그리고 정적 파일 경로가 파이썬 소스를
# 노출하거나 API 경로를 가리지 않는지 확인한다.
# [왜 중요한가] 대시보드는 별도 빌드 없이 API와 같은 서버에서 나간다(README). 정적 파일 이름이
# 바뀌어 HTML과 어긋나거나, /static 마운트가 api 폴더를 통째로 열면 화면이 깨지거나 소스가 새어
# 나간다.

from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

# 앱과 DB 엔진 의존성. 엔진 의존성을 '부르면 실패'로 바꿔 정적 응답이 DB를 건드리지 않는지 본다.
# STATIC_DIR은 압축을 풀어 받은 본문을 원본 파일과 비교하는 데 쓴다.
from bike_demand.api.main import STATIC_DIR, app, get_engine


# HTML에서 <link href>와 <script src> 등의 경로 중 /static/으로 시작하는 로컬 파일만 모은다.
# CDN(Leaflet) 같은 외부 주소는 제외한다.
class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.local = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        path = values.get("href") if tag == "link" else values.get("src")
        if path and path.startswith("/static/"):
            self.local.append(path)


# get_engine을 호출되면 바로 테스트를 실패시키는 함수로 바꾼 TestClient. 대시보드를 여는 것만으로 DB
# 접속이 일어나면 DB 장애 때 화면 자체가 안 뜨게 되므로 이를 막는다. 끝나면 이전 override를 그대로
# 되돌린다.
@pytest.fixture
def static_client():
    def no_database():
        pytest.fail("대시보드 HTML과 정적 파일은 DB에 접속하지 않아야 한다")

    previous = app.dependency_overrides.copy()
    app.dependency_overrides[get_engine] = no_database
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


# GET /이 한국어 HTML을 주고, 화면에 '반납은 반영하지 않은 값'이라는 한계 문구가 들어
# 있으며(docs/api.md: 응답과 화면에 이 한계를 적는다), HTML이 참조하는 로컬 파일이 정확히
# dashboard.css·dashboard.js 둘이고 각각 올바른 content-type으로 받아지는지 확인한다.
def test_dashboard_html_and_referenced_assets_without_database(static_client):
    response = static_client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<html lang="ko">' in response.text
    assert "반납은 반영하지 않은 값" in response.text
    assets = Assets()
    assets.feed(response.text)
    assert set(assets.local) == {"/static/dashboard.css", "/static/dashboard.js"}
    for path in assets.local:
        asset = static_client.get(path)
        assert asset.status_code == 200
        assert len(asset.content) > 0
        assert (
            "text/css" in asset.headers["content-type"]
            if path.endswith(".css")
            else ("javascript" in asset.headers["content-type"])
        )


# 보안·라우팅 확인: /static은 api/static 폴더만 내주므로 main.py는 404이고, URL 인코딩한
# '..'(%2e%2e)로 상위 폴더에 올라가는 경로 탐색도 404다. 대시보드 경로(/)는 OpenAPI 문서에서 빠져
# 있고(include_in_schema=False), API 경로 다섯 개는 그대로 남아 있다.
def test_static_mount_does_not_expose_python_or_mask_api_routes(static_client):
    assert static_client.get("/static/main.py").status_code == 404
    assert static_client.get("/static/%2e%2e/main.py").status_code == 404
    assert static_client.get("/static/missing.js").status_code == 404
    schema = static_client.get("/openapi.json").json()
    assert "/" not in schema["paths"]
    assert {"/health", "/stations", "/stations/{station_id}", "/shortage-risk",
            "/predictions/{station_id}"} <= schema["paths"].keys()  # fmt: skip


# 전송량: 1 KB 이상 응답은 요청에 Accept-Encoding: gzip이 있으면 gzip으로 압축해 보낸다(main.py의
# GZipMiddleware, minimum_size=1024, docs/api.md 공통). 대시보드 JS(약 21 KB)가 압축되고 httpx가
# 풀어 준 본문은 원본 파일과 같아야 한다. 받는 쪽이 압축을 받지 않겠다고 하면(identity) 그대로
# 보내고, 1 KB 미만 응답(없는 정적 파일의 404 JSON)은 압축하지 않는다. Vary: Accept-Encoding은 중간
# 캐시가 압축본과 원본을 섞지 않게 하는 헤더다.
def test_static_files_are_gzipped_only_when_large_and_accepted(static_client):
    source = (STATIC_DIR / "dashboard.js").read_bytes()
    compressed = static_client.get("/static/dashboard.js", headers={"Accept-Encoding": "gzip"})
    assert compressed.status_code == 200
    assert compressed.headers.get("content-encoding") == "gzip"
    assert "accept-encoding" in compressed.headers.get("vary", "").lower()
    assert compressed.content == source  # httpx가 압축을 풀어 준 본문
    assert compressed.num_bytes_downloaded < len(source)  # 실제로 받은 바이트는 원본보다 적다

    plain = static_client.get("/static/dashboard.js", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
    assert plain.content == source

    small = static_client.get("/static/missing.js", headers={"Accept-Encoding": "gzip"})
    assert small.status_code == 404
    assert "content-encoding" not in small.headers
