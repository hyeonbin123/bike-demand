from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from bike_demand.api.main import app, get_engine


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.local = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        path = values.get("href") if tag == "link" else values.get("src")
        if path and path.startswith("/static/"):
            self.local.append(path)


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


def test_static_mount_does_not_expose_python_or_mask_api_routes(static_client):
    assert static_client.get("/static/main.py").status_code == 404
    assert static_client.get("/static/%2e%2e/main.py").status_code == 404
    assert static_client.get("/static/missing.js").status_code == 404
    schema = static_client.get("/openapi.json").json()
    assert "/" not in schema["paths"]
    assert {"/health", "/stations", "/stations/{station_id}", "/shortage-risk",
            "/predictions/{station_id}"} <= schema["paths"].keys()  # fmt: skip
