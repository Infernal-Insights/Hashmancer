import requests
import pytest

from hashmancer.utils.github_client import create_issue


class DummyResponse:
    def __init__(self):
        self.status_code = 201

    def raise_for_status(self) -> None:  # pragma: no cover - always ok
        return None

    def json(self):
        return {"url": "https://api.github.com/repos/owner/repo/issues/1"}


def test_create_issue_requires_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        create_issue("owner/repo", "t")


def test_create_issue_posts(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    data = {}

    def fake_post(url, headers=None, json=None, timeout=0):
        data["url"] = url
        data["headers"] = headers
        data["json"] = json
        return DummyResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    res = create_issue("owner/repo", "Title", "Body")
    assert data["url"] == "https://api.github.com/repos/owner/repo/issues"
    assert data["json"] == {"title": "Title", "body": "Body"}
    assert res["url"].endswith("/issues/1")
