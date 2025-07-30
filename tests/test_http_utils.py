import requests
import pytest
from hashmancer.utils import http_utils

class DummyResp:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data


def test_post_with_retry(monkeypatch):
    calls = []
    def flaky_post(url, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise requests.RequestException("fail")
        return DummyResp({"ok": True})

    sleeps = []
    monkeypatch.setattr(http_utils.requests, "post", flaky_post)
    monkeypatch.setattr(http_utils.time, "sleep", lambda s: sleeps.append(s))

    resp = http_utils.post_with_retry("http://sv", retries=5, backoff=0.1)
    assert resp.json()["ok"] is True
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_get_with_retry_failure(monkeypatch):
    monkeypatch.setattr(http_utils.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.RequestException("fail")))
    sleeps = []
    monkeypatch.setattr(http_utils.time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(requests.RequestException):
        http_utils.get_with_retry("http://sv", retries=2, backoff=0.1)
    assert len(sleeps) == 1
