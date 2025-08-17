import os
import sys
import asyncio


from hashmancer.server import hashescom_client

class DummyResp:
    def __init__(self, json_data, status=200):
        self._json = json_data
        self.status_code = status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP error")


def test_upload_founds_success(monkeypatch, tmp_path):
    def mock_post(url, files=None, data=None, timeout=None):
        return DummyResp({"success": True})

    monkeypatch.setattr(hashescom_client.requests, "post", mock_post)
    sample = tmp_path / "f.txt"
    sample.write_text("hash:pass")

    assert hashescom_client.upload_founds(0, str(sample))


def test_upload_founds_failure(monkeypatch, tmp_path):
    def mock_post(url, files=None, data=None, timeout=None):
        return DummyResp({"success": False})

    monkeypatch.setattr(hashescom_client.requests, "post", mock_post)
    sample = tmp_path / "f.txt"
    sample.write_text("hash:pass")

    assert not hashescom_client.upload_founds(0, str(sample))


def test_fetch_jobs_sync_headers(monkeypatch):
    seen = {}

    def mock_get(url, timeout=None, headers=None):
        seen['url'] = url
        seen['headers'] = headers
        return DummyResp({"success": True, "list": [1]})

    monkeypatch.setattr(hashescom_client.requests, "get", mock_get)

    result = hashescom_client._fetch_jobs_sync("http://example", {"X-Api-Key": "k"})
    assert result == [1]
    assert seen['headers'] == {"X-Api-Key": "k"}


def test_fetch_jobs_passes_header(monkeypatch):
    captured = {}

    def fake_sync(url, headers=None):
        captured['url'] = url
        captured['headers'] = headers
        return []

    monkeypatch.setattr(hashescom_client, "_fetch_jobs_sync", fake_sync)
    monkeypatch.setattr(hashescom_client, "aiohttp", None)
    monkeypatch.setattr(hashescom_client, "HASHES_API", "abc")

    asyncio.run(hashescom_client.fetch_jobs())

    assert captured['url'] == "https://hashes.com/en/api/jobs"
    assert captured['headers'] == {"X-Api-Key": "abc"}
