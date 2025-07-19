import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Server import hashescom_client

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
