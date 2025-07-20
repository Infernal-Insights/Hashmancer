import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Worker.hashmancer_worker import bios_flasher


class DummyResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_process_task(monkeypatch):
    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent.update(json)

    monkeypatch.setattr(bios_flasher.requests, "post", fake_post)
    monkeypatch.setattr(bios_flasher, "md5_speed", lambda: 1.0)
    monkeypatch.setattr(bios_flasher, "apply_flash_settings", lambda g, s: True)
    monkeypatch.setattr(bios_flasher, "sign_message", lambda x: "sig")

    mgr = bios_flasher.GPUFlashManager("w", "http://sv", [{"uuid": "u1", "index": 0}])
    mgr.process_task("u1", {"vendor": "nvidia"})

    assert sent["gpu_uuid"] == "u1"
    assert sent["success"]
