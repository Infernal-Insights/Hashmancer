import sys
import os
import pytest


from hashmancer.worker.hashmancer_worker import bios_flasher


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
    monkeypatch.setattr(bios_flasher, "sign_message", lambda *a: "sig")

    mgr = bios_flasher.GPUFlashManager("w", "http://sv", [{"uuid": "u1", "index": 0}])
    mgr.process_task("u1", {"vendor": "nvidia"})

    assert sent["gpu_uuid"] == "u1"
    assert sent["success"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("core_offset", 300),
        ("mem_offset", -300),
        ("voltage", 1200),
    ],
)
def test_apply_flash_settings_invalid(monkeypatch, key, value):
    monkeypatch.setattr(bios_flasher.subprocess, "check_call", lambda *a, **k: None)
    monkeypatch.setattr(bios_flasher, "flash_rom", lambda *a, **k: True)
    captured = {}
    monkeypatch.setattr(bios_flasher.logging, "warning", lambda msg, *a: captured.setdefault("msg", msg))

    settings = {"vendor": "nvidia", key: value}
    ok = bios_flasher.apply_flash_settings({"index": 0}, settings)
    assert not ok
    assert key in captured.get("msg", "")


def test_process_task_logs_missing(monkeypatch):
    calls = {}

    def fake_log_error(component, wid, code, message, exc=None):
        calls["code"] = code

    monkeypatch.setattr(bios_flasher.event_logger, "log_error", fake_log_error)
    monkeypatch.setattr(bios_flasher.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(bios_flasher, "md5_speed", lambda: 1.0)

    def raise_missing(*a, **k):
        raise FileNotFoundError()

    monkeypatch.setattr(bios_flasher, "apply_flash_settings", raise_missing)
    monkeypatch.setattr(bios_flasher, "sign_message", lambda *a: "sig")

    mgr = bios_flasher.GPUFlashManager("w", "http://sv", [{"uuid": "u1", "index": 0}])
    mgr.process_task("u1", {"vendor": "nvidia"})

    assert calls.get("code") == "W006"


def test_process_task_logs_failure(monkeypatch):
    calls = {}

    def fake_log_error(component, wid, code, message, exc=None):
        calls["code"] = code

    monkeypatch.setattr(bios_flasher.event_logger, "log_error", fake_log_error)
    monkeypatch.setattr(bios_flasher.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(bios_flasher, "md5_speed", lambda: 1.0)
    monkeypatch.setattr(bios_flasher, "apply_flash_settings", lambda *a, **k: False)
    monkeypatch.setattr(bios_flasher, "sign_message", lambda *a: "sig")

    mgr = bios_flasher.GPUFlashManager("w", "http://sv", [{"uuid": "u1", "index": 0}])
    mgr.process_task("u1", {"vendor": "nvidia"})

    assert calls.get("code") == "W007"
