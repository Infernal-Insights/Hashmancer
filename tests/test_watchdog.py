import asyncio
import json
import pytest

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,
    pydantic_stub,
    install_stubs,
    FakeApp,
)

install_stubs()

import main

class FakeRedis:
    def __init__(self):
        self.store = {}
    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or kwargs)
    def hgetall(self, key):
        return dict(self.store.get(key, {}))
    def rpush(self, key, value):
        self.store.setdefault(key, []).append(value)


def test_get_worker_stats(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setitem(main.CONFIG, "watchdog_token", "tok")
    monkeypatch.setattr(main, "WATCHDOG_TOKEN", "tok")

    fake.hset("worker:alpha", mapping={"status": "idle", "hashrate": "5.0", "temps": "[70]"})

    data = asyncio.run(main.get_worker_stats("alpha", token="tok"))
    assert data["status"] == "idle"
    assert data["hashrate"] == 5.0
    assert data["temps"] == [70]

    with pytest.raises(main.HTTPException):
        asyncio.run(main.get_worker_stats("alpha", token="bad"))


def test_reboot_worker(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setitem(main.CONFIG, "watchdog_token", "tok")
    monkeypatch.setattr(main, "WATCHDOG_TOKEN", "tok")

    resp = asyncio.run(main.reboot_worker("alpha", token="tok"))
    assert resp["status"] == "queued"
    assert fake.store["reboot:alpha"] == ["reboot"]

    with pytest.raises(main.HTTPException):
        asyncio.run(main.reboot_worker("alpha", token="bad"))
