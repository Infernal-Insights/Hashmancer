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

import Server.main as main

class FakeRedis:
    def __init__(self):
        self.store = {}
    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or kwargs)
    def hgetall(self, key):
        return dict(self.store.get(key, {}))
    def rpush(self, key, value):
        self.store.setdefault(key, []).append(value)
    def lpop(self, key):
        vals = self.store.get(key, [])
        return vals.pop(0) if vals else None


def test_get_worker_stats(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setitem(main.CONFIG, "watchdog_token", "tok")
    monkeypatch.setattr(main, "WATCHDOG_TOKEN", "tok")

    fake.hset(
        "worker:alpha",
        mapping={
            "status": "idle",
            "hashrate": "5.0",
            "temps": "[70]",
            "power": "[120.5]",
            "utilization": "[80]",
        },
    )

    data = asyncio.run(main.get_worker_stats("alpha", token="tok"))
    assert data["status"] == "idle"
    assert data["hashrate"] == 5.0
    assert data["temps"] == [70]
    assert data["power"] == [120.5]
    assert data["utilization"] == [80]

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


def test_upgrade_and_restart_queue(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setitem(main.CONFIG, "watchdog_token", "tok")
    monkeypatch.setattr(main, "WATCHDOG_TOKEN", "tok")

    resp = asyncio.run(main.upgrade_worker("alpha", token="tok"))
    assert resp["status"] == "queued"
    assert fake.store["command:alpha"] == ["upgrade"]

    resp = asyncio.run(main.restart_worker("alpha", token="tok"))
    assert resp["status"] == "queued"
    assert fake.store["command:alpha"] == ["upgrade", "restart"]

    with pytest.raises(main.HTTPException):
        asyncio.run(main.upgrade_worker("alpha", token="bad"))


def test_get_worker_command(monkeypatch):
    fake = FakeRedis()
    fake.store["command:alpha"] = ["upgrade"]
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setattr(main, "verify_signature", lambda *a: True)

    data = asyncio.run(main.get_worker_command("alpha", 0, "sig"))
    assert data["command"] == "upgrade"
    assert fake.store["command:alpha"] == []

    monkeypatch.setattr(main, "verify_signature", lambda *a: False)
    data = asyncio.run(main.get_worker_command("alpha", 0, "sig"))
    assert data["status"] == "unauthorized"


def test_watchdog_marks_offline(monkeypatch):
    from Server.app.background import watchdog

    class WRedis(FakeRedis):
        def scan_iter(self, pattern):
            return [k for k in self.store if k.startswith("worker:")]

        def hset(self, key, field=None, value=None, mapping=None, **kwargs):
            if mapping is not None:
                self.store.setdefault(key, {}).update(mapping)
            elif field is not None:
                self.store.setdefault(key, {})[field] = value
            elif kwargs:
                self.store.setdefault(key, {}).update(kwargs)

    fake = WRedis()
    fake.hset("worker:old", mapping={"last_seen": "0", "status": "idle"})
    fake.hset("worker:new", mapping={"last_seen": "260", "status": "idle"})

    events = []

    monkeypatch.setattr(main, "r", fake)
    import sys
    sys.modules['main'] = main
    monkeypatch.setattr(watchdog, "STATUS_INTERVAL", 10)
    monkeypatch.setattr(watchdog, "log_watchdog_event", lambda p: events.append(p))
    monkeypatch.setattr(watchdog, "log_error", lambda *a, **k: None)
    monkeypatch.setattr(watchdog.time, "time", lambda: 300)

    async def run_once():
        async def fake_sleep(t):
            raise StopAsyncIteration
        orig = asyncio.sleep
        try:
            asyncio.sleep = fake_sleep
            await watchdog.watchdog_loop()
        except StopAsyncIteration:
            pass
        finally:
            asyncio.sleep = orig

    asyncio.run(run_once())

    assert fake.store["worker:old"]["status"] == "offline"
    assert fake.store["worker:new"]["status"] == "idle"
    assert events and events[0]["worker_id"] == "old"
