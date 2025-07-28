import asyncio

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,
    pydantic_stub,
    install_stubs,
    FakeApp,
)

install_stubs()

import hashmancer.server.main as main


class FakeRedis:
    def __init__(self):
        self.store = {}

    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or kwargs)

    def hget(self, key, field):
        return self.store.get(key, {}).get(field)

    def hgetall(self, key):
        return dict(self.store.get(key, {}))


def test_high_temp_triggers_alert(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setattr(main, "verify_signature", lambda *a: True)
    monkeypatch.setattr(main, "TEMP_THRESHOLD", 60)
    events = []
    monkeypatch.setattr(
        main,
        "log_error",
        lambda *a, **k: events.append(a[2] if len(a) > 2 else None),
    )

    class Req:
        name = "alpha"
        status = "idle"
        timestamp = 0
        signature = "s"
        temps = [70]
        power = None
        utilization = None
        progress = None

    resp = asyncio.run(main.set_worker_status(Req()))
    assert resp["status"] == "ok"
    assert "H001" in events
