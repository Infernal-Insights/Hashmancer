import sys, os
import asyncio
import sys
import os

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,

    install_stubs,
    FakeApp,
)

install_stubs()


import hashmancer.server.main as main


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.lists = {}

    def sadd(self, key, value):
        self.store.setdefault(key, set()).add(value)

    def smembers(self, key):
        return self.store.get(key, set())

    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or {})

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def lpop(self, key):
        vals = self.lists.get(key, [])
        return vals.pop(0) if vals else None

    def hincrby(self, key, field, amount):
        val = int(self.store.get(key, {}).get(field, 0)) + amount
        self.store.setdefault(key, {})[field] = val
        return val

    def hgetall(self, key):
        return dict(self.store.get(key, {}))


def test_flash_queue_on_register(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setattr(main, "verify_signature_with_key", lambda *a: True)
    monkeypatch.setattr(main, "assign_waifu", lambda s: "Agent")
    monkeypatch.setattr(main, "LOW_BW_ENGINE", "hashcat")

    class Req:
        worker_id = "id"
        signature = "s"
        pubkey = "p"
        timestamp = 0
        mode = "eco"
        provider = "on-prem"
        hardware = {"gpus": [{"uuid": "u1", "model": "RTX 3080", "index": 0}]}

    resp = asyncio.run(main.register_worker(Req()))

    assert resp["status"] == "ok"
    assert fake.lists["flash:Agent"]
