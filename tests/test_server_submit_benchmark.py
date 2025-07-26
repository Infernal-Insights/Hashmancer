import sys, os
import asyncio
import sys
import os

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
    def hget(self, key, field):
        return self.store.get(key, {}).get(field)
    def hgetall(self, key):
        return dict(self.store.get(key, {}))


def test_submit_benchmark(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main, 'verify_signature', lambda *a: True)

    class Req:
        worker_id = 'w1'
        gpu_uuid = 'g1'
        engine = 'hashcat'
        hashrates = {'MD5': 1.0, 'SHA1': 2.0, 'NTLM': 3.0}
        timestamp = 0
        signature = 's'

    resp = asyncio.run(main.submit_benchmark(Req()))

    assert resp['status'] == 'ok'
    assert fake.store['benchmark:g1']['engine'] == 'hashcat'
    assert fake.store['benchmark_total:w1']['NTLM'] == 3.0
