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


import hashmancer.server.main as main

class FakeRedis:
    def __init__(self):
        self.store = {}
    def hset(self, key, field=None, value=None, mapping=None):
        if mapping is not None:
            self.store.setdefault(key, {}).update(mapping)
        else:
            self.store.setdefault(key, {})[field] = value
    def hget(self, key, field):
        return self.store.get(key, {}).get(field)
    def scan_iter(self, pattern):
        return [k for k in self.store if k.startswith('hashes_job:')]


def test_set_hashes_job_config(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    req = type('Req', (), {'job_id': '1', 'priority': 5})
    resp = asyncio.run(main.set_hashes_job_config(req()))
    assert resp['status'] == 'ok'
    assert fake.store['hashes_job:1']['priority'] == 5
    cfg = asyncio.run(main.get_hashes_job_config())
    assert cfg == {'1': 5}


def test_set_hashes_algo_params(monkeypatch):
    monkeypatch.setattr(main, 'CONFIG', {})
    saved = {}
    monkeypatch.setattr(main, 'save_config', lambda: saved.setdefault('done', True))
    req = type('Req', (), {'algo': 'MD5', 'params': {'mask_length': 8, 'rule': 'best.rule'}})
    resp = asyncio.run(main.set_hashes_algo_params(req()))
    assert resp['status'] == 'ok'
    assert main.CONFIG['hashes_algo_params']['md5'] == {'mask_length': 8, 'rule': 'best.rule'}
    assert main.HASHES_ALGO_PARAMS['md5'] == {'mask_length': 8, 'rule': 'best.rule'}
    assert saved.get('done')
    params = asyncio.run(main.get_hashes_algo_params())
    assert params == {'md5': {'mask_length': 8, 'rule': 'best.rule'}}
