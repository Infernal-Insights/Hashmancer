import sys, os
import asyncio
import sys
import os
import json
import pytest

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,

    install_stubs,
    FakeApp,
)

install_stubs()



import hashmancer.server.main as main
from utils import redis_manager
from uuid import UUID

class FakeUploadFile:
    def __init__(self, name, data):
        self.filename = name
        self._data = data
        self._idx = 0
        self.calls = []
    async def read(self, n=-1):
        self.calls.append(n)
        if self._idx >= len(self._data):
            return b""
        if n < 0:
            n = len(self._data) - self._idx
        chunk = self._data[self._idx:self._idx+n]
        self._idx += n
        return chunk

class FakeRedis:
    def __init__(self):
        self.store = {}
        self.queue = []
        self.sets = {}
    def hset(self, key, mapping=None, **kwargs):
        self.store[key] = dict(mapping or {})
    def expire(self, key, ttl):
        pass
    def lpush(self, name, value):
        self.queue.insert(0, value)
    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)
    def smembers(self, key):
        return self.sets.get(key, set())
    def rpush(self, name, value):
        self.queue.append(value)
    def lrem(self, name, count, value):
        pass


def test_import_hashes(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(redis_manager, 'r', fake)
    monkeypatch.setattr(redis_manager.orchestrator_agent, 'build_mask_charsets', lambda: {})
    monkeypatch.setattr(redis_manager.orchestrator_agent, 'estimate_keyspace', lambda m, c: 10)
    ids = [UUID('11111111-1111-1111-1111-111111111111'), UUID('22222222-2222-2222-2222-222222222222')]
    def fake_uuid():
        return ids.pop(0)
    monkeypatch.setattr(redis_manager.uuid, 'uuid4', fake_uuid)
    monkeypatch.setattr(main, 'log_error', lambda *a, **k: None)
    data = (
        b"hash,mask,wordlist,target,hash_mode\n"
        b"h1,?a,wl.txt,t1,1200\n"
        b"h2,,,,1400\n"
    )
    file = FakeUploadFile('hashes.csv', data)
    resp = asyncio.run(main.import_hashes(file, '1000'))
    assert resp['queued'] == 2
    assert resp['errors'] == []
    batch = fake.store['batch:11111111-1111-1111-1111-111111111111']
    assert json.loads(batch['hashes']) == ['h1']
    assert batch['mask'] == '?a'
    assert batch['wordlist'] == 'wl.txt'
    assert batch['hash_mode'] == '1200'


def test_import_hashes_large_file(monkeypatch):
    monkeypatch.setattr(main, 'MAX_IMPORT_SIZE', 100)
    data = b'hash,mask,wordlist,target,hash_mode\n' + b'h,,,,' * 40
    file = FakeUploadFile('big.csv', data)
    with pytest.raises(main.HTTPException):
        asyncio.run(main.import_hashes(file, '0'))


def test_import_hashes_reads_chunks(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(redis_manager, 'r', fake)
    monkeypatch.setattr(redis_manager.orchestrator_agent, 'build_mask_charsets', lambda: {})
    monkeypatch.setattr(redis_manager.orchestrator_agent, 'estimate_keyspace', lambda m, c: 10)
    monkeypatch.setattr(main, 'log_error', lambda *a, **k: None)
    monkeypatch.setattr(main, 'MAX_IMPORT_SIZE', 8192)
    lines = [b'h%d,,,,' % i for i in range(200)]
    data = b'hash,mask,wordlist,target,hash_mode\n' + b'\n'.join(lines) + b'\n'
    file = FakeUploadFile('hashes.csv', data)
    asyncio.run(main.import_hashes(file, '0'))
    assert len([c for c in file.calls if c != -1]) > 1
