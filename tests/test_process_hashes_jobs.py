import sys, os
import asyncio
import sys
import os
import json
from pathlib import Path
import types

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,

    install_stubs,
    FakeApp,
)

install_stubs()



from hashmancer.server.app.background.hashes_jobs import process_hashes_jobs
import hashmancer.server.main as main
sys.modules['main'] = main
from utils import redis_manager
from uuid import UUID


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.queue = []
        self.lists = []
        self.sets = {}
        self.prio = {}

    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or {})

    def expire(self, key, ttl):
        pass

    def lpush(self, name, value):
        self.queue.insert(0, value)

    def rpush(self, name, value):
        self.lists.append((name, value))

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def smembers(self, key):
        return self.sets.get(key, set())

    def lrem(self, name, count, value):
        pass

    def zadd(self, name, mapping):
        self.prio.update(mapping)

    def zrevrange(self, name, start, end):
        ordered = sorted(self.prio.items(), key=lambda x: -x[1])
        return [k for k, _ in ordered[start:end+1]]

    def zrem(self, name, member):
        self.prio.pop(member, None)

    def hget(self, key, field):
        return self.store.get(key, {}).get(field)

    def hgetall(self, key):
        return dict(self.store.get(key, {}))

    def scan_iter(self, pattern):
        if pattern == 'hashes_job:*':
            return [k for k in self.store if k.startswith('hashes_job:')]
        return []


async def run_once():
    async def fake_sleep(t):
        raise StopAsyncIteration
    asyncio_sleep = asyncio.sleep
    try:
        asyncio.sleep = fake_sleep
        await process_hashes_jobs()
    except StopAsyncIteration:
        pass
    finally:
        asyncio.sleep = asyncio_sleep


def test_process_hashes_jobs(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(redis_manager, 'r', fake)
    monkeypatch.setattr(redis_manager.uuid, 'uuid4', lambda: UUID('11111111-1111-1111-1111-111111111111'))
    monkeypatch.setattr(redis_manager.orchestrator_agent, 'build_mask_charsets', lambda: {})
    monkeypatch.setattr(redis_manager.orchestrator_agent, 'estimate_keyspace', lambda m, c: 10)

    fake.store['hashes_job:1'] = {
        'hashes': json.dumps(['a', 'b']),
        'mask': '?d?d',
        'wordlist': 'wl.txt',
        'algorithmName': 'MD5',
        'algorithmId': '0'
    }

    asyncio.run(run_once())

    assert 'batch:11111111-1111-1111-1111-111111111111' in fake.store
    assert fake.queue == ['11111111-1111-1111-1111-111111111111']
    assert fake.store['hashes_job:1']['status'] == 'processed'


def test_process_hashes_known(monkeypatch, tmp_path):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(redis_manager, 'r', fake)
    monkeypatch.setattr(redis_manager.uuid, 'uuid4', lambda: UUID('22222222-2222-2222-2222-222222222222'))
    monkeypatch.setattr(redis_manager.orchestrator_agent, 'build_mask_charsets', lambda: {})
    monkeypatch.setattr(redis_manager.orchestrator_agent, 'estimate_keyspace', lambda m, c: 10)

    md5 = '5d41402abc4b2a76b9719d911017c592'
    fake.store['found:map'] = {md5: 'hello'}

    uploaded = {}

    def mock_upload(algo_id, path):
        uploaded['algo'] = algo_id
        uploaded['data'] = Path(path).read_text().strip()
        return True

    monkeypatch.setitem(sys.modules, 'hashescom_client', types.SimpleNamespace(upload_founds=mock_upload))

    fake.store['hashes_job:2'] = {
        'hashes': json.dumps([md5, 'deadbeef']),
        'mask': '',
        'wordlist': '',
        'algorithmName': 'MD5',
        'algorithmId': '0'
    }

    asyncio.run(run_once())

    assert uploaded['algo'] == '0'
    assert uploaded['data'] == f'{md5}:hello'
    assert 'batch:22222222-2222-2222-2222-222222222222' in fake.store
    batch = fake.store['batch:22222222-2222-2222-2222-222222222222']
    assert json.loads(batch['hashes']) == ['deadbeef']


def test_predefined_mask_batches(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(redis_manager, 'r', fake)
    ids = [UUID('33333333-3333-3333-3333-333333333333'), UUID('44444444-4444-4444-4444-444444444444')]
    monkeypatch.setattr(redis_manager.uuid, 'uuid4', lambda: ids.pop(0))
    monkeypatch.setattr(redis_manager.orchestrator_agent, 'build_mask_charsets', lambda: {})
    monkeypatch.setattr(redis_manager.orchestrator_agent, 'estimate_keyspace', lambda m, c: 10)
    monkeypatch.setattr(main, 'PREDEFINED_MASKS', ['?l?d'])

    fake.store['hashes_job:3'] = {
        'hashes': json.dumps(['dead']),
        'mask': '?d?d',
        'wordlist': '',
        'algorithmName': 'MD5',
        'algorithmId': '0',
        'priority': '2'
    }

    asyncio.run(run_once())

    assert 'batch:33333333-3333-3333-3333-333333333333' in fake.store
    assert 'batch:44444444-4444-4444-4444-444444444444' in fake.store
    assert fake.prio['33333333-3333-3333-3333-333333333333'] == 2
    assert fake.prio['44444444-4444-4444-4444-444444444444'] == 3
