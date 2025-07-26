import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
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


# Add repo paths
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Server'))

import main
import orchestrator_agent
import redis_manager

class FakeRedis:
    def __init__(self):
        self.queue = ['1']
        self.store = {'batch:1': {'hashes': '["h"]', 'mask': '?d?d'}}
        self.jobs = {}
        self.streams = []
        self.lists = {}
        self.prio = {}
    def rpop(self, name):
        return self.queue.pop(0) if self.queue else None
    def hgetall(self, key):
        return self.store.get(key, {})
    def hset(self, key, mapping=None, **kw):
        self.jobs[key] = dict(mapping or kw)
    def expire(self, key, ttl):
        pass
    def xadd(self, stream, mapping):
        self.streams.append((stream, mapping))
    def rpush(self, name, value):
        self.lists.setdefault(name, []).append(value)
    def lrem(self, name, count, value):
        lst = self.lists.get(name, [])
        while value in lst:
            lst.remove(value)
    def zadd(self, name, mapping):
        self.prio.update(mapping)
    def zrevrange(self, name, start, end):
        ordered = sorted(self.prio.items(), key=lambda x: -x[1])
        return [k for k, _ in ordered[start:end+1]]
    def zrem(self, name, member):
        self.prio.pop(member, None)
    def scan_iter(self, pattern):
        return []

def run_once():
    async def _run():
        async def fake_sleep(t):
            raise StopAsyncIteration
        orig = asyncio.sleep
        try:
            asyncio.sleep = fake_sleep
            await main.dispatch_loop()
        except StopAsyncIteration:
            pass
        finally:
            asyncio.sleep = orig
    asyncio.run(_run())

def test_dispatch_loop_queues_job(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(orchestrator_agent, 'r', fake)
    monkeypatch.setattr(redis_manager, 'r', fake)
    monkeypatch.setattr(orchestrator_agent, 'compute_backlog_target', lambda: 1)
    monkeypatch.setattr(orchestrator_agent, 'pending_count', lambda *a, **k: 0)
    monkeypatch.setattr(orchestrator_agent, 'any_darkling_workers', lambda: False)
    monkeypatch.setattr(orchestrator_agent, 'cache_wordlist', lambda p: '')
    monkeypatch.setattr(orchestrator_agent, 'average_benchmark_rate', lambda: 0.0)
    monkeypatch.setattr(orchestrator_agent, 'estimate_keyspace', lambda m, c: 0)
    monkeypatch.setattr(orchestrator_agent, 'compute_batch_range', lambda r, k: (0, 100))
    run_once()
    assert fake.streams
    stream, mapping = fake.streams[0]
    assert stream == orchestrator_agent.JOB_STREAM
    assert fake.jobs[f"job:{mapping['job_id']}"]['batch_id'] == '1'


def test_dispatch_loop_routes_low_bw(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(orchestrator_agent, 'r', fake)
    monkeypatch.setattr(redis_manager, 'r', fake)
    monkeypatch.setattr(orchestrator_agent, 'compute_backlog_target', lambda: 1)
    monkeypatch.setattr(orchestrator_agent, 'pending_count', lambda *a, **k: 0)
    monkeypatch.setattr(orchestrator_agent, 'any_darkling_workers', lambda: True)
    monkeypatch.setattr(orchestrator_agent, 'cache_wordlist', lambda p: '')
    monkeypatch.setattr(orchestrator_agent, 'average_benchmark_rate', lambda: 0.0)
    monkeypatch.setattr(orchestrator_agent, 'estimate_keyspace', lambda m, c: 0)
    monkeypatch.setattr(orchestrator_agent, 'compute_batch_range', lambda r, k: (0, 100))
    run_once()
    assert fake.streams
    stream, mapping = fake.streams[0]
    assert stream == orchestrator_agent.LOW_BW_JOB_STREAM
    assert fake.jobs[f"job:{mapping['job_id']}"]['batch_id'] == '1'


def test_dispatch_priority(monkeypatch):
    fake = FakeRedis()
    fake.queue.append('2')
    fake.store['batch:2'] = {'hashes': '["p"]', 'mask': '?d?d', 'priority': 5}
    fake.prio['2'] = 5
    monkeypatch.setattr(orchestrator_agent, 'r', fake)
    monkeypatch.setattr(redis_manager, 'r', fake)
    monkeypatch.setattr(orchestrator_agent, 'compute_backlog_target', lambda: 1)
    monkeypatch.setattr(orchestrator_agent, 'pending_count', lambda *a, **k: 0)
    monkeypatch.setattr(orchestrator_agent, 'any_darkling_workers', lambda: False)
    monkeypatch.setattr(orchestrator_agent, 'cache_wordlist', lambda p: '')
    monkeypatch.setattr(orchestrator_agent, 'average_benchmark_rate', lambda: 0.0)
    monkeypatch.setattr(orchestrator_agent, 'estimate_keyspace', lambda m, c: 0)
    monkeypatch.setattr(orchestrator_agent, 'compute_batch_range', lambda r, k: (0, 100))
    run_once()
    assert fake.streams
    stream, mapping = fake.streams[0]
    assert fake.jobs[f"job:{mapping['job_id']}"]['batch_id'] == '2'
