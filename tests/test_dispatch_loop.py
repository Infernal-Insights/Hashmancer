import asyncio
import sys
import os
import types

# Stub FastAPI and related heavy modules
fastapi_stub = types.ModuleType('fastapi')
class FakeApp:
    def add_middleware(self, *a, **kw):
        pass
    def on_event(self, *a, **kw):
        return lambda f: f
    def post(self, *a, **kw):
        return lambda f: f
    def get(self, *a, **kw):
        return lambda f: f
    def delete(self, *a, **kw):
        return lambda f: f
    def websocket(self, *a, **kw):
        return lambda f: f
fastapi_stub.FastAPI = lambda: FakeApp()
fastapi_stub.UploadFile = object
fastapi_stub.File = lambda *a, **kw: None
fastapi_stub.WebSocket = object
fastapi_stub.WebSocketDisconnect = type('WebSocketDisconnect', (), {})
class HTTPException(Exception):
    pass
fastapi_stub.HTTPException = HTTPException
sys.modules.setdefault('fastapi', fastapi_stub)

cors_stub = types.ModuleType('fastapi.middleware.cors')
cors_stub.CORSMiddleware = object
sys.modules.setdefault('fastapi.middleware.cors', cors_stub)

resp_stub = types.ModuleType('fastapi.responses')
resp_stub.HTMLResponse = object
sys.modules.setdefault('fastapi.responses', resp_stub)

pydantic_stub = types.ModuleType('pydantic')
class BaseModel:
    pass
pydantic_stub.BaseModel = BaseModel
sys.modules.setdefault('pydantic', pydantic_stub)

# Stub minimal cryptography pieces used by auth_utils
crypto_stub = types.ModuleType('cryptography')
exc_stub = types.ModuleType('cryptography.exceptions')
class InvalidSignature(Exception):
    pass
exc_stub.InvalidSignature = InvalidSignature
prim_stub = types.ModuleType('cryptography.hazmat.primitives')
prim_stub.asymmetric = types.SimpleNamespace(padding=object())
prim_stub.hashes = types.SimpleNamespace(SHA256=lambda: None)
prim_stub.serialization = types.SimpleNamespace(load_pem_public_key=lambda x: None)
crypto_stub.hazmat = types.SimpleNamespace(primitives=prim_stub)
crypto_stub.exceptions = exc_stub
sys.modules.setdefault('cryptography', crypto_stub)
sys.modules.setdefault('cryptography.exceptions', exc_stub)
sys.modules.setdefault('cryptography.hazmat.primitives', prim_stub)
sys.modules.setdefault('cryptography.hazmat.primitives.asymmetric', prim_stub.asymmetric)

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
