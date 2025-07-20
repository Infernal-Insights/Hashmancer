import asyncio
import sys
import os
import types
import json

# Stub FastAPI and related modules
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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Server'))

import main
import redis_manager
from uuid import UUID


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.queue = []

    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or {})

    def expire(self, key, ttl):
        pass

    def lpush(self, name, value):
        self.queue.insert(0, value)

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
        await main.process_hashes_jobs()
    except StopAsyncIteration:
        pass
    finally:
        asyncio.sleep = asyncio_sleep


def test_process_hashes_jobs(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(redis_manager, 'r', fake)
    monkeypatch.setattr(redis_manager.uuid, 'uuid4', lambda: UUID('11111111-1111-1111-1111-111111111111'))

    fake.store['hashes_job:1'] = {
        'hashes': json.dumps(['a', 'b']),
        'mask': '?d?d',
        'wordlist': 'wl.txt'
    }

    asyncio.run(run_once())

    assert 'batch:11111111-1111-1111-1111-111111111111' in fake.store
    assert fake.queue == ['11111111-1111-1111-1111-111111111111']
    assert fake.store['hashes_job:1']['status'] == 'processed'
