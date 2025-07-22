import asyncio
import sys
import os
import types
import json

# Stub modules as in other server tests
fastapi_stub = types.ModuleType("fastapi")
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
fastapi_stub.WebSocketDisconnect = type("WebSocketDisconnect", (), {})
class HTTPException(Exception):
    pass
fastapi_stub.HTTPException = HTTPException
sys.modules.setdefault("fastapi", fastapi_stub)

cors_stub = types.ModuleType("fastapi.middleware.cors")
cors_stub.CORSMiddleware = object
sys.modules.setdefault("fastapi.middleware.cors", cors_stub)

resp_stub = types.ModuleType("fastapi.responses")
resp_stub.HTMLResponse = object
resp_stub.FileResponse = object
sys.modules.setdefault("fastapi.responses", resp_stub)

pydantic_stub = types.ModuleType("pydantic")
class BaseModel:
    pass
pydantic_stub.BaseModel = BaseModel
sys.modules.setdefault("pydantic", pydantic_stub)

crypto_stub = types.ModuleType("cryptography")
exc_stub = types.ModuleType("cryptography.exceptions")
class InvalidSignature(Exception):
    pass
exc_stub.InvalidSignature = InvalidSignature
prim_stub = types.ModuleType("cryptography.hazmat.primitives")
prim_stub.asymmetric = types.SimpleNamespace(padding=object())
prim_stub.hashes = types.SimpleNamespace(SHA256=lambda: None)
prim_stub.serialization = types.SimpleNamespace(load_pem_public_key=lambda x: None)
crypto_stub.hazmat = types.SimpleNamespace(primitives=prim_stub)
crypto_stub.exceptions = exc_stub
sys.modules.setdefault("cryptography", crypto_stub)
sys.modules.setdefault("cryptography.exceptions", exc_stub)
sys.modules.setdefault("cryptography.hazmat.primitives", prim_stub)
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", prim_stub.asymmetric)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Server'))

import main
import redis_manager
from uuid import UUID

class FakeUploadFile:
    def __init__(self, name, data):
        self.filename = name
        self._data = data
        self._idx = 0
    async def read(self, n=-1):
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
    def hset(self, key, mapping=None, **kwargs):
        self.store[key] = dict(mapping or {})
    def expire(self, key, ttl):
        pass
    def lpush(self, name, value):
        self.queue.insert(0, value)


def test_import_hashes(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(redis_manager, 'r', fake)
    ids = [UUID('11111111-1111-1111-1111-111111111111'), UUID('22222222-2222-2222-2222-222222222222')]
    def fake_uuid():
        return ids.pop(0)
    monkeypatch.setattr(redis_manager.uuid, 'uuid4', fake_uuid)
    monkeypatch.setattr(main, 'log_error', lambda *a, **k: None)
    data = b"hash,mask,wordlist,target\nh1,?a,wl.txt,t1\nh2,,,\n"
    file = FakeUploadFile('hashes.csv', data)
    resp = asyncio.run(main.import_hashes(file, '1000'))
    assert resp['queued'] == 2
    assert resp['errors'] == []
    batch = fake.store['batch:11111111-1111-1111-1111-111111111111']
    assert json.loads(batch['hashes']) == ['h1']
    assert batch['mask'] == '?a'
    assert batch['wordlist'] == 'wl.txt'
    assert batch['hash_mode'] == '1000'
