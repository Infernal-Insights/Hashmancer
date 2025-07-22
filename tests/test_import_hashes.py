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
exceptions_stub = types.ModuleType("cryptography.exceptions")
class InvalidSignature(Exception):
    pass
exceptions_stub.InvalidSignature = InvalidSignature
primitives_stub = types.ModuleType("cryptography.hazmat.primitives")
asym_stub = types.ModuleType("cryptography.hazmat.primitives.asymmetric")
asym_stub.padding = object()
primitives_stub.asymmetric = asym_stub
primitives_stub.hashes = types.SimpleNamespace(SHA256=lambda: None)
primitives_stub.serialization = types.SimpleNamespace(load_pem_public_key=lambda x: None)
crypto_stub.hazmat = types.SimpleNamespace(primitives=primitives_stub)
crypto_stub.exceptions = exceptions_stub
sys.modules.setdefault("cryptography", crypto_stub)
sys.modules.setdefault("cryptography.exceptions", exceptions_stub)
sys.modules.setdefault("cryptography.hazmat.primitives", primitives_stub)
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", asym_stub)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Server'))

import main
import redis_manager

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


def test_import_hashes(monkeypatch):
    calls = []
    def fake_store_batch(hashes, mask="", wordlist="", ttl=1800, target="any", hash_mode="0"):
        calls.append({
            "hashes": hashes,
            "mask": mask,
            "wordlist": wordlist,
            "target": target,
            "hash_mode": hash_mode,
        })
        return f"id{len(calls)}"
    monkeypatch.setattr(redis_manager, 'store_batch', fake_store_batch)
    monkeypatch.setattr(main, 'log_error', lambda *a, **k: None)
    data = b"hash,mask,wordlist,target\nh1,?a,wl.txt,t1\nh2,,,\n"
    file = FakeUploadFile('hashes.csv', data)
    resp = asyncio.run(main.import_hashes(file, '1000'))
    assert resp == {"queued": 2, "errors": []}
    assert calls == [
        {
            "hashes": ["h1"],
            "mask": "?a",
            "wordlist": "wl.txt",
            "target": "t1",
            "hash_mode": "1000",
        },
        {
            "hashes": ["h2"],
            "mask": "",
            "wordlist": "",
            "target": "any",
            "hash_mode": "1000",
        },
    ]
