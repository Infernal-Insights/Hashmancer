import asyncio
import sys
import os
import types

# Stub FastAPI and pydantic like other server tests
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
    monkeypatch.setattr(main, 'verify_signature', lambda a, b, c: True)

    class Req:
        worker_id = 'w1'
        gpu_uuid = 'g1'
        engine = 'hashcat'
        hashrates = {'MD5': 1.0, 'SHA1': 2.0, 'NTLM': 3.0}
        signature = 's'

    resp = asyncio.run(main.submit_benchmark(Req()))

    assert resp['status'] == 'ok'
    assert fake.store['benchmark:g1']['engine'] == 'hashcat'
    assert fake.store['benchmark_total:w1']['NTLM'] == 3.0
