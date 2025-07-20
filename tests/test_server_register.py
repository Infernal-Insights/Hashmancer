import asyncio
import sys
import os
import types

# Stub out FastAPI and Pydantic to avoid heavy dependencies during testing
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

# Stub minimal cryptography pieces used by auth_utils
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
primitives_stub.serialization = types.SimpleNamespace(
    load_pem_public_key=lambda x: None
)
crypto_stub.hazmat = types.SimpleNamespace(primitives=primitives_stub)
crypto_stub.exceptions = exceptions_stub
sys.modules.setdefault("cryptography", crypto_stub)
sys.modules.setdefault("cryptography.exceptions", exceptions_stub)
sys.modules.setdefault(
    "cryptography.hazmat.primitives", primitives_stub
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", asym_stub
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Server'))

import main

class FakeRedis:
    def __init__(self):
        self.store = {}
    def sadd(self, key, value):
        self.store.setdefault(key, set()).add(value)
    def smembers(self, key):
        return self.store.get(key, set())
    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or {})


def test_register_worker_policy(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main, 'verify_signature_with_key', lambda a,b,c: True)
    monkeypatch.setattr(main, 'assign_waifu', lambda s: 'Agent')
    monkeypatch.setattr(main, 'LOW_BW_ENGINE', 'darkling')

    class Req:
        worker_id = 'id'
        signature = 's'
        pubkey = 'p'
        mode = 'eco'
        provider = 'on-prem'
        hardware = {}

    req = Req()
    resp = asyncio.run(main.register_worker(req))

    assert resp['status'] == 'ok'
    assert fake.store['worker:Agent']['low_bw_engine'] == 'darkling'
