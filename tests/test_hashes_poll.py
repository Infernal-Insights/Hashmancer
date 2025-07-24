import asyncio
import sys
import types

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


sys.path.insert(0, '..')
sys.path.insert(0, 'Server')
import main
import hashescom_client

class FakeRedis:
    def __init__(self):
        self.store = {}
    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or {})
    def scan_iter(self, pattern):
        return []

async def run_once():
    await main.fetch_and_store_jobs()

def test_fetch_and_store_jobs(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main, 'HASHES_ALGORITHMS', ['md5'])
    async def fake_fetch_jobs():
        return [{
            'id': 8,
            'algorithmName': 'MD5',
            'currency': 'BTC',
            'pricePerHash': '1'
        }]

    monkeypatch.setattr(hashescom_client, 'fetch_jobs', fake_fetch_jobs)
    asyncio.run(run_once())
    assert 'hashes_job:8' in fake.store
