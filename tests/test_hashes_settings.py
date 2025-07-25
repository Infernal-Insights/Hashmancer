import asyncio
import sys
import os
import types
import pytest

# Minimal FastAPI and Pydantic stubs
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
resp_stub.FileResponse = object
sys.modules.setdefault('fastapi.responses', resp_stub)

pydantic_stub = types.ModuleType('pydantic')
class BaseModel:
    pass
pydantic_stub.BaseModel = BaseModel
sys.modules.setdefault('pydantic', pydantic_stub)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Server'))

import main


@pytest.mark.asyncio
async def test_set_hashes_settings(monkeypatch):
    monkeypatch.setattr(main, 'CONFIG', {})
    monkeypatch.setattr(main, 'HASHES_SETTINGS', {})
    monkeypatch.setattr(main, 'HASHES_POLL_INTERVAL', 1800)
    monkeypatch.setattr(main, 'HASHES_ALGO_PARAMS', {})
    saved = {}
    monkeypatch.setattr(main, 'save_config', lambda: saved.setdefault('done', True))
    req = type('Req', (), {
        'hashes_poll_interval': 60,
        'algo_params': {'md5': {'mask_length': 8}}
    })
    resp = await main.set_hashes_settings(req())
    assert resp['status'] == 'ok'
    assert main.CONFIG['hashes_settings']['hashes_poll_interval'] == 60
    assert main.HASHES_SETTINGS['hashes_poll_interval'] == 60
    assert main.HASHES_ALGO_PARAMS['md5'] == {'mask_length': 8}
    assert saved.get('done')
    settings = await main.get_hashes_settings()
    assert settings['hashes_poll_interval'] == 60


@pytest.mark.asyncio
async def test_poll_hashes_jobs_uses_setting(monkeypatch):
    monkeypatch.setattr(main, 'HASHES_SETTINGS', {'hashes_poll_interval': 5})
    called = {}
    async def fake_fetch():
        called['fetch'] = True
    monkeypatch.setattr(main, 'fetch_and_store_jobs', fake_fetch)

    async def fake_sleep(t):
        called['sleep'] = t
        raise StopAsyncIteration
    monkeypatch.setattr(asyncio, 'sleep', fake_sleep)
    with pytest.raises(StopAsyncIteration):
        await main.poll_hashes_jobs()
    assert called.get('sleep') == 5

