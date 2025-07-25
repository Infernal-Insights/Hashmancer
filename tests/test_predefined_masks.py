import asyncio
import sys
import os
import types

# Stub FastAPI and Pydantic like other server tests
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


def test_update_predefined_masks(monkeypatch):
    monkeypatch.setattr(main, 'CONFIG', {})
    saved = {}
    monkeypatch.setattr(main, 'save_config', lambda: saved.setdefault('done', True))
    req = type('Req', (), {'masks': ['?d?d', '?l?l']})
    resp = asyncio.run(main.set_predefined_masks(req()))
    assert resp['status'] == 'ok'
    assert main.CONFIG['predefined_masks'] == ['?d?d', '?l?l']
    assert main.PREDEFINED_MASKS == ['?d?d', '?l?l']
    assert saved.get('done')
    masks = asyncio.run(main.get_predefined_masks())
    assert masks == ['?d?d', '?l?l']
    resp = asyncio.run(main.clear_predefined_masks())
    assert resp['status'] == 'ok'
    assert main.CONFIG['predefined_masks'] == []
    assert main.PREDEFINED_MASKS == []
