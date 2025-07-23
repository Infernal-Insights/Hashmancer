import asyncio
import sys
import os
import types

# Stub FastAPI and Pydantic like other server tests
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


def test_train_llm_invokes_helper(monkeypatch, tmp_path):
    called = {}
    def fake_train(dataset, model, epochs, lr, out_dir):
        called['dataset'] = dataset
        called['model'] = model
        called['epochs'] = epochs
        called['lr'] = lr
        called['out'] = out_dir
    monkeypatch.setattr(main, '_train_llm', types.SimpleNamespace(train_model=fake_train))
    req = type('Req', (), {
        'dataset': str(tmp_path / 'data.txt'),
        'base_model': 'modelA',
        'epochs': 3,
        'learning_rate': 0.001,
        'output_dir': str(tmp_path / 'out')
    })
    resp = asyncio.run(main.train_llm(req()))
    assert resp['status'] == 'ok'
    assert called['dataset'] == tmp_path / 'data.txt'
    assert called['model'] == 'modelA'
    assert called['epochs'] == 3
    assert called['lr'] == 0.001
    assert called['out'] == tmp_path / 'out'
