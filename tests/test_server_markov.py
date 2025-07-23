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

class DummyRedis:
    def __init__(self):
        self.store = {}
    def scard(self, key):
        return 0
    def llen(self, key):
        return 0


def test_train_markov_invokes_processor(monkeypatch, tmp_path):
    called = {}
    def fake_process(dir_path, lang='english'):
        called['dir'] = dir_path
        called['lang'] = lang
    monkeypatch.setattr(main, 'WORDLISTS_DIR', tmp_path)
    monkeypatch.setattr(main, 'learn_trends', types.SimpleNamespace(process_wordlists=fake_process))
    req = type('Req', (), {'lang': 'french', 'directory': None})
    resp = asyncio.run(main.train_markov(req()))
    assert resp['status'] == 'ok'
    assert called['dir'] == tmp_path
    assert called['lang'] == 'french'


def test_update_probabilistic_order(monkeypatch):
    monkeypatch.setattr(main, 'CONFIG', {})
    saved = {}
    monkeypatch.setattr(main, 'save_config', lambda: saved.setdefault('done', True))
    req = type('Req', (), {'enabled': True})
    resp = asyncio.run(main.set_probabilistic_order(req()))
    assert resp['status'] == 'ok'
    assert main.CONFIG['probabilistic_order'] is True
    assert main.PROBABILISTIC_ORDER is True
    assert saved.get('done')


def test_update_markov_lang(monkeypatch):
    monkeypatch.setattr(main, 'CONFIG', {})
    saved = {}
    monkeypatch.setattr(main, 'save_config', lambda: saved.setdefault('done', True))
    req = type('Req', (), {'lang': 'german'})
    resp = asyncio.run(main.set_markov_lang(req()))
    assert resp['status'] == 'ok'
    assert main.CONFIG['markov_lang'] == 'german'
    assert main.MARKOV_LANG == 'german'
    assert saved.get('done')


def test_server_status_includes_settings(monkeypatch):
    fake = DummyRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main, 'PROBABILISTIC_ORDER', True)
    monkeypatch.setattr(main, 'MARKOV_LANG', 'spanish')
    monkeypatch.setattr(main, 'LLM_TRAIN_EPOCHS', 2)
    monkeypatch.setattr(main, 'LLM_TRAIN_LEARNING_RATE', 0.002)
    monkeypatch.setattr(main.orchestrator_agent, 'compute_backlog_target', lambda: 5)
    monkeypatch.setattr(main.orchestrator_agent, 'pending_count', lambda: 2)
    status = asyncio.run(main.server_status())
    assert status['probabilistic_order'] is True
    assert status['markov_lang'] == 'spanish'
    assert status['llm_train_epochs'] == 2
    assert status['llm_train_learning_rate'] == 0.002
