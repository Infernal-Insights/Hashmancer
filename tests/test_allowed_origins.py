import os
import sys
import json
import types
import importlib

psutil_stub = types.ModuleType("psutil")
psutil_stub.cpu_percent = lambda interval=None: 0.0
psutil_stub.virtual_memory = lambda: types.SimpleNamespace(percent=0.0, used=0)
psutil_stub.disk_usage = lambda path: types.SimpleNamespace(percent=0.0)
psutil_stub.getloadavg = lambda: (0.0, 0.0, 0.0)
sys.modules.setdefault("psutil", psutil_stub)

fastapi_stub = types.ModuleType("fastapi")

class FakeApp:
    def __init__(self):
        self.added = []

    def add_middleware(self, *a, **kw):
        self.added.append((a, kw))

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
fastapi_stub.WebSocketDisconnect = type("WebSocketDisconnect", (Exception,), {})

class HTTPException(Exception):
    pass
fastapi_stub.HTTPException = HTTPException

pydantic_stub = types.ModuleType("pydantic")
class BaseModel:
    pass
pydantic_stub.BaseModel = BaseModel

cors_stub = types.ModuleType("fastapi.middleware.cors")
cors_stub.CORSMiddleware = type("CORSMiddleware", (), {})

resp_stub = types.ModuleType("fastapi.responses")
class DummyHTMLResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        await send({"status": self.status_code, "body": self.text})

resp_stub.HTMLResponse = DummyHTMLResponse
resp_stub.FileResponse = object


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Server'))


def test_allowed_origins_applied(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'fastapi', fastapi_stub)
    monkeypatch.setitem(sys.modules, 'fastapi.middleware.cors', cors_stub)
    monkeypatch.setitem(sys.modules, 'fastapi.responses', resp_stub)
    monkeypatch.setitem(sys.modules, 'pydantic', pydantic_stub)

    cfg_dir = tmp_path / '.hashmancer'
    cfg_dir.mkdir()
    cfg = cfg_dir / 'server_config.json'
    cfg.write_text(json.dumps({"allowed_origins": ["https://foo"]}))
    monkeypatch.setenv('HOME', str(tmp_path))

    import main
    importlib.reload(main)

    app = main.app
    found = False
    for args, kw in app.added:
        if args and args[0] is cors_stub.CORSMiddleware:
            found = kw.get('allow_origins') == ["https://foo"]
    assert found
