import asyncio
import sys
import os
import types

# Stub FastAPI and Pydantic similar to other server tests
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

# Stub cryptography pieces used by auth_utils
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

# Stub psutil for system metrics
psutil_stub = types.ModuleType("psutil")
psutil_stub.cpu_percent = lambda interval=None: 10.0
psutil_stub.virtual_memory = lambda: types.SimpleNamespace(percent=20.0, used=123456)
psutil_stub.disk_usage = lambda path: types.SimpleNamespace(percent=30.0)
psutil_stub.getloadavg = lambda: (1.0, 0.5, 0.25)
sys.modules.setdefault("psutil", psutil_stub)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Server'))

import main

class DummyRedis:
    def scard(self, key):
        return 0
    def llen(self, key):
        return 0


def test_server_status_reports_system_metrics(monkeypatch):
    fake = DummyRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main.orchestrator_agent, 'compute_backlog_target', lambda: 7)
    monkeypatch.setattr(main.orchestrator_agent, 'pending_count', lambda: 3)
    status = asyncio.run(main.server_status())
    assert 'cpu_usage' in status
    assert 'memory_utilization' in status
    assert 'disk_space' in status
    assert 'cpu_load' in status
    assert 'memory_usage' in status
    assert 'backlog_target' in status
    assert 'pending_jobs' in status
    assert 'queued_batches' in status
