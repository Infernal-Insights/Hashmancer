import asyncio
import sys
import os
import types
from pathlib import Path
import threading

# Stub FastAPI and Pydantic as in other server tests
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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "Server"))

import main
import redis_manager

class FakeRedis:
    def __init__(self):
        self.store = {}
        self.lists = []
    def rpush(self, name, value):
        self.lists.append((name, value))
    def lrem(self, name, count, value):
        pass
    def hgetall(self, key):
        return {}
    def xack(self, *a, **kw):
        pass
    def hset(self, key, field=None, value=None, mapping=None):
        if mapping is not None:
            self.store.setdefault(key, {}).update(mapping)
        else:
            self.store.setdefault(key, {})[field] = value


async def call():
    payload = {
        "worker_id": "w",
        "batch_id": "b",
        "founds": ["h1:p1", "h2:p2"],
        "timestamp": 0,
        "signature": "s",
    }
    return await main.submit_founds(payload)


def test_submit_founds_maps(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setattr(redis_manager, "r", fake)
    monkeypatch.setattr(main, "verify_signature", lambda *a: True)
    tmp = Path("/tmp/founds.txt")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(main, "FOUNDS_FILE", tmp)
    resp = asyncio.run(call())
    assert resp["status"] == "ok"
    assert fake.store["found:map"]["h1"] == "p1"
    assert fake.store["found:map"]["h2"] == "p2"
    assert tmp.read_text().splitlines() == ["h1:p1", "h2:p2"]


def _thread_call(payload):
    asyncio.run(main.submit_founds(payload))


def test_submit_founds_thread_safe(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setattr(redis_manager, "r", fake)
    monkeypatch.setattr(main, "verify_signature", lambda *a: True)
    tmp = Path("/tmp/founds_thread.txt")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(main, "FOUNDS_FILE", tmp)

    threads = []
    lines = []
    for i in range(5):
        line = f"h{i}:p{i}"
        lines.append(line)
        payload = {
            "worker_id": f"w{i}",
            "batch_id": f"b{i}",
            "founds": [line],
            "timestamp": 0,
            "signature": "s",
        }
        t = threading.Thread(target=_thread_call, args=(payload,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    file_lines = tmp.read_text().splitlines()
    assert set(file_lines) == set(lines)
    assert len(file_lines) == len(lines)
