import asyncio
import sys
import os
import types

# Stub modules similar to other server tests
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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "Server"))

import main


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.lists = {}

    def sadd(self, key, value):
        self.store.setdefault(key, set()).add(value)

    def smembers(self, key):
        return self.store.get(key, set())

    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or {})

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def lpop(self, key):
        vals = self.lists.get(key, [])
        return vals.pop(0) if vals else None

    def hincrby(self, key, field, amount):
        val = int(self.store.get(key, {}).get(field, 0)) + amount
        self.store.setdefault(key, {})[field] = val
        return val

    def hgetall(self, key):
        return dict(self.store.get(key, {}))


def test_flash_queue_on_register(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setattr(main, "verify_signature_with_key", lambda *a: True)
    monkeypatch.setattr(main, "assign_waifu", lambda s: "Agent")
    monkeypatch.setattr(main, "LOW_BW_ENGINE", "hashcat")

    class Req:
        worker_id = "id"
        signature = "s"
        pubkey = "p"
        timestamp = 0
        mode = "eco"
        provider = "on-prem"
        hardware = {"gpus": [{"uuid": "u1", "model": "RTX 3080", "index": 0}]}

    resp = asyncio.run(main.register_worker(Req()))

    assert resp["status"] == "ok"
    assert fake.lists["flash:Agent"]
