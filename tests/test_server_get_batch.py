import asyncio
import sys
import os
import types

# Stub FastAPI and related modules just like in test_server_register
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
primitives_stub.serialization = types.SimpleNamespace(load_pem_public_key=lambda x: None)
crypto_stub.hazmat = types.SimpleNamespace(primitives=primitives_stub)
crypto_stub.exceptions = exceptions_stub
sys.modules.setdefault("cryptography", crypto_stub)
sys.modules.setdefault("cryptography.exceptions", exceptions_stub)
sys.modules.setdefault("cryptography.hazmat.primitives", primitives_stub)
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", asym_stub)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "Server"))

import main


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.stream = [("jobs", [("1-0", {"job_id": "job1"})])]
        self.acked = False
        self.ack_args = None
        self.read_args = None

    def xgroup_create(self, *a, **kw):
        pass

    def xreadgroup(self, group, consumer, streams, count=1, block=0):
        self.read_args = (group, streams)
        return self.stream

    def xack(self, *a, **kw):
        self.acked = True
        self.ack_args = a

    def hgetall(self, key):
        return dict(self.store.get(key, {}))

    def hset(self, key, mapping=None, *args, **kwargs):
        if mapping is not None and not isinstance(mapping, dict):
            field = mapping
            value = args[0] if args else None
            self.store.setdefault(key, {})[field] = value
        else:
            self.store.setdefault(key, {}).update(mapping or kwargs)

    def rpush(self, *a, **kw):
        pass

    def ltrim(self, *a, **kw):
        pass


def test_get_batch_returns_batch_id(monkeypatch):
    fake = FakeRedis()
    fake.store["job:job1"] = {"batch_id": "batch1"}
    fake.store["worker:worker"] = {"low_bw_engine": "hashcat"}
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setattr(main, "verify_signature", lambda a, b, c: True)

    resp = asyncio.run(main.get_batch("worker", "sig"))

    assert resp["batch_id"] == "batch1"
    assert fake.store["worker:worker"]["last_batch"] == "batch1"
    assert resp["msg_id"] == "1-0"
    assert not fake.acked
    # verify correct stream was used
    assert fake.read_args[0] == main.HTTP_GROUP
    assert list(fake.read_args[1].keys())[0] == main.JOB_STREAM

    payload = {
        "worker_id": "worker",
        "batch_id": resp["batch_id"],
        "job_id": resp["job_id"],
        "msg_id": resp["msg_id"],
        "signature": "sig",
    }
    monkeypatch.setattr(main, "verify_signature", lambda a, b, c: True)
    asyncio.run(main.submit_no_founds(payload))
    assert fake.ack_args == (main.JOB_STREAM, main.HTTP_GROUP, "1-0")
