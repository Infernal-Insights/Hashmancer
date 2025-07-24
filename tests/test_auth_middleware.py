import asyncio
import json
import sys
import os
import types

# Stub FastAPI and related pieces similar to other tests
fastapi_stub = types.ModuleType("fastapi")

class FakeApp:
    def __init__(self):
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        await send({"done": True})

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
fastapi_stub.WebSocketDisconnect = type("WebSocketDisconnect", (Exception,), {})

class HTTPException(Exception):
    pass

fastapi_stub.HTTPException = HTTPException
sys.modules.setdefault("fastapi", fastapi_stub)

cors_stub = types.ModuleType("fastapi.middleware.cors")
cors_stub.CORSMiddleware = object
sys.modules.setdefault("fastapi.middleware.cors", cors_stub)

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
        self.results = ["f1"]

    def hset(self, key, *args, mapping=None, **kwargs):
        data = self.store.setdefault(key, {})
        if mapping:
            data.update(mapping)
        if len(args) == 2:
            field, val = args
            data[field] = val
        data.update(kwargs)

    def hgetall(self, key):
        return dict(self.store.get(key, {}))

    def scan_iter(self, pattern):
        prefix = pattern.split("*")[0]
        for k in self.store.keys():
            if k.startswith(prefix):
                yield k

    def llen(self, key):
        return len(self.results)

    def lrange(self, key, start, end):
        end = len(self.results) - 1 if end == -1 else end
        return self.results[start : end + 1]

    def rpush(self, key, value):
        self.store.setdefault(key, []).append(value)

    def set(self, key, value, ex=None):
        self.store[key] = value
        if ex:
            self.store[f"ttl:{key}"] = ex

    def exists(self, key):
        return key in self.store

    def delete(self, key):
        self.store.pop(key, None)


class FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, text):
        self.sent.append(text)


def test_portal_auth_denies_without_key(monkeypatch):
    events = []

    async def send(evt):
        events.append(evt)

    app = FakeApp()
    mw = main.PortalAuthMiddleware(app, key="secret")
    scope = {"type": "http", "path": "/portal", "headers": [(b"x-api-key", b"bad")]} 

    asyncio.run(mw(scope, lambda: None, send))

    assert not app.called
    assert events and events[0]["status"] == 401


def test_logout_revokes_session(monkeypatch):
    fake_r = FakeRedis()
    monkeypatch.setattr(main, "r", fake_r)
    monkeypatch.setitem(main.CONFIG, "portal_passkey", "pass")
    monkeypatch.setattr(main, "PORTAL_PASSKEY", "pass")

    class Req:
        passkey = "pass"

    resp = asyncio.run(main.login(Req()))
    token = resp["cookie"]

    class LReq:
        pass

    lreq = LReq()
    lreq.token = token
    asyncio.run(main.logout(lreq))

    events = []

    async def send(evt):
        events.append(evt)

    app = FakeApp()
    mw = main.PortalAuthMiddleware(app, key="apikey")
    scope = {
        "type": "http",
        "path": "/portal",
        "headers": [(b"cookie", f"session={token}".encode())],
    }

    asyncio.run(mw(scope, lambda: None, send))

    assert not app.called
    assert events and events[0]["status"] == 401


def test_root_auth_denies_without_key(monkeypatch):
    events = []

    async def send(evt):
        events.append(evt)

    app = FakeApp()
    mw = main.PortalAuthMiddleware(app, key="secret")
    scope = {"type": "http", "path": "/", "headers": []}

    asyncio.run(mw(scope, lambda: None, send))

    assert not app.called
    assert events and events[0]["status"] == 401


def test_portal_auth_allows_with_key(monkeypatch):
    events = []

    async def send(evt):
        events.append(evt)

    app = FakeApp()
    mw = main.PortalAuthMiddleware(app, key="secret")
    scope = {"type": "http", "path": "/portal", "headers": [(b"x-api-key", b"secret")]} 

    asyncio.run(mw(scope, lambda: None, send))

    assert app.called
    assert events and events[0].get("done") is True


def test_root_auth_allows_with_key(monkeypatch):
    events = []

    async def send(evt):
        events.append(evt)

    app = FakeApp()
    mw = main.PortalAuthMiddleware(app, key="secret")
    scope = {"type": "http", "path": "/", "headers": [(b"x-api-key", b"secret")]}

    asyncio.run(mw(scope, lambda: None, send))

    assert app.called
    assert events and events[0].get("done") is True


def test_portal_ws_reports_status(monkeypatch):
    fake_r = FakeRedis()
    monkeypatch.setattr(main, "r", fake_r)
    monkeypatch.setattr(main, "verify_signature", lambda a, b, c: True)

    class Req:
        name = "alpha"
        status = "busy"
        signature = "s"

    asyncio.run(main.set_worker_status(Req()))

    async def fake_server_status():
        return {"worker_count": 1}

    monkeypatch.setattr(main, "server_status", fake_server_status)

    async def fake_sleep(_):
        raise main.WebSocketDisconnect()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    ws = FakeWebSocket()
    asyncio.run(main.portal_ws(ws))

    assert ws.accepted
    assert len(ws.sent) == 1
    data = json.loads(ws.sent[0])
    assert any(w["name"] == "alpha" and w["status"] == "busy" for w in data["workers"])


def test_login_creates_session(monkeypatch):
    fake_r = FakeRedis()
    monkeypatch.setattr(main, "r", fake_r)
    monkeypatch.setitem(main.CONFIG, "portal_passkey", "pass")
    monkeypatch.setattr(main, "PORTAL_PASSKEY", "pass")

    class Req:
        passkey = "pass"

    resp = asyncio.run(main.login(Req()))
    assert resp["status"] == "ok"
    cookie = resp["cookie"]
    sid = cookie.split("|")[0]
    assert f"session:{sid}" in fake_r.store


def test_portal_auth_allows_cookie(monkeypatch):
    fake_r = FakeRedis()
    monkeypatch.setattr(main, "r", fake_r)
    monkeypatch.setitem(main.CONFIG, "portal_passkey", "pass")
    monkeypatch.setattr(main, "PORTAL_PASSKEY", "pass")

    class Req:
        passkey = "pass"

    resp = asyncio.run(main.login(Req()))
    cookie = resp["cookie"]

    events = []

    async def send(evt):
        events.append(evt)

    app = FakeApp()
    mw = main.PortalAuthMiddleware(app, key="apikey")
    scope = {
        "type": "http",
        "path": "/portal",
        "headers": [(b"cookie", f"session={cookie}".encode())],
    }

    asyncio.run(mw(scope, lambda: None, send))

    assert app.called
    assert events and events[0].get("done") is True


def test_session_expiry(monkeypatch):
    fake_r = FakeRedis()
    monkeypatch.setattr(main, "r", fake_r)
    monkeypatch.setitem(main.CONFIG, "portal_passkey", "pass")
    monkeypatch.setattr(main, "PORTAL_PASSKEY", "pass")

    class Req:
        passkey = "pass"

    monkeypatch.setattr(main.time, "time", lambda: 0)
    resp = asyncio.run(main.login(Req()))
    cookie = resp["cookie"]

    monkeypatch.setattr(main.time, "time", lambda: main.SESSION_TTL + 1)
    events = []

    async def send(evt):
        events.append(evt)

    app = FakeApp()
    mw = main.PortalAuthMiddleware(app, key="apikey")
    scope = {
        "type": "http",
        "path": "/portal",
        "headers": [(b"cookie", f"session={cookie}".encode())],
    }

    asyncio.run(mw(scope, lambda: None, send))

    assert not app.called
    assert events and events[0]["status"] == 401
