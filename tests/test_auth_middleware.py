import sys, os
import asyncio
import json
import sys
import os
import pytest

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,
    pydantic_stub,
    install_stubs,
    FakeApp,
)

install_stubs()




from hashmancer.server.app import app as app_mod
import hashmancer.server.main as main


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

    def ttl(self, key):
        return self.store.get(f"ttl:{key}", -1)

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
    mw = app_mod.PortalAuthMiddleware(app, key="secret")
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
    mw = app_mod.PortalAuthMiddleware(app, key="apikey")
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
    mw = app_mod.PortalAuthMiddleware(app, key="secret")
    scope = {"type": "http", "path": "/", "headers": []}

    asyncio.run(mw(scope, lambda: None, send))

    assert not app.called
    assert events and events[0]["status"] == 401


def test_portal_auth_allows_with_key(monkeypatch):
    events = []

    async def send(evt):
        events.append(evt)

    app = FakeApp()
    mw = app_mod.PortalAuthMiddleware(app, key="secret")
    scope = {"type": "http", "path": "/portal", "headers": [(b"x-api-key", b"secret")]} 

    asyncio.run(mw(scope, lambda: None, send))

    assert app.called
    assert events and events[0].get("done") is True


def test_root_auth_allows_with_key(monkeypatch):
    events = []

    async def send(evt):
        events.append(evt)

    app = FakeApp()
    mw = app_mod.PortalAuthMiddleware(app, key="secret")
    scope = {"type": "http", "path": "/", "headers": [(b"x-api-key", b"secret")]}

    asyncio.run(mw(scope, lambda: None, send))

    assert app.called
    assert events and events[0].get("done") is True


def test_portal_ws_reports_status(monkeypatch):
    fake_r = FakeRedis()
    monkeypatch.setattr(main, "r", fake_r)
    monkeypatch.setattr(main, "verify_signature", lambda *a: True)

    class Req:
        name = "alpha"
        status = "busy"
        timestamp = 0
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


def test_initial_login_requires_credentials(monkeypatch):
    fake_r = FakeRedis()
    monkeypatch.setattr(main, "r", fake_r)
    monkeypatch.setitem(main.CONFIG, "initial_admin_token", "token")
    monkeypatch.setattr(main, "PORTAL_PASSKEY", "pass")
    logged = {}
    monkeypatch.setattr(main, "log_error", lambda *a, **k: logged.setdefault("err", True))

    class Req:
        passkey = "token"

    with pytest.raises(main.HTTPException):
        asyncio.run(main.login(Req()))
    assert logged.get("err")


def test_initial_login_sets_credentials(monkeypatch):
    fake_r = FakeRedis()
    monkeypatch.setattr(main, "r", fake_r)
    monkeypatch.setitem(main.CONFIG, "initial_admin_token", "token")
    monkeypatch.setattr(main, "PORTAL_PASSKEY", "pass")

    class Req:
        passkey = "token"
        username = "admin"
        password = "secret"

    resp = asyncio.run(main.login(Req()))
    assert resp["status"] == "ok"
    assert main.CONFIG.get("admin_username") == "admin"
    assert "initial_admin_token" not in main.CONFIG
    hashval = main.CONFIG.get("admin_password_hash")
    assert hashval and main.password_hasher.verify(hashval, "secret")


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
    mw = app_mod.PortalAuthMiddleware(app, key="apikey")
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
    mw = app_mod.PortalAuthMiddleware(app, key="apikey")
    scope = {
        "type": "http",
        "path": "/portal",
        "headers": [(b"cookie", f"session={cookie}".encode())],
    }

    asyncio.run(mw(scope, lambda: None, send))

    assert not app.called
    assert events and events[0]["status"] == 401
