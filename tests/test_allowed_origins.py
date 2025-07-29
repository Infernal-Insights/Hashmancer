import sys, os
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

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,

    install_stubs,
    FakeApp,
)

install_stubs()




def test_allowed_origins_applied(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'fastapi', fastapi_stub)
    monkeypatch.setitem(sys.modules, 'fastapi.middleware.cors', cors_stub)
    monkeypatch.setitem(sys.modules, 'fastapi.responses', resp_stub)

    cfg_dir = tmp_path / '.hashmancer'
    cfg_dir.mkdir()
    cfg = cfg_dir / 'server_config.json'
    cfg.write_text(json.dumps({"allowed_origins": ["https://foo"]}))
    monkeypatch.setenv('HOME', str(tmp_path))

    from hashmancer.server.app import config, app
    import importlib
    importlib.reload(config)
    importlib.reload(app)

    app_instance = app.app
    found = False
    for args, kw in app_instance.added:
        if args and args[0] is cors_stub.CORSMiddleware:
            found = kw.get('allow_origins') == ["https://foo"]
    assert found
