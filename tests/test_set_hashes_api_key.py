import asyncio
import importlib
import sys

import hashmancer.server.main as main

async def _call(monkeypatch):
    monkeypatch.setattr(main, "CONFIG", {})
    monkeypatch.setattr(main, "save_config", lambda: None)
    called = {}
    def fake_reload(mod):
        called["module"] = mod
        return mod
    monkeypatch.setattr(importlib, "reload", fake_reload)
    req = type("Req", (), {"api_key": "abc"})()
    await main.set_hashes_api_key(req)
    return called


def test_set_hashes_api_key_reload(monkeypatch):
    called = asyncio.run(_call(monkeypatch))
    assert called["module"] is sys.modules["hashmancer.server.hashescom_client"]

