import asyncio
import sys
import os
import types
import pytest

# Minimal FastAPI stubs to avoid heavy dependencies
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Server'))

import main

@pytest.mark.asyncio
async def test_tasks_cancelled_on_shutdown(monkeypatch):
    started = set()
    cancelled = set()

    def make_stub(name):
        async def stub():
            started.add(name)
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.add(name)
                raise
        return stub

    monkeypatch.setattr(main, 'broadcast_presence', make_stub('bcast'))
    monkeypatch.setattr(main, 'poll_hashes_jobs', make_stub('poll'))
    monkeypatch.setattr(main, 'process_hashes_jobs', make_stub('process'))
    monkeypatch.setattr(main, 'dispatch_loop', make_stub('dispatch'))
    monkeypatch.setattr(main, 'print_logo', lambda: None)
    monkeypatch.setattr(main, 'BROADCAST_ENABLED', True)

    main.BACKGROUND_TASKS.clear()

    before = len(main.BACKGROUND_TASKS)

    await main.start_broadcast()
    await asyncio.sleep(0)

    assert len(main.BACKGROUND_TASKS) - before == 4

    await main.shutdown_event()

    assert cancelled == {'bcast', 'poll', 'process', 'dispatch'}
    assert main.BACKGROUND_TASKS == []
