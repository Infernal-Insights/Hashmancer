import sys
import types

class FakeApp:
    def __init__(self):
        self.called = False
        self.added = []

    async def __call__(self, scope, receive, send):
        self.called = True
        await send({"done": True})

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

class HTTPException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class DummyHTMLResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        await send({"status": self.status_code, "body": self.text})

class BaseModel:
    pass

# FastAPI stub
fastapi_stub = types.ModuleType("fastapi")
fastapi_stub.FastAPI = lambda *a, **kw: FakeApp()
fastapi_stub.UploadFile = object
fastapi_stub.File = lambda *a, **kw: None
fastapi_stub.WebSocket = object
fastapi_stub.WebSocketDisconnect = type("WebSocketDisconnect", (Exception,), {})
fastapi_stub.HTTPException = HTTPException

# CORS stub
cors_stub = types.ModuleType("fastapi.middleware.cors")
cors_stub.CORSMiddleware = object

# Response stub
resp_stub = types.ModuleType("fastapi.responses")
resp_stub.HTMLResponse = DummyHTMLResponse
resp_stub.FileResponse = object


def install_stubs():
    sys.modules.setdefault("fastapi", fastapi_stub)
    sys.modules.setdefault("fastapi.middleware.cors", cors_stub)
    sys.modules.setdefault("fastapi.responses", resp_stub)


