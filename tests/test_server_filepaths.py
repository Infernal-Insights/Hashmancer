import asyncio
import sys
import os
import types
from pathlib import Path

# Stub modules as in other server tests
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
fastapi_stub.FastAPI = lambda: FakeApp()
fastapi_stub.UploadFile = object
fastapi_stub.File = lambda *a, **kw: None
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Server'))

import main

class FakeUploadFile:
    def __init__(self, name, data):
        self.filename = name
        self._data = data
        self._idx = 0
    async def read(self, n=-1):
        if self._idx >= len(self._data):
            return b""
        if n < 0:
            n = len(self._data) - self._idx
        chunk = self._data[self._idx:self._idx+n]
        self._idx += n
        return chunk


def test_wordlist_upload_sanitizes(tmp_path, monkeypatch):
    dest = tmp_path / "wl"
    dest.mkdir()
    monkeypatch.setattr(main, 'WORDLISTS_DIR', dest)
    monkeypatch.setattr(main, 'log_error', lambda *a, **k: None)
    monkeypatch.setattr(main, 'log_info', lambda *a, **k: None)
    file = FakeUploadFile("../evil.txt", b"data")
    asyncio.run(main.upload_wordlist(file))
    assert (dest / "evil.txt").read_bytes() == b"data"


def test_restore_upload_sanitizes(tmp_path, monkeypatch):
    dest = tmp_path / "rest"
    dest.mkdir()
    monkeypatch.setattr(main, 'RESTORE_DIR', dest)
    monkeypatch.setattr(main, 'log_error', lambda *a, **k: None)
    monkeypatch.setattr(main, 'log_info', lambda *a, **k: None)
    file = FakeUploadFile("../res.restore", b"r")
    asyncio.run(main.upload_restore(file))
    assert (dest / "res.restore").read_bytes() == b"r"


def test_create_mask_sanitizes_and_delete(monkeypatch, tmp_path):
    dest = tmp_path / "m"
    dest.mkdir()
    monkeypatch.setattr(main, 'MASKS_DIR', dest)
    monkeypatch.setattr(main, 'log_error', lambda *a, **k: None)
    monkeypatch.setattr(main, 'log_info', lambda *a, **k: None)
    asyncio.run(main.create_mask("../mask.hcmask", "abc"))
    assert (dest / "mask.hcmask").read_text() == "abc"
    # delete using traversal
    (dest / "mask.hcmask").write_text("abc")
    asyncio.run(main.delete_mask("../mask.hcmask"))
    assert not (dest / "mask.hcmask").exists()
