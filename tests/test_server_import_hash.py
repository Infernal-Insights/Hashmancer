import asyncio
import pytest

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,

    install_stubs,
    FakeApp,
)

install_stubs()

import hashmancer.server.main as main
from utils import redis_manager


def test_import_hash(monkeypatch):
    called = {}

    def fake_store(hashes, mask="", wordlist="", rule="", ttl=1800, target="any", hash_mode="0", priority=0):
        called["hashes"] = hashes
        called["mode"] = hash_mode
        return "id1"

    monkeypatch.setattr(redis_manager, "store_batch", fake_store)
    monkeypatch.setattr(main, "log_error", lambda *a, **k: None)

    resp = asyncio.run(main.import_hash("abc", "1400"))
    assert resp == {"batch_id": "id1"}
    assert called["hashes"] == ["abc"]
    assert called["mode"] == "1400"


def test_import_hash_failure(monkeypatch):
    monkeypatch.setattr(redis_manager, "store_batch", lambda *a, **kw: None)
    with pytest.raises(main.HTTPException):
        asyncio.run(main.import_hash("a", "0"))

