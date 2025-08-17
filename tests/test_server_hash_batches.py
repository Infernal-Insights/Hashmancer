import asyncio

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


class FakeRedis:
    def __init__(self):
        self.sets = {}

    def smembers(self, key):
        return self.sets.get(key, set())


async def call():
    return await main.get_hash_batches("h1")


def test_get_hash_batches(monkeypatch):
    fake = FakeRedis()
    fake.sets["hash_batches:h1"] = {"b1", "b2"}
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setattr(redis_manager, "r", fake)

    res = asyncio.run(call())
    assert set(res) == {"b1", "b2"}
