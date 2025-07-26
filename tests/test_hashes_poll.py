import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import asyncio
import sys

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,
    pydantic_stub,
    install_stubs,
    FakeApp,
)

install_stubs()


sys.path.insert(0, '..')
sys.path.insert(0, 'Server')
import main
import hashescom_client

class FakeRedis:
    def __init__(self):
        self.store = {}
    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or {})
    def scan_iter(self, pattern):
        return []

async def run_once():
    await main.fetch_and_store_jobs()

def test_fetch_and_store_jobs(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main, 'HASHES_ALGORITHMS', ['md5'])
    async def fake_fetch_jobs():
        return [{
            'id': 8,
            'algorithmName': 'MD5',
            'currency': 'BTC',
            'pricePerHash': '1'
        }]

    monkeypatch.setattr(hashescom_client, 'fetch_jobs', fake_fetch_jobs)
    asyncio.run(run_once())
    assert 'hashes_job:8' in fake.store
