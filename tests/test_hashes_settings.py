import sys, os
import asyncio
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


import Server.main as main
sys.modules['main'] = main
from Server.app.background import hashes_jobs


@pytest.mark.asyncio
async def test_set_hashes_settings(monkeypatch):
    monkeypatch.setattr(main, 'CONFIG', {})
    monkeypatch.setattr(main, 'HASHES_SETTINGS', {})
    monkeypatch.setattr(main, 'HASHES_POLL_INTERVAL', 1800)
    monkeypatch.setattr(main, 'HASHES_ALGO_PARAMS', {})
    saved = {}
    monkeypatch.setattr(main, 'save_config', lambda: saved.setdefault('done', True))
    req = type('Req', (), {
        'hashes_poll_interval': 60,
        'algo_params': {'md5': {'mask_length': 8}}
    })
    resp = await main.set_hashes_settings(req())
    assert resp['status'] == 'ok'
    assert main.CONFIG['hashes_settings']['hashes_poll_interval'] == 60
    assert main.HASHES_SETTINGS['hashes_poll_interval'] == 60
    assert main.HASHES_ALGO_PARAMS['md5'] == {'mask_length': 8}
    assert saved.get('done')
    settings = await main.get_hashes_settings()
    assert settings['hashes_poll_interval'] == 60


@pytest.mark.asyncio
async def test_poll_hashes_jobs_uses_setting(monkeypatch):
    monkeypatch.setattr(main, 'HASHES_SETTINGS', {'hashes_poll_interval': 5})
    called = {}
    async def fake_fetch():
        called['fetch'] = True
    monkeypatch.setattr(hashes_jobs, 'fetch_and_store_jobs', fake_fetch)

    async def fake_sleep(t):
        called['sleep'] = t
        raise StopAsyncIteration
    monkeypatch.setattr(asyncio, 'sleep', fake_sleep)
    with pytest.raises(StopAsyncIteration):
        await hashes_jobs.poll_hashes_jobs()
    assert called.get('sleep') == 5

