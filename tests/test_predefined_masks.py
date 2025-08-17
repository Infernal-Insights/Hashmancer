import sys, os
import asyncio
import sys
import os

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,

    install_stubs,
    FakeApp,
)

install_stubs()


import hashmancer.server.main as main


def test_update_predefined_masks(monkeypatch):
    monkeypatch.setattr(main, 'CONFIG', {})
    saved = {}
    monkeypatch.setattr(main, 'save_config', lambda: saved.setdefault('done', True))
    req = type('Req', (), {'masks': ['?d?d', '?l?l']})
    resp = asyncio.run(main.set_predefined_masks(req()))
    assert resp['status'] == 'ok'
    assert main.CONFIG['predefined_masks'] == ['?d?d', '?l?l']
    assert main.PREDEFINED_MASKS == ['?d?d', '?l?l']
    assert saved.get('done')
    masks = asyncio.run(main.get_predefined_masks())
    assert masks == ['?d?d', '?l?l']
    resp = asyncio.run(main.clear_predefined_masks())
    assert resp['status'] == 'ok'
    assert main.CONFIG['predefined_masks'] == []
    assert main.PREDEFINED_MASKS == []
