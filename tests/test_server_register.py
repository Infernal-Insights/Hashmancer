import sys, os
import asyncio
import sys
import os
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

class FakeRedis:
    def __init__(self):
        self.store = {}
    def sadd(self, key, value):
        self.store.setdefault(key, set()).add(value)
    def smembers(self, key):
        return self.store.get(key, set())
    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or {})


def test_register_worker_policy(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    import base64

    def fake_load_pem_public_key(data):
        class Key:
            def verify(self, *a, **kw):
                pass

        return Key()

    monkeypatch.setattr(serialization, 'load_pem_public_key', fake_load_pem_public_key)
    monkeypatch.setattr(padding, 'PKCS1v15', lambda: None)
    monkeypatch.setattr(hashes, 'SHA256', lambda: None)
    monkeypatch.setattr(base64, 'b64decode', lambda s: b'')
    monkeypatch.setattr(main, 'verify_signature_with_key', lambda *a: True)
    monkeypatch.setattr(main, 'assign_waifu', lambda s: 'Agent')
    monkeypatch.setattr(main, 'LOW_BW_ENGINE', 'darkling')

    class Req:
        worker_id = 'id'
        signature = 's'
        pubkey = 'p'
        timestamp = 0
        pin = None
        mode = 'eco'
        provider = 'on-prem'
        hardware = {}

    req = Req()
    resp = asyncio.run(main.register_worker(req))

    assert resp['status'] == 'ok'
    assert fake.store['worker:Agent']['low_bw_engine'] == 'darkling'


def _build_req():
    return type('Req', (), {
        'worker_id': 'id',
        'signature': 's',
        'pubkey': 'p',
        'timestamp': 0,
        'pin': None,
        'mode': 'eco',
        'provider': 'on-prem',
        'hardware': {}
    })()


def test_trusted_key_allows_registration(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main, 'verify_signature_with_key', lambda *a: True)
    monkeypatch.setattr(main, 'assign_waifu', lambda s: 'Ok')
    monkeypatch.setattr(main, 'TRUSTED_KEY_FINGERPRINTS', {'abc'})
    monkeypatch.setattr(main, 'fingerprint_public_key', lambda k: 'abc')

    resp = asyncio.run(main.register_worker(_build_req()))
    assert resp['status'] == 'ok'


def test_untrusted_key_rejected(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main, 'verify_signature_with_key', lambda *a: True)
    monkeypatch.setattr(main, 'assign_waifu', lambda s: 'Nope')
    monkeypatch.setattr(main, 'log_error', lambda *a, **k: None)
    monkeypatch.setattr(main, 'TRUSTED_KEY_FINGERPRINTS', {'abc'})
    monkeypatch.setattr(main, 'fingerprint_public_key', lambda k: 'xyz')

    with pytest.raises(main.HTTPException):
        asyncio.run(main.register_worker(_build_req()))


def test_worker_pin_required(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main, 'verify_signature_with_key', lambda *a: True)
    monkeypatch.setattr(main, 'assign_waifu', lambda s: 'P')
    monkeypatch.setattr(main, 'TRUSTED_KEY_FINGERPRINTS', set())
    monkeypatch.setitem(main.CONFIG, 'worker_pin', '123')

    req = _build_req()
    req.pin = 'wrong'
    with pytest.raises(main.HTTPException):
        asyncio.run(main.register_worker(req))


def test_worker_pin_valid(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main, 'verify_signature_with_key', lambda *a: True)
    monkeypatch.setattr(main, 'assign_waifu', lambda s: 'Okie')
    monkeypatch.setattr(main, 'TRUSTED_KEY_FINGERPRINTS', set())
    monkeypatch.setitem(main.CONFIG, 'worker_pin', '123')

    req = _build_req()
    req.pin = '123'
    resp = asyncio.run(main.register_worker(req))
    assert resp['status'] == 'ok'
