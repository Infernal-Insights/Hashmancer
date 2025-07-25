import importlib
import types
import sys

sys.path.insert(0, 'Server')

import auth_utils


def _setup(monkeypatch, current_time):
    monkeypatch.setattr(auth_utils.time, 'time', lambda: current_time)
    class Key:
        def verify(self, *a, **k):
            pass
    monkeypatch.setattr(auth_utils, 'get_worker_pubkey', lambda wid: Key())
    monkeypatch.setattr(auth_utils.base64, 'b64decode', lambda s: b'')
    monkeypatch.setattr(auth_utils.padding, 'PKCS1v15', lambda: None)
    monkeypatch.setattr(auth_utils.hashes, 'SHA256', lambda: None)


def test_valid_timestamp(monkeypatch):
    _setup(monkeypatch, 100)
    assert auth_utils.verify_signature('w', 'm', 100, 's')


def test_expired_timestamp(monkeypatch):
    _setup(monkeypatch, 100)
    assert not auth_utils.verify_signature('w', 'm', 50, 's')
