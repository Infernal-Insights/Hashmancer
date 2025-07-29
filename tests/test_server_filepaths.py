import sys, os
import asyncio
import sys
import os
import pytest
from pathlib import Path
import sqlite3

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,

    install_stubs,
    FakeApp,
)

install_stubs()



import hashmancer.server.main as main

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
    db_path = tmp_path / "wl.db"
    monkeypatch.setattr(main.wordlist_db, 'DB_PATH', db_path)
    monkeypatch.setattr(main, 'log_error', lambda *a, **k: None)
    monkeypatch.setattr(main, 'log_info', lambda *a, **k: None)
    file = FakeUploadFile("../evil.txt", b"data")
    asyncio.run(main.upload_wordlist(file))
    conn = sqlite3.connect(db_path)
    row = conn.execute('SELECT name, data FROM wordlists').fetchone()
    conn.close()
    assert row == ("evil.txt", b"data")


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


def test_create_mask_too_long(monkeypatch, tmp_path):
    dest = tmp_path / "m2"
    dest.mkdir()
    monkeypatch.setattr(main, 'MASKS_DIR', dest)
    called = {}

    def fake_log(*a, **k):
        called['logged'] = True

    monkeypatch.setattr(main, 'log_error', fake_log)
    long_mask = '?1' * (main.MAX_MASK_LENGTH + 1)
    with pytest.raises(main.HTTPException):
        asyncio.run(main.create_mask('bad.hcmask', long_mask))
    assert called.get('logged')


def test_upload_wordlist_too_big(monkeypatch, tmp_path):
    db_path = tmp_path / 'wl.db'
    monkeypatch.setattr(main.wordlist_db, 'DB_PATH', db_path)
    monkeypatch.setattr(main, 'UPLOAD_MAX_SIZE', 10)
    called = {}

    def fake_log(*a, **k):
        called['logged'] = True

    monkeypatch.setattr(main, 'log_error', fake_log)
    data = b'a' * 20
    file = FakeUploadFile('big.txt', data)
    with pytest.raises(main.HTTPException):
        asyncio.run(main.upload_wordlist(file))
    assert called.get('logged')
