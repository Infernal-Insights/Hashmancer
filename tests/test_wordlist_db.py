import sys, os
import asyncio
import sys
import os
import sqlite3
import json
import base64
import gzip

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,

    install_stubs,
    FakeApp,
)

install_stubs()


import hashmancer.server.main as main
import orchestrator_agent
import wordlist_db


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
        chunk = self._data[self._idx : self._idx + n]
        self._idx += n
        return chunk


class FakeRedis:
    def __init__(self):
        self.store = {}

    def exists(self, key):
        return key in self.store

    def set(self, key, value):
        self.store[key] = value

    def append(self, key, value):
        self.store[key] = self.store.get(key, "") + value

    def pipeline(self):
        parent = self

        class Pipe:
            def __init__(self):
                self.cmds = []

            def set(self, k, v):
                self.cmds.append(("set", k, v))

            def append(self, k, v):
                self.cmds.append(("append", k, v))

            def execute(self):
                for cmd, k, v in self.cmds:
                    if cmd == "set":
                        parent.set(k, v)
                    else:
                        parent.append(k, v)
                self.cmds = []

        return Pipe()


def test_wordlist_db_persist_and_cache(monkeypatch, tmp_path):
    db_path = tmp_path / "wl.db"
    monkeypatch.setattr(wordlist_db, "DB_PATH", db_path)
    monkeypatch.setattr(main.wordlist_db, "DB_PATH", db_path)
    monkeypatch.setattr(orchestrator_agent.wordlist_db, "DB_PATH", db_path)
    monkeypatch.setattr(orchestrator_agent, "log_error", lambda *a, **k: None)
    monkeypatch.setattr(main, "log_error", lambda *a, **k: None)
    monkeypatch.setattr(main, "log_info", lambda *a, **k: None)

    file = FakeUploadFile("foo.txt", b"abc\n123")
    asyncio.run(main.upload_wordlist(file))

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT name, data FROM wordlists").fetchone()
    conn.close()
    assert row == ("foo.txt", b"abc\n123")

    fake_r = FakeRedis()
    monkeypatch.setattr(orchestrator_agent, "r", fake_r)
    key = orchestrator_agent.cache_wordlist("foo.txt")
    stored = fake_r.store["wlcache:" + key]
    assert gzip.decompress(base64.b64decode(stored)) == b"abc\n123"
