import sys, os
import asyncio
import sys
import os
import json

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


def test_import_hashes(monkeypatch):
    calls = []
    def fake_store_batch(hashes, mask="", wordlist="", rule="", ttl=1800, target="any", hash_mode="0", priority=0):
        calls.append({
            "hashes": hashes,
            "mask": mask,
            "wordlist": wordlist,
            "target": target,
            "hash_mode": hash_mode,
            "rule": rule,
            "priority": priority,
        })
        return f"id{len(calls)}"
    monkeypatch.setattr(redis_manager, 'store_batch', fake_store_batch)
    monkeypatch.setattr(main, 'log_error', lambda *a, **k: None)
    data = (
        b"hash,mask,wordlist,target,hash_mode\n"
        b"h1,?a,wl.txt,t1,1200\n"
        b"h2,,,,1400\n"
    )
    file = FakeUploadFile('hashes.csv', data)
    resp = asyncio.run(main.import_hashes(file, '1000'))
    assert resp == {"queued": 2, "errors": []}
    assert calls == [
        {
            "hashes": ["h1"],
            "mask": "?a",
            "wordlist": "wl.txt",
            "target": "t1",
            "hash_mode": "1200",
            "rule": "",
            "priority": 0,
        },
        {
            "hashes": ["h2"],
            "mask": "",
            "wordlist": "",
            "target": "any",
            "hash_mode": "1400",
            "rule": "",
            "priority": 0,
        },
    ]
