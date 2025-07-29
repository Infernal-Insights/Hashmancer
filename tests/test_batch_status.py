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
        self.store = {}
        self.stream = [("jobs", [("1-0", {"job_id": "job1"})])]
        self.persisted = []
        self.ack_args = None
    def xgroup_create(self, *a, **kw):
        pass
    def xreadgroup(self, group, consumer, streams, count=1, block=0):
        return self.stream
    def xack(self, *args):
        self.ack_args = args
    def hgetall(self, key):
        return dict(self.store.get(key, {}))
    def hset(self, key, mapping=None, *args, **kwargs):
        if mapping is not None and not isinstance(mapping, dict):
            field = mapping
            value = args[0] if args else None
            self.store.setdefault(key, {})[field] = value
        else:
            self.store.setdefault(key, {}).update(mapping or kwargs)
    def rpush(self, name, value):
        self.store.setdefault(name, []).append(value)
    def lrem(self, name, count, value):
        lst = self.store.get(name, [])
        while value in lst:
            lst.remove(value)
    def persist(self, key):
        self.persisted.append(key)


def setup(monkeypatch):
    fake = FakeRedis()
    fake.store["job:job1"] = {"batch_id": "batch1"}
    fake.store["batch:batch1"] = {"status": "queued"}
    fake.store["worker:w"] = {}
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setattr(redis_manager, "r", fake)
    monkeypatch.setattr(main, "verify_signature", lambda *a: True)
    return fake


def test_status_updates_no_founds(monkeypatch):
    fake = setup(monkeypatch)
    resp = asyncio.run(main.get_batch("w", 0, "sig"))
    assert fake.store["batch:batch1"]["status"] == "processing"
    assert "batch:batch1" in fake.persisted

    payload = type(
        "Req",
        (),
        {
            "worker_id": "w",
            "batch_id": resp["batch_id"],
            "job_id": resp["job_id"],
            "msg_id": resp["msg_id"],
            "timestamp": 0,
            "signature": "sig",
        },
    )()
    asyncio.run(main.submit_no_founds(payload))
    assert fake.store["batch:batch1"]["status"] == "done"


def test_status_updates_founds(monkeypatch, tmp_path):
    fake = setup(monkeypatch)
    monkeypatch.setattr(main, "FOUNDS_FILE", tmp_path / "f.txt")
    resp = asyncio.run(main.get_batch("w", 0, "sig"))
    payload = type(
        "Req",
        (),
        {
            "worker_id": "w",
            "batch_id": resp["batch_id"],
            "job_id": resp["job_id"],
            "msg_id": resp["msg_id"],
            "timestamp": 0,
            "signature": "sig",
            "founds": ["h:p"],
        },
    )()
    asyncio.run(main.submit_founds(payload))
    assert fake.store["batch:batch1"]["status"] == "done"
