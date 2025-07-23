import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Server"))
import json
import redis_manager

import orchestrator_agent

orchestrator_agent.redis.exceptions.ResponseError = Exception


class FakeRedis:
    def __init__(self, info=None, raise_err=False):
        self.info = info
        self.raise_err = raise_err
        self.group_created = False
        self.lists = {}

    def xpending(self, stream, group):
        if self.raise_err:
            raise orchestrator_agent.redis.exceptions.ResponseError("no group")
        return self.info

    def xgroup_create(self, stream, group, id="0", mkstream=True):
        self.group_created = True

    def rpush(self, name, value):
        self.lists.setdefault(name, []).append(value)

    def lrem(self, name, count, value):
        lst = self.lists.get(name, [])
        while value in lst:
            lst.remove(value)


def test_compute_backlog_target(monkeypatch):
    monkeypatch.setattr(orchestrator_agent, "gpu_metrics", lambda: [(16, 10.0), (4, 0.0)])
    assert orchestrator_agent.compute_backlog_target() == 9


def test_pending_count_dict(monkeypatch):
    fake = FakeRedis(info={"pending": 5})
    monkeypatch.setattr(orchestrator_agent, "r", fake)
    assert orchestrator_agent.pending_count() == 5


def test_pending_count_create_group(monkeypatch):
    fake = FakeRedis(info=None, raise_err=True)
    monkeypatch.setattr(orchestrator_agent, "r", fake)
    assert orchestrator_agent.pending_count() == 0
    assert fake.group_created


def test_token_id_mapping():
    assert orchestrator_agent.TOKEN_TO_ID["$c"] == "?5"
    assert orchestrator_agent.TOKEN_TO_ID["$e"] == "?6"
    assert orchestrator_agent.ID_TO_CHARSET["?5"] == orchestrator_agent.charsets.COMMON_SYMBOLS
    assert orchestrator_agent.ID_TO_CHARSET["?6"] == orchestrator_agent.charsets.EMOJI


def test_darkling_transformed_mask(monkeypatch, tmp_path):
    wl = tmp_path / "wl.txt"
    wl.write_text("Abc\n1234\n")

    class DR(FakeRedis):
        def __init__(self):
            super().__init__()
            self.jobs = {}
            self.store = {"batch:1": {"hashes": json.dumps(["h"]), "wordlist": str(wl)}}
            self.queue = ["1"]
            self.queued = []

        def rpop(self, key):
            return self.queue.pop(0) if self.queue else None

        def hgetall(self, key):
            return self.store.get(key, {})

        def hset(self, key, mapping=None, **kw):
            self.jobs[key] = dict(mapping or {})

        def expire(self, key, ttl):
            pass

        def xadd(self, stream, mapping):
            self.queued.append((stream, mapping))

        def scan_iter(self, pattern):
            return []

    fake = DR()

    monkeypatch.setattr(orchestrator_agent, "r", fake)
    monkeypatch.setattr(redis_manager, "r", fake)
    monkeypatch.setattr(orchestrator_agent, "compute_backlog_target", lambda: 1)
    monkeypatch.setattr(orchestrator_agent, "pending_count", lambda *a, **k: 0)
    monkeypatch.setattr(orchestrator_agent, "any_darkling_workers", lambda: True)
    monkeypatch.setattr(orchestrator_agent, "cache_wordlist", lambda p: "key")
    monkeypatch.setattr(orchestrator_agent, "generate_mask", lambda length: "$U$l$d$s$c$e")
    monkeypatch.setattr(orchestrator_agent, "compute_batch_range", lambda r, k: (0, 500))

    orchestrator_agent.dispatch_batches()

    stream, mapping = fake.queued[-1]
    job = fake.jobs[f"job:{mapping['job_id']}"]

    assert stream == orchestrator_agent.LOW_BW_JOB_STREAM
    assert job["attack_mode"] == "mask"
    assert job["mask"] == "?1?2?3?4?5?6"
    cs = json.loads(job["mask_charsets"])
    assert all(k in cs for k in ["?1", "?2", "?3", "?4", "?5", "?6"])
    assert job["start"] == 0
    assert job["end"] == 500


def test_compute_batch_range_scales():
    small = orchestrator_agent.compute_batch_range(5.0, 10000)[1]
    large = orchestrator_agent.compute_batch_range(20.0, 10000)[1]
    assert large > small
    capped = orchestrator_agent.compute_batch_range(100.0, 500)[1]
    assert capped == 500


def test_dispatch_skips_cracked(monkeypatch):
    class FR(FakeRedis):
        def __init__(self):
            super().__init__()
            self.jobs = {}
            self.queue = ["1"]
            self.store = {"batch:1": {"hashes": json.dumps(["h1", "h2"]), "mask": "?d"}}
            self.queued = []

        def rpop(self, key):
            return self.queue.pop(0) if self.queue else None

        def hgetall(self, key):
            return self.store.get(key, {})

        def hset(self, key, mapping=None, **kw):
            self.jobs[key] = dict(mapping or {})

        def expire(self, key, ttl):
            pass

        def xadd(self, stream, mapping):
            self.queued.append((stream, mapping))

        def scan_iter(self, pattern):
            return []

    fake = FR()

    monkeypatch.setattr(orchestrator_agent, "r", fake)
    monkeypatch.setattr(redis_manager, "r", fake)
    monkeypatch.setattr(orchestrator_agent, "compute_backlog_target", lambda: 1)
    monkeypatch.setattr(orchestrator_agent, "pending_count", lambda *a, **k: 0)
    monkeypatch.setattr(orchestrator_agent, "any_darkling_workers", lambda: False)
    monkeypatch.setattr(orchestrator_agent, "cache_wordlist", lambda p: "")
    monkeypatch.setattr(orchestrator_agent, "average_benchmark_rate", lambda: 0.0)
    monkeypatch.setattr(orchestrator_agent, "estimate_keyspace", lambda m, c: 0)
    monkeypatch.setattr(orchestrator_agent, "compute_batch_range", lambda r, k: (0, 100))

    calls = []

    def fake_is_already_cracked(h):
        calls.append(h)
        return h == "h1"

    monkeypatch.setattr(orchestrator_agent, "is_already_cracked", fake_is_already_cracked)

    orchestrator_agent.dispatch_batches()

    assert calls == ["h1", "h2"]
    stream, mapping = fake.queued[-1]
    job = fake.jobs[f"job:{mapping['job_id']}"]
    assert json.loads(job["hashes"]) == ["h2"]

