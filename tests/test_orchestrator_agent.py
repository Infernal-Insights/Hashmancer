import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Server"))
import json

import orchestrator_agent

orchestrator_agent.redis.exceptions.ResponseError = Exception


class FakeRedis:
    def __init__(self, info=None, raise_err=False):
        self.info = info
        self.raise_err = raise_err
        self.group_created = False

    def xpending(self, stream, group):
        if self.raise_err:
            raise orchestrator_agent.redis.exceptions.ResponseError("no group")
        return self.info

    def xgroup_create(self, stream, group, id="0", mkstream=True):
        self.group_created = True


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


def test_darkling_transformed_mask(monkeypatch, tmp_path):
    wl = tmp_path / "wl.txt"
    wl.write_text("Abc\n1234\n")

    class DR(FakeRedis):
        def __init__(self):
            super().__init__()
            self.jobs = {}
            self.store = {"batch:1": {"hashes": "[]", "wordlist": str(wl)}}
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
    monkeypatch.setattr(orchestrator_agent, "compute_backlog_target", lambda: 1)
    monkeypatch.setattr(orchestrator_agent, "pending_count", lambda *a, **k: 0)
    monkeypatch.setattr(orchestrator_agent, "any_darkling_workers", lambda: True)
    monkeypatch.setattr(orchestrator_agent, "cache_wordlist", lambda p: "key")
    monkeypatch.setattr(orchestrator_agent, "generate_mask", lambda length: "$U$l$d$s")

    orchestrator_agent.dispatch_batches()

    stream, mapping = fake.queued[-1]
    job = fake.jobs[f"job:{mapping['job_id']}"]

    assert stream == orchestrator_agent.LOW_BW_JOB_STREAM
    assert job["attack_mode"] == "mask"
    assert job["mask"] == "?1?2?3?4"
    cs = json.loads(job["mask_charsets"])
    assert "?1" in cs and "?2" in cs and "?3" in cs and "?4" in cs
    assert job["start"] == 0
    assert job["end"] == 1000

