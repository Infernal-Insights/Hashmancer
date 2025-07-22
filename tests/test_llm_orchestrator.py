import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Server"))

import orchestrator_agent

# ensure redis exceptions are simple
orchestrator_agent.redis.exceptions.ResponseError = Exception


class StubLLM:
    def __init__(self, queue="high", size=(0, 100)):
        self.queue = queue
        self.size = size

    def choose_job_stream(self, batch_info, high_pending, low_pending):
        return self.queue

    def suggest_batch_size(self, batch_info):
        return self.size


class FakeRedis:
    def __init__(self):
        self.queue = ["1"]
        self.store = {"batch:1": {"hashes": json.dumps(["h"]), "mask": "?d?d"}}
        self.jobs = {}
        self.streams = []

    def rpop(self, name):
        return self.queue.pop(0) if self.queue else None

    def hgetall(self, key):
        return self.store.get(key, {})

    def hset(self, key, mapping=None, **kw):
        self.jobs[key] = dict(mapping or kw)

    def expire(self, key, ttl):
        pass

    def xadd(self, stream, mapping):
        self.streams.append((stream, mapping))

    def scan_iter(self, pattern):
        return []


def setup(monkeypatch, queue, size):
    fake = FakeRedis()
    monkeypatch.setattr(orchestrator_agent, "r", fake)
    monkeypatch.setattr(orchestrator_agent, "_LLM", StubLLM(queue, size))
    monkeypatch.setattr(orchestrator_agent, "compute_backlog_target", lambda: 1)
    monkeypatch.setattr(orchestrator_agent, "pending_count", lambda *a, **k: 0)
    monkeypatch.setattr(orchestrator_agent, "any_darkling_workers", lambda: True)
    monkeypatch.setattr(orchestrator_agent, "cache_wordlist", lambda p: "")
    monkeypatch.setattr(orchestrator_agent, "build_mask_charsets", lambda lang=None: {})
    monkeypatch.setattr(orchestrator_agent, "is_already_cracked", lambda h: False)
    return fake


def test_llm_low_queue(monkeypatch):
    fake = setup(monkeypatch, "low", (5, 10))
    orchestrator_agent.dispatch_batches()
    assert fake.streams
    stream, mapping = fake.streams[0]
    assert stream == orchestrator_agent.LOW_BW_JOB_STREAM
    job = fake.jobs[f"job:{mapping['job_id']}"]
    assert job["start"] == 5
    assert job["end"] == 10


def test_llm_high_queue(monkeypatch):
    fake = setup(monkeypatch, "high", (1, 2))
    orchestrator_agent.dispatch_batches()
    assert fake.streams
    stream, mapping = fake.streams[0]
    assert stream == orchestrator_agent.JOB_STREAM
    job = fake.jobs[f"job:{mapping['job_id']}"]
    assert job["start"] == 1
    assert job["end"] == 2
