import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Server"))

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

