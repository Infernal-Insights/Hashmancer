import sys, os
import asyncio
import sys
import os
import types

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,
    pydantic_stub,
    install_stubs,
    FakeApp,
)

install_stubs()

# Stub cryptography pieces used by auth_utils

# Stub psutil for system metrics
psutil_stub = types.ModuleType("psutil")
psutil_stub.cpu_percent = lambda interval=None: 10.0
psutil_stub.virtual_memory = lambda: types.SimpleNamespace(percent=20.0, used=123456)
psutil_stub.disk_usage = lambda path: types.SimpleNamespace(percent=30.0)
psutil_stub.getloadavg = lambda: (1.0, 0.5, 0.25)
sys.modules.setdefault("psutil", psutil_stub)


import main

class DummyRedis:
    def scard(self, key):
        return 0
    def llen(self, key):
        return 0


def test_server_status_reports_system_metrics(monkeypatch):
    fake = DummyRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main.orchestrator_agent, 'compute_backlog_target', lambda: 7)
    monkeypatch.setattr(main.orchestrator_agent, 'pending_count', lambda: 3)
    status = asyncio.run(main.server_status())
    assert 'cpu_usage' in status
    assert 'memory_utilization' in status
    assert 'disk_space' in status
    assert 'cpu_load' in status
    assert 'memory_usage' in status
    assert 'backlog_target' in status
    assert 'pending_jobs' in status
    assert 'queued_batches' in status


def test_server_status_reports_llm_defaults(monkeypatch):
    fake = DummyRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main, 'LLM_TRAIN_EPOCHS', 4)
    monkeypatch.setattr(main, 'LLM_TRAIN_LEARNING_RATE', 0.003)
    monkeypatch.setattr(main.orchestrator_agent, 'compute_backlog_target', lambda: 0)
    monkeypatch.setattr(main.orchestrator_agent, 'pending_count', lambda: 0)
    status = asyncio.run(main.server_status())
    assert status['llm_train_epochs'] == 4
    assert status['llm_train_learning_rate'] == 0.003
