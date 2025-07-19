import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Worker.hashmancer_worker import worker_agent

class DummyResp:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data

class FakeRedis:
    def __init__(self):
        self.store = {}
    def hset(self, key, mapping=None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or {})
    def sadd(self, key, *values):
        self.store.setdefault(key, set()).update(values)
    def set(self, key, value):
        self.store[key] = value

def test_register_worker_success(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(worker_agent, "r", fake)

    def mock_post(url, json=None, timeout=None):
        return DummyResp({"status": "ok", "waifu": "TestWaifu"})

    monkeypatch.setattr(worker_agent.requests, "post", mock_post)
    monkeypatch.setattr(worker_agent, "load_public_key_pem", lambda: "PUB")

    gpus = [{"uuid": "gpu1", "model": "GPU", "pci_bus": "0", "memory_mb": 0, "pci_link_width": 16}]
    name = worker_agent.register_worker("id123", gpus)

    assert name == "TestWaifu"
    assert fake.store.get("worker_name") == "TestWaifu"
