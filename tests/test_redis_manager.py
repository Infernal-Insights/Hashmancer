import os
import sys
import json
from uuid import UUID

ROOT = os.path.dirname(os.path.dirname(__file__))

import redis_manager


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.queue = []
        self.prio = {}

    def sadd(self, key, value):
        self.store.setdefault(key, set()).add(value)

    def smembers(self, key):
        return self.store.get(key, set())

    def hset(self, key, mapping=None, **kwargs):
        self.store[key] = dict(mapping or {})

    def expire(self, key, ttl):
        pass

    def lpush(self, name, value):
        self.queue.insert(0, value)

    def rpush(self, name, value):
        self.store.setdefault(name, []).append(value)

    def lrem(self, name, count, value):
        lst = self.store.get(name, [])
        while value in lst:
            lst.remove(value)

    def rpop(self, name):
        return self.queue.pop() if self.queue else None

    def zadd(self, name, mapping):
        self.prio.update(mapping)

    def zrevrange(self, name, start, end):
        ordered = sorted(self.prio.items(), key=lambda x: -x[1])
        return [k for k, _ in ordered[start:end+1]]

    def zrem(self, name, member):
        self.prio.pop(member, None)

    def hgetall(self, key):
        return dict(self.store.get(key, {}))


def test_store_and_get_batch(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(redis_manager, "r", fake)
    monkeypatch.setattr(redis_manager.uuid, "uuid4", lambda: UUID("11111111-1111-1111-1111-111111111111"))
    monkeypatch.setattr(redis_manager.orchestrator_agent, "build_mask_charsets", lambda: {})
    monkeypatch.setattr(redis_manager.orchestrator_agent, "estimate_keyspace", lambda m, c: 10)

    batch_id = redis_manager.store_batch(["h"], mask="?a")
    assert batch_id == "11111111-1111-1111-1111-111111111111"
    assert fake.queue == [batch_id]

    data = redis_manager.get_next_batch()
    assert data["batch_id"] == batch_id
    assert json.loads(data["hashes"]) == ["h"]
    assert data["mask"] == "?a"
    assert data["keyspace"] == 10
    assert batch_id in fake.smembers("hash_batches:h")
