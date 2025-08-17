import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))

from hashmancer.server import pattern_to_mask

class FakeRedis:
    def __init__(self):
        self.data = {}

    def zrevrange(self, key, start, end):
        items = self.data.get(key, {})
        ordered = sorted(items.items(), key=lambda x: -x[1])
        return [p for p, _ in ordered[start:end+1]]

def test_pattern_to_mask_simple():
    assert pattern_to_mask.pattern_to_mask("$U$l$d") == "?u?l?d"

def test_get_top_masks(monkeypatch):
    fake = FakeRedis()
    fake.data = {"dictionary:patterns:english": {"$U$l$d": 5, "$l$l": 2}}
    monkeypatch.setattr(pattern_to_mask, "get_redis", lambda: fake)
    masks = pattern_to_mask.get_top_masks(1)
    assert masks == ["?u?l?d"]

