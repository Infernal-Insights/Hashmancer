import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))

from hashmancer.server import learn_trends
from hashmancer.darkling import statistics
from hashmancer.server import pattern_to_mask


class FakeRedis:
    def __init__(self):
        self.store = {}

    # sorted set helpers
    def zincrby(self, key, amt, member):
        self.store.setdefault(key, {})
        self.store[key][member] = self.store[key].get(member, 0) + amt

    def zrevrange(self, key, start, end, withscores=False):
        items = self.store.get(key, {})
        ordered = sorted(items.items(), key=lambda x: -x[1])
        if end == -1:
            result = ordered[start:]
        else:
            result = ordered[start:end + 1]
        if withscores:
            return [(p, v) for p, v in result]
        return [p for p, _ in result]

    def zrange(self, key, start, end, withscores=False):
        items = self.store.get(key, {})
        ordered = sorted(items.items(), key=lambda x: x[1])
        if end == -1:
            result = ordered[start:]
        else:
            result = ordered[start:end + 1]
        if withscores:
            return [(p, v) for p, v in result]
        return [p for p, _ in result]


def test_lang_specific_storage_and_top(monkeypatch, tmp_path):
    fake = FakeRedis()
    monkeypatch.setattr(learn_trends, "get_redis", lambda: fake)
    wl = tmp_path / "wl.txt"
    wl.write_text("Password1\nhello\n")

    learn_trends.process_wordlists(tmp_path, lang="testlang")

    assert f"dictionary:patterns:testlang" in fake.store

    res = learn_trends.top_patterns(1, lang="testlang", r=fake)
    assert res


def test_load_markov_lang(monkeypatch):
    fake = FakeRedis()
    fake.zincrby("dictionary:patterns:english", 3, "$U$l")
    fake.zincrby("dictionary:patterns:english", 1, "$l$l")

    markov = statistics.load_markov(r=fake, lang="english")
    assert markov["start"].get("$U") == 3


def test_get_top_masks_lang(monkeypatch):
    fake = FakeRedis()
    fake.zincrby("dictionary:patterns:french", 2, "$l$d")
    monkeypatch.setattr(pattern_to_mask, "get_redis", lambda: fake)
    masks = pattern_to_mask.get_top_masks(1, lang="french")
    assert masks == ["?l?d"]

