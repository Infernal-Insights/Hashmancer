import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Worker.hashmancer_worker import gpu_sidecar


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def hset(self, key, mapping=None, **kwargs):
        pass


class DummyStdout:
    def __init__(self, lines):
        self.lines = lines
        self.idx = 0

    def readline(self):
        if self.idx < len(self.lines):
            line = self.lines[self.idx]
            self.idx += 1
            return line
        return ""


class DummyProc:
    def __init__(self, lines, outfile):
        self.stdout = DummyStdout(lines)
        self.returncode = 0
        self.outfile = outfile
        self.done = False

    def poll(self):
        if self.stdout.idx >= len(self.stdout.lines) and not self.done:
            Path(self.outfile).write_text("hash:pass\n")
            self.done = True
            return self.returncode
        return None


def test_run_hashcat(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("worker", {"uuid": "gpu", "index": 0}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        return DummyProc(['{"speed": [50], "progress": 10}'], "/tmp/job1.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda x: "sig")

    batch = {
        "batch_id": "job1",
        "hashes": json.dumps(["hash"]),
        "mask": "?a",
        "attack_mode": "mask",
        "hash_mode": "0",
    }

    founds = sidecar.run_hashcat(batch)
    assert founds == ["hash:pass"]
    assert sidecar.hashrate == 50.0
    assert sidecar.progress == 10

