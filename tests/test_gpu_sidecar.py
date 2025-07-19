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

    captured = {}
    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        captured['cmd'] = cmd
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

    monkeypatch.setenv("HASHCAT_WORKLOAD", "4")
    monkeypatch.setenv("HASHCAT_OPTIMIZED", "1")
    founds = sidecar.run_hashcat(batch)
    assert founds == ["hash:pass"]
    assert sidecar.hashrate == 50.0
    assert sidecar.progress == 10
    # verify options were added to the hashcat command
    assert "-w" in captured['cmd'] and "4" in captured['cmd']
    assert "-O" in captured['cmd']


def test_darkling_engine_selected(monkeypatch):
    def fake_get(url, timeout=None):
        class Resp:
            def json(self_inner):
                return {"low_bw_engine": "darkling"}

        return Resp()

    monkeypatch.setattr(gpu_sidecar.requests, "get", fake_get)
    sidecar = gpu_sidecar.GPUSidecar(
        "worker",
        {"uuid": "gpu", "index": 0, "pci_link_width": 4},
        "http://sv",
    )

    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    captured = {}

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        captured["cmd"] = cmd
        return DummyProc(["{}"], "/tmp/job2.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda x: "sig")

    batch = {
        "batch_id": "job2",
        "hashes": json.dumps(["hash"]),
        "mask": "?a",
        "attack_mode": "mask",
        "hash_mode": "0",
    }

    sidecar.execute_job(batch)

    assert captured["cmd"][0] == "darkling-engine"


def test_power_limit_nvidia(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("worker", {"uuid": "gpu", "index": 1}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    cmds = {}

    def fake_check_call(cmd, stdout=None, stderr=None):
        cmds["cmd"] = cmd

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        return DummyProc(["{}"], "/tmp/job3.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda x: "sig")

    batch = {
        "batch_id": "job3",
        "hashes": json.dumps(["h"]),
        "mask": "?a",
        "attack_mode": "mask",
        "hash_mode": "0",
    }

    monkeypatch.setenv("GPU_POWER_LIMIT", "120")
    sidecar.run_hashcat(batch)

    assert cmds["cmd"][0] == "nvidia-smi"
    assert "120" in cmds["cmd"]


def test_power_limit_rocm(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("worker", {"uuid": "gpu", "index": 1}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    cmds = {}

    call_order = []

    def fake_check_call(cmd, stdout=None, stderr=None):
        call_order.append(cmd[0])
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError()
        cmds["cmd"] = cmd

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        return DummyProc(["{}"], "/tmp/job4.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda x: "sig")

    batch = {
        "batch_id": "job4",
        "hashes": json.dumps(["h"]),
        "mask": "?a",
        "attack_mode": "mask",
        "hash_mode": "0",
    }

    monkeypatch.setenv("GPU_POWER_LIMIT", "130")
    sidecar.run_hashcat(batch)

    assert call_order[0] == "nvidia-smi"
    assert cmds["cmd"][0] == "rocm-smi"
    assert "130" in cmds["cmd"]


def test_power_limit_intel(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("worker", {"uuid": "gpu", "index": 1}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    cmds = {}

    def fake_check_call(cmd, stdout=None, stderr=None):
        if cmd[0] in ("nvidia-smi", "rocm-smi"):
            raise FileNotFoundError()
        cmds["cmd"] = cmd

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        return DummyProc(["{}"], "/tmp/job5.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda x: "sig")

    batch = {
        "batch_id": "job5",
        "hashes": json.dumps(["h"]),
        "mask": "?a",
        "attack_mode": "mask",
        "hash_mode": "0",
    }

    monkeypatch.setenv("GPU_POWER_LIMIT", "140")
    sidecar.run_hashcat(batch)

    assert cmds["cmd"][0] == "intel_gpu_frequency"
    assert "140" in cmds["cmd"]

