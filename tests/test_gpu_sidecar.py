import os
import sys
import json
import subprocess
import pytest
from pathlib import Path


from hashmancer.worker.hashmancer_worker import gpu_sidecar


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
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda *a: "sig")

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
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda *a: "sig")

    batch = {
        "batch_id": "job2",
        "hashes": json.dumps(["hash"]),
        "mask": "?a",
        "attack_mode": "mask",
        "hash_mode": "0",
        "start": 0,
        "end": 1000,
    }

    sidecar.execute_job(batch)

    assert captured["cmd"][0] == "darkling-engine"
    assert "--start" in captured["cmd"] and "0" in captured["cmd"]
    assert "--end" in captured["cmd"] and "1000" in captured["cmd"]


def test_custom_mask_charsets(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("worker", {"uuid": "gpu", "index": 0}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    captured = {}

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        captured["cmd"] = cmd
        return DummyProc(["{}"], "/tmp/job7.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda *a: "sig")

    batch = {
        "batch_id": "job7",
        "hashes": json.dumps(["h"]),
        "mask": "?1?2?3",
        "attack_mode": "mask",
        "hash_mode": "0",
        "mask_charsets": json.dumps({"?1": "ABC", "?2": "def", "?3": "123"}),
    }

    sidecar.run_darkling_engine(batch)

    assert "-1" in captured["cmd"] and "ABC" in captured["cmd"]
    assert "-2" in captured["cmd"] and "def" in captured["cmd"]
    assert "-3" in captured["cmd"] and "123" in captured["cmd"]


def test_reuse_preloaded_charsets(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("worker", {"uuid": "gpu", "index": 0}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    cmds = []

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        cmds.append(cmd)
        return DummyProc(["{}"], f"/tmp/job8{len(cmds)}.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda *a: "sig")

    batch = {
        "batch_id": "job8",
        "hashes": json.dumps(["h"]),
        "mask": "?1",
        "attack_mode": "mask",
        "hash_mode": "0",
        "mask_charsets": json.dumps({"?1": "ABC"}),
        "start": 0,
        "end": 10,
    }

    sidecar.run_darkling_engine(batch)
    sidecar.run_darkling_engine(batch)

    # first call should include charset options, second should not
    assert any("-1" in c for c in cmds[0])
    assert not any("-1" in c for c in cmds[1])


def test_power_limit_nvidia(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("worker", {"uuid": "gpu", "index": 1}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    cmds = {}

    def fake_check_call(cmd, stdout=None, stderr=None, **kwargs):
        cmds["cmd"] = cmd

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        return DummyProc(["{}"], "/tmp/job3.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda *a: "sig")

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

    def fake_check_call(cmd, stdout=None, stderr=None, **kwargs):
        call_order.append(cmd[0])
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError()
        cmds["cmd"] = cmd

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        return DummyProc(["{}"], "/tmp/job4.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda *a: "sig")

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


def test_power_overdrive_rocm(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("worker", {"uuid": "gpu", "index": 1}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    cmds = {}
    call_order = []

    def fake_check_call(cmd, stdout=None, stderr=None, **kwargs):
        call_order.append(cmd[0])
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError()
        cmds["cmd"] = cmd

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        return DummyProc(["{}"], "/tmp/job4b.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda *a: "sig")

    batch = {
        "batch_id": "job4b",
        "hashes": json.dumps(["h"]),
        "mask": "?a",
        "attack_mode": "mask",
        "hash_mode": "0",
    }

    monkeypatch.setenv("GPU_POWER_LIMIT", "+10%")
    sidecar.run_hashcat(batch)

    assert cmds["cmd"][0] == "rocm-smi"
    assert call_order == ["rocm-smi"]
    assert "--setpoweroverdrive" in cmds["cmd"]
    assert "+10" in cmds["cmd"]


def test_power_limit_intel(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("worker", {"uuid": "gpu", "index": 1}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    cmds = {}

    def fake_check_call(cmd, stdout=None, stderr=None, **kwargs):
        if cmd[0] in ("nvidia-smi", "rocm-smi"):
            raise FileNotFoundError()
        cmds["cmd"] = cmd

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        return DummyProc(["{}"], "/tmp/job5.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda *a: "sig")

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


def test_sidecar_run_executes_job(monkeypatch):
    class DummyResp:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    # First call for server_status, second for get_batch
    def fake_get(url, params=None, timeout=None):
        if "server_status" in url:
            return DummyResp({"low_bw_engine": "hashcat"})
        return DummyResp({"batch_id": "job6", "hashes": "[]", "mask": "", "attack_mode": "mask"})

    monkeypatch.setattr(gpu_sidecar.requests, "get", fake_get)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda *a: "sig")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    executed = {}

    def fake_execute(self, batch):
        executed["batch"] = batch
        self.running = False

    monkeypatch.setattr(gpu_sidecar.GPUSidecar, "execute_job", fake_execute)

    sidecar = gpu_sidecar.GPUSidecar("worker", {"uuid": "gpu", "index": 0}, "http://sv")
    cleaned = []
    sidecar.darkling_ctx.cleanup = lambda: cleaned.append(True)
    sidecar.run()

    assert executed["batch"]["batch_id"] == "job6"
    assert cleaned == [True]


def test_run_hashcat_benchmark(monkeypatch):
    outputs = {
        0: "Speed.#1.........: 10.0 MH/s (42.00ms)\n",
        100: "Speed.#1.........: 20.0 MH/s (42.00ms)\n",
        1000: "Speed.#1.........: 30.0 MH/s (42.00ms)\n",
    }

    def fake_check_output(cmd, stderr=None, text=None):
        mode = int(cmd[cmd.index("-m") + 1])
        return outputs[mode]

    monkeypatch.setattr(gpu_sidecar.subprocess, "check_output", fake_check_output)

    gpu = {"uuid": "gpu", "index": 0}
    rates = gpu_sidecar.run_hashcat_benchmark(gpu)
    assert rates == {"MD5": 10.0e6, "SHA1": 20.0e6, "NTLM": 30.0e6}


def test_run_darkling_benchmark(monkeypatch):
    def fake_popen(cmd, stdout=None, stderr=None, text=None):
        mode = int(cmd[cmd.index("-m") + 1])
        if mode == 0:
            return DummyProc(["{\"speed\": [10]}"], "/tmp/db1.out")
        if mode == 100:
            return DummyProc(["{\"speed\": [20]}"], "/tmp/db2.out")
        raise subprocess.SubprocessError("unsupported")

    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)

    gpu = {"uuid": "gpu", "index": 0}
    rates = gpu_sidecar.run_darkling_benchmark(gpu)
    assert rates == {"MD5": 10.0, "SHA1": 20.0, "NTLM": 0.0}


def test_parse_benchmark_units(monkeypatch):
    outputs = {
        0: "Speed.#1.........: 1.5 GH/s (42.00ms)\n",
        100: "Speed.#1.........: 500 kH/s (42.00ms)\n",
        1000: "Speed.#1.........: 2.5 TH/s (42.00ms)\n",
    }

    def fake_check_output(cmd, stderr=None, text=None):
        mode = int(cmd[cmd.index("-m") + 1])
        return outputs[mode]

    monkeypatch.setattr(gpu_sidecar.subprocess, "check_output", fake_check_output)

    gpu = {"uuid": "gpu", "index": 0}
    rates = gpu_sidecar.run_hashcat_benchmark(gpu)
    assert rates == {"MD5": 1.5e9, "SHA1": 500e3, "NTLM": 2.5e12}


def test_parse_benchmark_unknown_unit(monkeypatch):
    outputs = {0: "Speed.#1.........: 7.5 QQ/s (42.00ms)\n"}

    def fake_check_output(cmd, stderr=None, text=None):
        return outputs[0]

    monkeypatch.setattr(gpu_sidecar.subprocess, "check_output", fake_check_output)

    gpu = {"uuid": "gpu", "index": 0}
    rates = gpu_sidecar.run_hashcat_benchmark(gpu)
    assert rates["MD5"] == 7.5


def test_darkling_mask_length_limit(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("w", {"uuid": "gpu", "index": 0}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    def fake_popen(cmd, stdout=None, stderr=None, text=None, env=None):
        return DummyProc(["{}"], "/tmp/joblen.out")

    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda *a: "sig")

    batch = {
        "batch_id": "joblen",
        "hashes": json.dumps(["h"]),
        "mask": "a" * (gpu_sidecar.MAX_MASK_LEN - 1),
        "attack_mode": "mask",
        "hash_mode": "0",
    }

    sidecar.run_darkling_engine(batch)

    batch["mask"] = "a" * gpu_sidecar.MAX_MASK_LEN
    with pytest.raises(ValueError):
        sidecar.run_darkling_engine(batch)


def test_darkling_hash_batching(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("w", {"uuid": "gpu", "index": 0}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    calls = []

    def fake_run_engine(self, engine, batch, range_start=None, range_end=None, skip_charsets=False):
        calls.append((engine, json.loads(batch["hashes"]), skip_charsets))
        return [f"{h}:pass" for h in json.loads(batch["hashes"])]

    monkeypatch.setattr(gpu_sidecar.GPUSidecar, "_run_engine", fake_run_engine)
    monkeypatch.setattr(gpu_sidecar.requests, "post", lambda *a, **k: None)
    monkeypatch.setattr(gpu_sidecar, "sign_message", lambda *a: "sig")

    hashes = [f"h{i}" for i in range(gpu_sidecar.MAX_HASHES + 5)]
    batch = {
        "batch_id": "jobbatch",
        "hashes": json.dumps(hashes),
        "mask": "?a",
        "attack_mode": "mask",
        "hash_mode": "0",
    }

    res = sidecar.run_darkling_engine(batch)
    assert len(calls) == 2
    assert len(res) == len(hashes)
    assert calls[0][2] is False
    assert calls[1][2] is True


def test_temp_files_cleanup_on_exception(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar("w", {"uuid": "gpu", "index": 0}, "http://sv")
    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    def fake_popen(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(gpu_sidecar.subprocess, "Popen", fake_popen)

    batch = {
        "batch_id": "jobexc",
        "hashes": json.dumps(["h"]),
        "mask": "?a",
        "attack_mode": "mask",
        "hash_mode": "0",
    }

    with pytest.raises(RuntimeError):
        sidecar.run_hashcat(batch)

    assert not Path("/tmp/jobexc.hashes").exists()
    assert not Path("/tmp/jobexc.out").exists()
    assert not Path("/tmp/jobexc.restore").exists()


def test_probability_index_order_inverse(monkeypatch):
    sidecar = gpu_sidecar.GPUSidecar(
        "w",
        {"uuid": "gpu", "index": 0},
        "http://sv",
        probabilistic_order=True,
        inverse_order=True,
    )

    monkeypatch.setattr(gpu_sidecar, "r", FakeRedis())

    captured = {}

    def fake_prob_order(mask, cs_map, markov, limit=None, *, inverse=False):
        captured["inverse"] = inverse
        return [0]

    monkeypatch.setattr(
        gpu_sidecar.statistics,
        "probability_index_order",
        fake_prob_order,
    )
    monkeypatch.setattr(
        gpu_sidecar.statistics,
        "load_markov",
        lambda lang="english": {},
    )
    monkeypatch.setattr(gpu_sidecar.GPUSidecar, "_run_engine", lambda *a, **k: [])

    batch = {
        "batch_id": "jobprob",
        "hashes": json.dumps(["h"]),
        "mask": "?a",
        "attack_mode": "mask",
        "hash_mode": "0",
    }

    sidecar.run_darkling_engine(batch)

    assert captured.get("inverse") is True
