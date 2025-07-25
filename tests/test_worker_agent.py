import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "Server"))

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
    monkeypatch.setattr(worker_agent, "sign_message", lambda *a: "sig")

    gpus = [
        {
            "uuid": "gpu1",
            "model": "GPU",
            "pci_bus": "0",
            "memory_mb": 0,
            "pci_link_width": 16,
        }
    ]
    name = worker_agent.register_worker("id123", gpus)

    assert name == "TestWaifu"
    assert fake.store.get("worker_name") == "TestWaifu"


def test_register_worker_retry(monkeypatch):
    class FlakyRedis(FakeRedis):
        def __init__(self, fails=2):
            super().__init__()
            self.fails = fails

        def _maybe_fail(self):
            if self.fails > 0:
                self.fails -= 1
                raise worker_agent.RedisError("down")

        def hset(self, *a, **k):
            self._maybe_fail()
            return super().hset(*a, **k)

        def sadd(self, *a, **k):
            self._maybe_fail()
            return super().sadd(*a, **k)

        def set(self, *a, **k):
            self._maybe_fail()
            return super().set(*a, **k)

    fake = FlakyRedis()
    monkeypatch.setattr(worker_agent, "r", fake)

    monkeypatch.setattr(
        worker_agent.requests,
        "post",
        lambda url, json=None, timeout=None: DummyResp({"status": "ok", "waifu": "W"}),
    )
    monkeypatch.setattr(worker_agent, "load_public_key_pem", lambda: "PUB")
    monkeypatch.setattr(worker_agent, "sign_message", lambda *a: "sig")

    sleeps = []
    monkeypatch.setattr(worker_agent.time, "sleep", lambda s: sleeps.append(s))

    gpus = [
        {
            "uuid": "gpu1",
            "model": "GPU",
            "pci_bus": "0",
            "memory_mb": 0,
            "pci_link_width": 16,
        }
    ]
    name = worker_agent.register_worker("id-retry", gpus)

    assert name == "W"
    assert fake.store.get("worker_name") == "W"
    assert len(sleeps) >= 1


def test_benchmark_low_bw_gpu(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(worker_agent, "r", fake)
    monkeypatch.setattr(worker_agent, "print_logo", lambda: None)
    monkeypatch.setattr(
        worker_agent,
        "detect_gpus",
        lambda: [{"uuid": "g1", "index": 0, "pci_link_width": 4}],
    )
    monkeypatch.setattr(worker_agent, "register_worker", lambda wid, g: "name")

    monkeypatch.setattr(
        worker_agent.requests,
        "get",
        lambda url, timeout=5: DummyResp(
            {
                "low_bw_engine": "darkling",
                "probabilistic_order": False,
                "markov_lang": "english",
            }
        ),
    )

    called = {}

    def fake_dark(gpu):
        called["dark"] = gpu
        return {}

    def fake_post(url, json=None, timeout=None):
        pass

    monkeypatch.setattr(worker_agent, "run_darkling_benchmark", fake_dark)
    monkeypatch.setattr(
        worker_agent, "run_hashcat_benchmark", lambda g, engine="hashcat": {}
    )
    monkeypatch.setattr(worker_agent.requests, "post", fake_post)
    monkeypatch.setattr(worker_agent, "sign_message", lambda *a: "sig")

    class DummySidecar:
        def __init__(
            self, name, gpu, url, probabilistic_order=False, markov_lang="english"
        ):
            self.gpu = gpu
            self.progress = 0.0
            self.current_job = None
            self.running = True

        def start(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(worker_agent, "GPUSidecar", DummySidecar)

    class DummyFlash:
        def __init__(self, *a, **k):
            self.running = False

        def start(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(worker_agent, "GPUFlashManager", DummyFlash)

    monkeypatch.setattr(worker_agent, "get_gpu_temps", lambda: [])

    def stop(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(worker_agent.time, "sleep", stop)

    try:
        worker_agent.main()
    except KeyboardInterrupt:
        pass

    assert "dark" in called


def test_prob_order_from_server(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(worker_agent, "r", fake)
    monkeypatch.setattr(worker_agent, "print_logo", lambda: None)
    monkeypatch.setattr(
        worker_agent, "detect_gpus", lambda: [{"uuid": "g1", "index": 0}]
    )
    monkeypatch.setattr(worker_agent, "register_worker", lambda wid, g: "name")

    monkeypatch.setattr(
        worker_agent.requests,
        "get",
        lambda url, timeout=5: DummyResp(
            {
                "low_bw_engine": "hashcat",
                "probabilistic_order": True,
                "markov_lang": "spanish",
            }
        ),
    )

    captured = {}

    def fake_post(url, json=None, timeout=None):
        pass

    monkeypatch.setattr(
        worker_agent, "run_hashcat_benchmark", lambda g, engine="hashcat": {}
    )
    monkeypatch.setattr(worker_agent.requests, "post", fake_post)
    monkeypatch.setattr(worker_agent, "sign_message", lambda *a: "sig")

    class DummySidecar:
        def __init__(
            self, name, gpu, url, probabilistic_order=False, markov_lang="english"
        ):
            captured["prob"] = probabilistic_order
            captured["lang"] = markov_lang
            self.gpu = gpu
            self.progress = 0.0
            self.current_job = None
            self.running = True

        def start(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(worker_agent, "GPUSidecar", DummySidecar)

    class DummyFlash:
        def __init__(self, *a, **k):
            self.running = False

        def start(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(worker_agent, "GPUFlashManager", DummyFlash)

    monkeypatch.setattr(worker_agent, "get_gpu_temps", lambda: [])

    def stop(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(worker_agent.time, "sleep", stop)

    try:
        worker_agent.main([])
    except KeyboardInterrupt:
        pass

    assert captured["prob"] is True
    assert captured["lang"] == "spanish"
