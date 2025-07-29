import pytest
from httpx import AsyncClient, ASGITransport


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.stream = []
        self.acked = None
        self.read_args = None

    def set(self, key, value, ex=None):
        self.store[key] = value
        if ex:
            self.store[f"ttl:{key}"] = ex

    def delete(self, key):
        self.store.pop(key, None)

    def sadd(self, key, value):
        self.store.setdefault(key, set()).add(value)

    def smembers(self, key):
        return self.store.get(key, set())

    def hset(self, key, mapping=None, *args, **kwargs):
        if mapping is not None and not isinstance(mapping, dict):
            field = mapping
            value = args[0] if args else None
            self.store.setdefault(key, {})[field] = value
        else:
            self.store.setdefault(key, {}).update(mapping or {})
            self.store[key].update(kwargs)

    def hgetall(self, key):
        return dict(self.store.get(key, {}))

    def rpush(self, name, value):
        self.store.setdefault(name, []).append(value)

    def xgroup_create(self, *a, **kw):
        pass

    def xreadgroup(self, group, consumer, streams, count=1, block=0):
        self.read_args = (group, streams)
        return self.stream

    def xack(self, stream, group, msg_id):
        self.acked = (stream, group, msg_id)


@pytest.mark.asyncio
async def test_basic_api_flow(monkeypatch, tmp_path):
    import sys, importlib
    for mod in ["fastapi", "fastapi.middleware.cors", "fastapi.responses", "pydantic"]:
        sys.modules.pop(mod, None)
    import fastapi  # noqa: F401  ensure real modules
    import fastapi.middleware.cors  # noqa: F401
    import fastapi.responses  # noqa: F401
    import pydantic  # noqa: F401

    import hashmancer.server.app.api.models as models
    importlib.reload(models)
    import hashmancer.server.app.app as app_mod
    importlib.reload(app_mod)
    import hashmancer.server.main as main
    importlib.reload(main)
    app = main.app

    fake = FakeRedis()
    monkeypatch.setattr(main, "r", fake)
    monkeypatch.setitem(main.CONFIG, "portal_passkey", "pass")
    monkeypatch.setattr(main, "PORTAL_PASSKEY", "pass")
    monkeypatch.setattr(main, "start_loops", lambda: [])
    monkeypatch.setattr(main, "print_logo", lambda *a, **k: None)
    monkeypatch.setattr(main, "verify_signature", lambda *a, **k: True)
    monkeypatch.setattr(main, "verify_signature_with_key", lambda *a, **k: True)
    monkeypatch.setattr(main, "assign_waifu", lambda *a, **k: "Agent")
    monkeypatch.setattr(main, "WORDLISTS_DIR", tmp_path)

    (tmp_path / "list.txt").write_text("data")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/login", json={"passkey": "pass"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        sid = data["cookie"].split("|")[0]
        assert f"session:{sid}" in fake.store

        payload = {
            "worker_id": "w1",
            "timestamp": 0,
            "signature": "sig",
            "pubkey": "pub",
            "mode": "eco",
            "provider": "on-prem",
            "hardware": {},
        }
        resp = await client.post("/register_worker", json=payload)
        assert resp.status_code == 200
        assert resp.json()["waifu"] == "Agent"
        assert fake.store["worker:Agent"]["id"] == "w1"

        resp = await client.get("/wordlists")
        assert resp.status_code == 200
        assert "list.txt" in resp.json()

        fake.stream = [("jobs", [("1-0", {"job_id": "job1"})])]
        fake.store["job:job1"] = {"batch_id": "batch1"}
        fake.store["worker:w1"] = {"low_bw_engine": "hashcat"}

        resp = await client.get(
            "/get_batch",
            params={"worker_id": "w1", "timestamp": 0, "signature": "sig"},
        )
        assert resp.status_code == 200
        assert resp.json()["batch_id"] == "batch1"

    for mod in ["fastapi", "fastapi.middleware.cors", "fastapi.responses", "pydantic"]:
        sys.modules.pop(mod, None)

