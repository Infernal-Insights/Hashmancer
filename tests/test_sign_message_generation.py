import os
import sys
import importlib
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))


def _run_worker(tmp_path: Path):
    priv = tmp_path / "nested" / "worker_priv.pem"
    pub = tmp_path / "nested" / "worker_pub.pem"
    os.environ["PRIVATE_KEY_PATH"] = str(priv)
    os.environ["PUBLIC_KEY_PATH"] = str(pub)
    module = importlib.import_module("hashmancer.worker.hashmancer_worker.crypto_utils")
    module = importlib.reload(module)
    module._PRIVATE_KEY = None
    module.sign_message("msg")
    assert priv.exists()
    assert pub.exists()


def _run_server(tmp_path: Path, monkeypatch):
    priv = tmp_path / "srv" / "server_priv.pem"

    # Prevent default path from being created during import
    open_orig = open

    def fail_default(path, *a, **kw):
        if path == "./keys/private_key.pem":
            raise FileNotFoundError
        return open_orig(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fail_default)
    module = importlib.import_module("hashmancer.server.signing_utils")
    module = importlib.reload(module)
    monkeypatch.setattr("builtins.open", open_orig)
    module.KEY_PATH = str(priv)
    module._PRIVATE_KEY = None
    module.sign_message("msg")
    assert priv.exists()


def test_worker_sign_message_generates_keys(tmp_path):
    _run_worker(tmp_path)


def test_server_sign_message_generates_key(monkeypatch, tmp_path):
    _run_server(tmp_path, monkeypatch)
