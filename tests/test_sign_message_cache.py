import os
import sys
import importlib
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Server"))


def _make_key(path: Path) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    data = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    path.write_bytes(data)
    return path


def _run_test(module_name: str, tmp_path: Path, monkeypatch, env_var: str | None = None, path_attr: str | None = None):
    key_path = _make_key(tmp_path / "key.pem")
    if env_var:
        monkeypatch.setenv(env_var, str(key_path))
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    if path_attr:
        setattr(module, path_attr, str(key_path))
    module._PRIVATE_KEY = None

    calls = []
    orig_open = open

    def counting_open(*args, **kwargs):
        calls.append(args[0])
        return orig_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    module.sign_message("one")
    module.sign_message("two")
    assert len(calls) == 1


def test_worker_sign_message_load_once(monkeypatch, tmp_path):
    _run_test(
        "Worker.hashmancer_worker.crypto_utils",
        tmp_path,
        monkeypatch,
        env_var="PRIVATE_KEY_PATH",
    )


def test_server_sign_message_load_once(monkeypatch, tmp_path):
    _run_test(
        "Server.signing_utils",
        tmp_path,
        monkeypatch,
        path_attr="KEY_PATH",
    )
