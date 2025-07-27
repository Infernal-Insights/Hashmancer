import os
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("setup", Path(__file__).resolve().parents[1] / "setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_download_prebuilt_engine(monkeypatch, tmp_path):
    data = b'binary-data'

    class Resp:
        status_code = 200
        content = data
        def raise_for_status(self):
            pass

    monkeypatch.setenv('DARKLING_ENGINE_URL', 'http://example.com/engine')
    monkeypatch.setenv('DARKLING_GPU_BACKEND', 'cuda')

    urls = []

    def fake_get(url, *args, **kwargs):
        urls.append(url)
        return Resp()

    monkeypatch.setattr(setup.requests, 'get', fake_get)
    monkeypatch.setattr(setup, 'CONFIG_DIR', tmp_path)

    dest = tmp_path / 'bin' / 'darkling-engine'
    if dest.exists():
        dest.unlink()

    setup.download_prebuilt_engine()
    assert dest.exists()
    assert dest.read_bytes() == data
    assert urls == ['http://example.com/engine-cuda']
