import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Worker.hashmancer_worker import flash_tuner


def test_tune_power_limit(monkeypatch):
    seq = iter([1.0, 0.95, 0.95, 0.75])
    monkeypatch.setattr(flash_tuner, "md5_speed", lambda: next(seq))
    applied = []

    def fake_apply(gpu, settings):
        applied.append(settings["power_limit"])
        return True

    monkeypatch.setattr(flash_tuner, "apply_flash_settings", fake_apply)
    preset = {"power_limit": 100, "core_clock": 1500, "mem_clock": 4000}
    best = flash_tuner.tune_power_limit(0, "nvidia", preset)
    assert best == 95
    assert applied == [100, 95, 90, 95]

