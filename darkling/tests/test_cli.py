import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "attack, extra",
    [
        ("0", ["--mask", "?d"]),
        ("1", []),
        ("2", ["--ruleset", "configs/ruleset_baked256.json"]),
        ("3", []),
    ],
)
def test_darkling_cli_attacks(attack, extra):
    """Basic smoke tests for darkling-engine CLI attack options."""
    if shutil.which("darkling-engine") is None:
        pytest.skip("darkling-engine not built")

    darkling_dir = Path(__file__).resolve().parents[1]
    cmd = ["darkling-engine", "--attack", attack, *extra]
    proc = subprocess.run(cmd, cwd=darkling_dir, capture_output=True, text=True)
    assert proc.returncode == 0
