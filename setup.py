"""Entry module exposing helper functions for installation scripts."""

from pathlib import Path

try:
    from manage import main  # type: ignore
except ImportError:  # pragma: no cover - fallback when not on path
    def main() -> None:  # type: ignore
        raise RuntimeError("manage script unavailable")
import os
import shutil
from hashmancer.utils import event_logger
try:
    import requests
except ImportError:  # pragma: no cover - optional during build
    requests = None  # type: ignore

CONFIG_DIR = Path.home() / ".hashmancer"

__all__ = [
    "main",
    "download_prebuilt_engine",
    "CONFIG_DIR",
    "requests",
]


def download_prebuilt_engine() -> None:
    """Download a vendor specific darkling-engine if DARKLING_ENGINE_URL is set."""
    base = os.getenv("DARKLING_ENGINE_URL")
    if not base:
        return

    backend = os.getenv("DARKLING_GPU_BACKEND")
    if not backend:
        if shutil.which("nvidia-smi"):
            backend = "cuda"
        elif shutil.which("rocm-smi"):
            backend = "hip"
        else:
            backend = "opencl"

    url = f"{base}-{backend}"

    dest_dir = CONFIG_DIR / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "darkling-engine"

    try:
        print(f"\U0001F53D Downloading prebuilt darkling-engine ({backend})...")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        dest.chmod(0o755)
        print(f"Downloaded darkling-engine to {dest}")
    except (requests.RequestException, OSError) as e:
        event_logger.log_error("setup", "system", "S001", "Download prebuilt engine failed", e)
        print(f"⚠️  Failed to download prebuilt engine: {e}")

