import os
import time
import threading
import hashlib
import subprocess
import requests

from .crypto_utils import sign_message


def md5_speed() -> float:
    """Return MB/s for computing md5 on random data."""
    data = os.urandom(1024 * 1024)  # 1 MB block
    start = time.time()
    for _ in range(50):
        hashlib.md5(data).digest()
    duration = time.time() - start
    if duration == 0:
        return 0.0
    return 50 / duration


def apply_flash_settings(gpu: dict, settings: dict) -> bool:
    """Attempt to apply clock/power settings to a GPU."""
    index = str(gpu.get("index", 0))
    vendor = settings.get("vendor", "nvidia").lower()
    try:
        if vendor == "nvidia":
            pl = settings.get("power_limit")
            if pl:
                subprocess.check_call(
                    [
                        "nvidia-smi",
                        "-i",
                        index,
                        "-pl",
                        str(pl),
                    ]
                )
            core = settings.get("core_clock")
            if core:
                subprocess.check_call(
                    [
                        "nvidia-smi",
                        "-i",
                        index,
                        "--lock-gpu-clocks",
                        str(core),
                    ]
                )
            mem = settings.get("mem_clock")
            if mem:
                subprocess.check_call(
                    [
                        "nvidia-smi",
                        "-i",
                        index,
                        "--lock-memory-clocks",
                        str(mem),
                    ]
                )
        elif vendor == "amd":
            pl = settings.get("power_limit")
            if pl:
                subprocess.check_call(
                    [
                        "rocm-smi",
                        "-d",
                        index,
                        "--setpowerlimit",
                        str(pl),
                    ]
                )
            core = settings.get("core_clock")
            if core:
                subprocess.check_call(
                    [
                        "rocm-smi",
                        "-d",
                        index,
                        "--setsclk",
                        str(core),
                    ]
                )
            mem = settings.get("mem_clock")
            if mem:
                subprocess.check_call(
                    [
                        "rocm-smi",
                        "-d",
                        index,
                        "--setmclk",
                        str(mem),
                    ]
                )
    except FileNotFoundError:
        return False
    except Exception:
        return False
    return True


class GPUFlashManager(threading.Thread):
    """Background thread that handles BIOS flashing/undervolting."""

    def __init__(self, worker_id: str, server_url: str, gpus: list[dict]):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.server_url = server_url
        self.gpus = gpus
        self.running = True

    def process_task(self, gpu_uuid: str, settings: dict):
        gpu = next((g for g in self.gpus if g["uuid"] == gpu_uuid), None)
        if not gpu:
            return
        baseline = md5_speed()
        success = apply_flash_settings(gpu, settings)
        post = md5_speed() if success else 0
        success = success and post >= baseline * 0.8
        payload = {
            "worker_id": self.worker_id,
            "gpu_uuid": gpu_uuid,
            "success": success,
            "signature": sign_message(self.worker_id),
        }
        try:
            requests.post(f"{self.server_url}/flash_result", json=payload, timeout=5)
        except Exception:
            pass

    def run(self):
        while self.running:
            try:
                params = {
                    "worker_id": self.worker_id,
                    "signature": sign_message(self.worker_id),
                }
                resp = requests.get(
                    f"{self.server_url}/get_flash_task", params=params, timeout=5
                )
                task = resp.json()
                if task.get("status") != "ok":
                    time.sleep(10)
                    continue
                gpu_uuid = task.get("gpu_uuid")
                settings = task.get("settings", {})
                self.process_task(gpu_uuid, settings)
            except Exception:
                time.sleep(5)
