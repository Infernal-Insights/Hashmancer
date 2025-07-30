import os
import time
import threading
import hashlib
import subprocess
import requests
import logging
import shutil
from hashmancer.utils import event_logger

from .crypto_utils import sign_message


def detect_pci_address(index: int, vendor: str) -> str:
    """Return the PCI bus address for a GPU index."""
    try:
        if vendor == "nvidia":
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(index),
                    "--query-gpu=pci.bus_id",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            return out.strip()
        if vendor == "amd":
            out = subprocess.check_output(
                ["rocm-smi", "-d", str(index), "--showbus"],
                text=True,
            )
            for line in out.splitlines():
                if "PCI Bus" in line:
                    return line.split()[-1]
    except Exception:
        pass

    # fallback using lspci
    try:
        out = subprocess.check_output(["lspci", "-D"], text=True)
        vendor_name = "NVIDIA" if vendor == "nvidia" else "AMD"
        lines = [l for l in out.splitlines() if vendor_name in l.upper()]
        if index < len(lines):
            return lines[index].split()[0]
    except Exception:
        pass
    return ""


def dump_current_rom(index: int, vendor: str, dest_dir: str) -> str:
    """Dump the current BIOS to a file and return the path."""
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, f"gpu{index}.rom")
    try:
        if vendor == "nvidia":
            subprocess.check_call(["nvflash_linux", "-i", str(index), "--save", out_path])
        else:
            subprocess.check_call(["amdvbflash", "-i", str(index), "-s", out_path])
    except Exception:
        return ""
    return out_path


def flash_rom(index: int, vendor: str, rom_path: str, backup_dir: str = "/tmp") -> bool:
    """Flash a ROM to the GPU and restore from backup if it fails."""
    backup = dump_current_rom(index, vendor, backup_dir)
    if not backup:
        return False
    try:
        if vendor == "nvidia":
            subprocess.check_call(["nvflash_linux", "-i", str(index), "-6", rom_path])
        else:
            subprocess.check_call(["amdvbflash", "-i", str(index), "-p", "0", rom_path])
        return True
    except Exception:
        try:
            if vendor == "nvidia":
                subprocess.check_call(["nvflash_linux", "-i", str(index), backup])
            else:
                subprocess.check_call(["amdvbflash", "-i", str(index), "-p", "0", backup])
        except Exception:
            pass
        return False


def verify_flashed_rom(index: int, vendor: str, rom_path: str, tmp_dir: str = "/tmp") -> bool:
    """Return True if the flashed ROM matches the provided file."""
    verify_path = dump_current_rom(index, vendor, tmp_dir)
    if not verify_path:
        return False
    try:
        with open(verify_path, "rb") as f:
            new_data = f.read()
        with open(rom_path, "rb") as f:
            orig_data = f.read()
    except OSError:
        return False
    finally:
        try:
            os.remove(verify_path)
        except OSError:
            pass
    return hashlib.sha256(new_data).digest() == hashlib.sha256(orig_data).digest()


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
    bios = settings.get("bios_rom") or settings.get("rom_path")
    core_offset = settings.get("core_offset")
    if core_offset is not None:
        try:
            co = int(core_offset)
        except (TypeError, ValueError):
            logging.warning("Invalid core_offset value: %s", core_offset)
            return False
        if co < -200 or co > 200:
            logging.warning("Invalid core_offset value: %s", core_offset)
            return False
    mem_offset = settings.get("mem_offset")
    if mem_offset is not None:
        try:
            mo = int(mem_offset)
        except (TypeError, ValueError):
            logging.warning("Invalid mem_offset value: %s", mem_offset)
            return False
        if mo < -200 or mo > 200:
            logging.warning("Invalid mem_offset value: %s", mem_offset)
            return False
    voltage = settings.get("voltage")
    if voltage is not None:
        try:
            vol = int(voltage)
        except (TypeError, ValueError):
            logging.warning("Invalid voltage value: %s", voltage)
            return False
        if vol < 700 or vol > 1100:
            logging.warning("Invalid voltage value: %s", voltage)
            return False
    if bios:
        ok = flash_rom(int(index), vendor, bios)
        if ok:
            if not verify_flashed_rom(int(index), vendor, bios):
                logging.error("ROM verification failed for GPU %s", index)
                return False
        else:
            return False
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
    except FileNotFoundError as e:
        logging.error("Utility missing while applying settings: %s", e)
        raise
    except Exception as e:
        logging.error("Failed to apply settings: %s", e)
        raise
    return True


class GPUFlashManager(threading.Thread):
    """Background thread that handles BIOS flashing/undervolting."""

    def __init__(self, worker_id: str, server_url: str, gpus: list[dict]):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.server_url = server_url
        self.gpus = gpus
        for g in self.gpus:
            if not g.get("pci_bus"):
                try:
                    vendor = g.get("vendor", "nvidia").lower()
                    g["pci_bus"] = detect_pci_address(g.get("index", 0), vendor)
                except Exception:
                    g["pci_bus"] = ""
        self.running = True

    def process_task(self, gpu_uuid: str, settings: dict):
        gpu = next((g for g in self.gpus if g["uuid"] == gpu_uuid), None)
        if not gpu:
            return
        baseline = md5_speed()
        error_logged = False
        try:
            success = apply_flash_settings(gpu, settings)
            if not success:
                event_logger.log_error(
                    "flasher",
                    self.worker_id,
                    "W007",
                    "Settings application failed",
                )
        except FileNotFoundError as e:
            event_logger.log_error(
                "flasher",
                self.worker_id,
                "W006",
                "Flashing utility missing",
                e,
            )
            success = False
            error_logged = True
        except Exception as e:
            event_logger.log_error(
                "flasher",
                self.worker_id,
                "W007",
                str(e),
                e,
            )
            success = False
            error_logged = True
        post = md5_speed() if success else 0
        if success and post < baseline * 0.8:
            event_logger.log_error(
                "flasher",
                self.worker_id,
                "W007",
                "Performance drop after flashing",
            )
            success = False
        if not success and not error_logged:
            event_logger.log_error(
                "flasher",
                self.worker_id,
                "W007",
                "Flashing failed",
            )
        payload = {
            "worker_id": self.worker_id,
            "gpu_uuid": gpu_uuid,
            "success": success,
            "timestamp": int(time.time()),
            "signature": None,
        }
        payload["signature"] = sign_message(self.worker_id, payload["timestamp"])
        try:
            requests.post(f"{self.server_url}/flash_result", json=payload, timeout=5)
        except Exception:
            pass

    def run(self):
        while self.running:
            try:
                ts = int(time.time())
                params = {
                    "worker_id": self.worker_id,
                    "timestamp": ts,
                    "signature": sign_message(self.worker_id, ts),
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
            except Exception as e:
                event_logger.log_error(
                    "flasher",
                    self.worker_id,
                    "W007",
                    "Flash manager error",
                    e,
                )
                time.sleep(5)
