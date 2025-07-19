import os
import json
import subprocess
import time
import uuid
import redis
import requests
from pathlib import Path
import glob
import socket

from .gpu_sidecar import GPUSidecar
from .crypto_utils import load_public_key_pem
from ascii_logo import print_logo

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", "30"))

CONFIG_FILE = Path.home() / ".hashmancer" / "worker_config.json"
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        SERVER_URL = os.getenv("SERVER_URL", cfg.get("server_url", SERVER_URL))
        REDIS_HOST = os.getenv("REDIS_HOST", cfg.get("redis_host", REDIS_HOST))
        REDIS_PORT = int(os.getenv("REDIS_PORT", cfg.get("redis_port", REDIS_PORT)))
    except Exception:
        pass

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def detect_gpus() -> list[dict]:
    """Return a list of GPUs with uuid, model, pci_bus, memory_mb, pci_link_width."""
    # NVIDIA detection via nvidia-smi
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pci.bus_id,memory.total,pci.link.width.current",
                "--format=csv,noheader",
            ],
            text=True,
        )
        gpus = []
        for line in output.strip().splitlines():
            idx, uuid_str, name, bus, mem, width = [x.strip() for x in line.split(',')]
            gpus.append(
                {
                    "index": int(idx),
                    "uuid": uuid_str,
                    "model": name,
                    "pci_bus": bus,
                    "pci_width": int(width),
                    "memory_mb": int(mem.split()[0]),
                    "pci_link_width": int(width),
                }
            )
        return gpus
    except Exception:
        pass

    # AMD detection via rocm-smi
    try:
        output = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showbus", "--showuniqueid", "--showmeminfo", "vram"],
            text=True,
        )
        gpus = []
        current = {}
        for line in output.splitlines():
            if line.startswith("GPU") and "Unique ID" in line:
                if current:
                    gpus.append(current)
                    current = {}
                parts = line.split()
                idx = int(parts[0].split("[")[1].split("]")[0])
                uuid_str = parts[-1]
                current = {
                    "index": idx,
                    "uuid": uuid_str,
                    "model": "AMD GPU",
                    "pci_bus": "",
                    "pci_width": 16,
                    "memory_mb": 0,
                    "pci_link_width": 16,
                }
            elif line.startswith("GPU") and "PCI Bus" in line:
                current["pci_bus"] = line.split()[-1]
            elif line.startswith("GPU") and "VRAM Total" in line:
                current["memory_mb"] = int(line.split()[-2])
        if current:
            gpus.append(current)
        if gpus:
            return gpus
    except Exception:
        pass

    # Generic detection via lspci for Intel or unknown GPUs
    try:
        output = subprocess.check_output(["lspci"], text=True)
        gpus = []
        for line in output.splitlines():
            if "VGA compatible controller" in line or "3D controller" in line:
                bus = line.split()[0]
                model = line.split(":", 2)[-1].strip()
                gpus.append(
                    {
                        "index": len(gpus),
                        "uuid": bus,
                        "model": model,
                        "pci_bus": bus,
                        "pci_width": 16,
                        "memory_mb": 0,
                        "pci_link_width": 0,
                    }
                )
        if gpus:
            return gpus
    except Exception:
        pass

    # fallback single fake gpu so worker still runs
    return [
        {
            "index": 0,
            "uuid": str(uuid.uuid4()),
            "model": "CPU",  # placeholder
            "pci_bus": "0000:00:00.0",
            "pci_width": 16,
            "memory_mb": 0,
            "pci_link_width": 0,
        }
    ]


def get_gpu_temps() -> list[int]:
    """Return GPU temperatures if available."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        )
        return [int(t.strip()) for t in output.strip().splitlines()]
    except Exception:
        pass

    temps = []
    for path in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/temp*_input"):
        try:
            with open(path) as f:
                val = int(f.read().strip())
                temps.append(val // 1000)
        except Exception:
            continue
    return temps


def register_worker(worker_id: str, gpus: list[dict]):
    ip = socket.gethostbyname(socket.gethostname())
    ts = int(time.time())
    r.hset(
        f"worker:{worker_id}",
        mapping={"ip": ip, "status": "idle", "last_seen": ts},
    )
    r.sadd("workers", worker_id)
    for g in gpus:
        r.hset(
            f"gpu:{g['uuid']}",
            mapping={
                "model": g["model"],
                "pci_bus": g["pci_bus"],
                "memory_mb": g["memory_mb"],
                "pci_link_width": g.get("pci_link_width", 0),
                "worker": worker_id,
            },
        )
        r.sadd(f"worker:{worker_id}:gpus", g["uuid"])


def main():
    print_logo()
    worker_id = os.getenv("WORKER_ID", str(uuid.uuid4()))
    gpus = detect_gpus()
    name = register_worker(worker_id, gpus)
    threads = [GPUSidecar(name, gpu, SERVER_URL) for gpu in gpus]
    for t in threads:
        t.start()
    print(f"Worker {name} started with {len(gpus)} GPUs")
    try:
        while True:
            temps = get_gpu_temps()
            progress = {
                t.gpu.get("uuid"): t.progress for t in threads if t.current_job
            }
            try:
                requests.post(
                    f"{SERVER_URL}/worker_status",
                    json={
                        "name": name,
                        "status": "online",
                        "temps": temps,
                        "progress": progress,
                    },
                    timeout=5,
                )
            except Exception:
                pass
            time.sleep(STATUS_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping worker...")
        for t in threads:
            t.running = False
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
