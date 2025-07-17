import os
import json
import subprocess
import time
import uuid
import redis
import requests

from .gpu_sidecar import GPUSidecar
from .crypto_utils import load_public_key_pem

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", "30"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def detect_gpus() -> list[dict]:
    """Return a list of GPUs with basic specs."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pci.bus_id,pci.link.width.max,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        )
        gpus = []
        for line in output.strip().splitlines():
            idx, uuid_str, name, bus, width, mem = [x.strip() for x in line.split(',')]
            gpus.append(
                {
                    "index": int(idx),
                    "uuid": uuid_str,
                    "model": name,
                    "pci_bus": bus,
                    "pci_width": int(width),
                    "memory_mb": int(mem.split()[0]),
                }
            )
        return gpus
    except Exception:
        # fallback single fake gpu so worker still runs
        return [
            {
                "index": 0,
                "uuid": str(uuid.uuid4()),
                "model": "CPU",  # placeholder
                "pci_bus": "0000:00:00.0",
                "pci_width": 16,
                "memory_mb": 0,
            }
        ]


def register_worker(worker_id: str, gpus: list[dict]) -> str:
    """Register with the server and return the assigned name."""
    payload = {
        "worker_id": worker_id,
        "hardware": {"gpus": gpus},
    }
    try:
        payload["pubkey"] = load_public_key_pem()
    except FileNotFoundError:
        payload["pubkey"] = ""
    resp = requests.post(f"{SERVER_URL}/register_worker", json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    name = data.get("waifu", worker_id)
    return name


def main():
    worker_id = os.getenv("WORKER_ID", str(uuid.uuid4()))
    gpus = detect_gpus()
    name = register_worker(worker_id, gpus)
    threads = [GPUSidecar(name, gpu, SERVER_URL) for gpu in gpus]
    for t in threads:
        t.start()
    print(f"Worker {name} started with {len(gpus)} GPUs")
    try:
        while True:
            requests.post(
                f"{SERVER_URL}/worker_status",
                json={"name": name, "status": "online"},
                timeout=5,
            )
            time.sleep(STATUS_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping worker...")
        for t in threads:
            t.running = False
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
