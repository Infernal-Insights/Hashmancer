import os
import json
import socket
import subprocess
import time
import uuid
import threading
import redis

from .gpu_sidecar import GPUSidecar

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
STREAM = os.getenv("JOBS_STREAM", "jobs")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def detect_gpus() -> list[dict]:
    """Return a list of GPUs with uuid, model, pci_bus, memory_mb, pci_link_width."""
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
                    "memory_mb": int(mem.split()[0]),
                    "pci_link_width": int(width),
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
                "memory_mb": 0,
                "pci_link_width": 0,
            }
        ]


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
    worker_id = os.getenv("WORKER_ID", str(uuid.uuid4()))
    gpus = detect_gpus()
    register_worker(worker_id, gpus)
    try:
        r.xgroup_create(STREAM, worker_id, id='$', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
    threads = [GPUSidecar(worker_id, gpu) for gpu in gpus]
    for t in threads:
        t.start()
    print(f"Worker {worker_id} started with {len(gpus)} GPUs")
    try:
        while True:
            r.hset(f"worker:{worker_id}", "last_seen", int(time.time()))
            time.sleep(10)
    except KeyboardInterrupt:
        print("Stopping worker...")
        for t in threads:
            t.running = False
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
