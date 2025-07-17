import os
import time
import threading
import redis
import requests
import json
import random

from .crypto_utils import sign_message

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


class GPUSidecar(threading.Thread):
    """Background thread that fetches and executes jobs via the HTTP API."""

    def __init__(self, worker_id: str, gpu: dict, server_url: str):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.gpu = gpu
        self.server_url = server_url
        self.running = True

    def run(self):
        while self.running:
            try:
                params = {
                    "worker_id": self.worker_id,
                    "signature": sign_message(self.worker_id),
                }
                resp = requests.get(
                    f"{self.server_url}/get_batch", params=params, timeout=10
                )
                data = resp.json()
                if data.get("status") == "none" or "batch_id" not in data:
                    time.sleep(5)
                    continue
                self.execute_job(data)
            except Exception as e:
                print(f"Sidecar error on {self.gpu['uuid']}: {e}")
                time.sleep(5)

    def execute_job(self, batch: dict):
        """Simulate GPU work and submit results to the server."""
        job_id = batch["batch_id"]
        r.hset(f"job:{job_id}", mapping=batch)
        if self.gpu.get("pci_width", 16) <= 4:
            r.hset(
                f"vram:{self.gpu['uuid']}:{job_id}",
                mapping={"payload": json.dumps(batch)},
            )
        print(f"GPU {self.gpu['uuid']} processing {job_id}")
        founds = self.simulate_crack(job_id)
        if founds:
            payload = {
                "worker_id": self.worker_id,
                "batch_id": job_id,
                "founds": founds,
                "signature": sign_message(json.dumps(founds)),
            }
            endpoint = "submit_founds"
        else:
            payload = {
                "worker_id": self.worker_id,
                "batch_id": job_id,
                "signature": sign_message(job_id),
            }
            endpoint = "submit_no_founds"
        try:
            requests.post(
                f"{self.server_url}/{endpoint}", json=payload, timeout=10
            )
        except Exception as e:
            print(f"Result submission failed: {e}")

    def simulate_crack(self, job_id: str) -> list[str]:
        """Return dummy results or an empty list to mimic cracking."""
        time.sleep(1)
        return [f"{job_id}:dummy"] if random.random() < 0.5 else []

