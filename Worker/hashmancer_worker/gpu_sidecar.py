import os
import time
import threading
import redis
import requests
import json
import random
import subprocess
from pathlib import Path

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
        self.current_job = None
        self.hashrate = 0.0
        self.progress = 0.0

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
        """Run hashcat for the provided batch and submit the results."""
        job_id = batch["batch_id"]
        self.current_job = job_id
        self.hashrate = 0.0
        self.progress = 0.0

        r.hset(f"job:{job_id}", mapping=batch)

        if self.gpu.get("pci_width", 16) <= 4:
            r.hset(
                f"vram:{self.gpu['uuid']}:{job_id}",
                mapping={"payload": json.dumps(batch)},
            )
            if batch.get("wordlist"):
                try:
                    with open(batch["wordlist"], "rb") as f:
                        r.set(f"vram:{self.gpu['uuid']}:{job_id}:wordlist", f.read())
                except Exception:
                    pass

        print(f"GPU {self.gpu['uuid']} processing {job_id}")
        founds = self.run_hashcat(batch)

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

        self.current_job = None

    def run_hashcat(self, batch: dict) -> list[str]:
        """Execute hashcat according to the batch parameters."""
        job_id = batch["batch_id"]
        hashes = json.loads(batch.get("hashes", "[]"))
        hash_file = Path(f"/tmp/{job_id}.hashes")
        hash_file.write_text("\n".join(hashes))

        outfile = Path(f"/tmp/{job_id}.out")
        restore = Path(f"/tmp/{job_id}.restore")

        attack = batch.get("attack_mode", "mask")
        cmd = ["hashcat", "-m", batch.get("hash_mode", "0"), str(hash_file)]

        if attack == "mask" and batch.get("mask"):
            cmd += ["-a", "3", batch["mask"]]
        elif attack == "dict" and batch.get("wordlist"):
            cmd += ["-a", "0", batch["wordlist"]]
        elif attack == "hybrid" and batch.get("wordlist") and batch.get("mask"):
            cmd += ["-a", "6", batch["wordlist"], batch["mask"]]

        cmd += [
            "--quiet",
            "--status",
            "--status-json",
            "--status-timer",
            "10",
            "--outfile",
            str(outfile),
            "--outfile-format",
            "2",
            "--restore-file",
            str(restore),
            "-d",
            str(self.gpu.get("index", 0)),
        ]

        env = os.environ.copy()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        while proc.poll() is None:
            line = proc.stdout.readline()
            if not line:
                time.sleep(1)
                continue
            try:
                status = json.loads(line.strip())
                if isinstance(status, dict):
                    speeds = status.get("speed", [0])
                    self.hashrate = float(speeds[0]) if speeds else 0.0
                    self.progress = status.get("progress", 0.0)
                    try:
                        requests.post(
                            f"{self.server_url}/submit_hashrate",
                            json={"worker_id": self.worker_id, "hashrate": self.hashrate},
                            timeout=5,
                        )
                    except Exception:
                        pass
            except json.JSONDecodeError:
                continue

        founds = []
        if outfile.is_file():
            founds = [line.strip() for line in outfile.read_text().splitlines() if line.strip()]

        if proc.returncode != 0 and restore.is_file():
            try:
                with open(restore, "rb") as f:
                    requests.post(
                        f"{self.server_url}/upload_restore",
                        files={"file": (restore.name, f)},
                        timeout=5,
                    )
            except Exception:
                pass

        try:
            hash_file.unlink(missing_ok=True)
            outfile.unlink(missing_ok=True)
            restore.unlink(missing_ok=True)
        except Exception:
            pass

        return founds

