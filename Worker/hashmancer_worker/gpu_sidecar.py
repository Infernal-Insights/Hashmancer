import os
import time
import threading
import redis
import requests
import json
import random
import subprocess
import base64
import gzip
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
        self.low_bw_engine = "hashcat"
        try:
            resp = requests.get(f"{self.server_url}/server_status", timeout=5)
            data = resp.json()
            self.low_bw_engine = data.get("low_bw_engine", "hashcat")
        except Exception:
            pass

    def _apply_power_limit(self, engine: str):
        """Set GPU power limit if configured via environment variables."""
        limit = None
        # allow a dedicated value when running the darkling engine
        if engine == "darkling-engine":
            limit = os.getenv("DARKLING_GPU_POWER_LIMIT")
        if not limit:
            limit = os.getenv("GPU_POWER_LIMIT")
        if not limit:
            return

        index = str(self.gpu.get("index", 0))
        commands = [
            ["nvidia-smi", "-i", index, "-pl", str(limit)],
            ["rocm-smi", "-d", index, "--setpowerlimit", str(limit)],
            [
                "intel_gpu_frequency",
                "--min",
                str(limit),
                "--max",
                str(limit),
            ],
        ]

        for cmd in commands:
            try:
                subprocess.check_call(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return
            except FileNotFoundError:
                continue
            except Exception as e:
                print(
                    f"Failed to set power limit using {' '.join(cmd)} on {self.gpu.get('uuid')}: {e}"
                )
                return

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
        batch_id = batch["batch_id"]
        job_id = batch.get("job_id", batch_id)
        self.current_job = batch_id
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

        print(f"GPU {self.gpu['uuid']} processing {batch_id}")
        if (
            self.gpu.get("pci_link_width", self.gpu.get("pci_width", 16)) <= 4
            and self.low_bw_engine == "darkling"
        ):
            founds = self.run_darkling_engine(batch)
        else:
            founds = self.run_hashcat(batch)

        if founds:
            payload = {
                "worker_id": self.worker_id,
                "batch_id": batch_id,
                "founds": founds,
                "signature": sign_message(json.dumps(founds)),
            }
            endpoint = "submit_founds"
        else:
            payload = {
                "worker_id": self.worker_id,
                "batch_id": batch_id,
                "signature": sign_message(batch_id),
            }
            endpoint = "submit_no_founds"

        try:
            requests.post(
                f"{self.server_url}/{endpoint}", json=payload, timeout=10
            )
        except Exception as e:
            print(f"Result submission failed: {e}")

        self.current_job = None

    def _run_engine(self, engine: str, batch: dict) -> list[str]:
        """Execute the given cracking engine according to the batch parameters."""
        batch_id = batch["batch_id"]
        hashes = json.loads(batch.get("hashes", "[]"))
        hash_file = Path(f"/tmp/{batch_id}.hashes")
        hash_file.write_text("\n".join(hashes))

        outfile = Path(f"/tmp/{batch_id}.out")
        restore = Path(f"/tmp/{batch_id}.restore")

        attack = batch.get("attack_mode", "mask")
        cmd = [engine, "-m", batch.get("hash_mode", "0"), str(hash_file)]

        workload = os.getenv("HASHCAT_WORKLOAD")
        if workload:
            cmd += ["-w", workload]
        if os.getenv("HASHCAT_OPTIMIZED", "0") == "1":
            cmd.append("-O")

        wordlist_path = batch.get("wordlist")
        if not wordlist_path and batch.get("wordlist_key"):
            data_b64 = r.get(f"wlcache:{batch['wordlist_key']}")
            if data_b64:
                tmp = Path(f"/tmp/{batch_id}.wl")
                tmp.write_bytes(
                    gzip.decompress(base64.b64decode(data_b64.encode()))
                )
                wordlist_path = str(tmp)

        if attack == "mask" and batch.get("mask"):
            cmd += ["-a", "3", batch["mask"]]
        elif attack == "dict" and wordlist_path:
            cmd += ["-a", "0", wordlist_path]
        elif attack == "hybrid" and wordlist_path and batch.get("mask"):
            cmd += ["-a", "6", wordlist_path, batch["mask"]]

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
        self._apply_power_limit(engine)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        def monitor():
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
                                json={
                                    "worker_id": self.worker_id,
                                    "gpu_uuid": self.gpu.get("uuid"),
                                    "hashrate": self.hashrate,
                                    "signature": sign_message(self.worker_id),
                                },
                                timeout=5,
                            )
                        except Exception:
                            pass
                except json.JSONDecodeError:
                    continue

        t = threading.Thread(target=monitor, daemon=True)
        t.start()
        while proc.poll() is None:
            time.sleep(0.1)
        t.join()

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
            if wordlist_path and wordlist_path.startswith("/tmp/") and wordlist_path.endswith(".wl"):
                Path(wordlist_path).unlink(missing_ok=True)
        except Exception:
            pass

        return founds

    def run_hashcat(self, batch: dict) -> list[str]:
        """Execute hashcat with the given batch."""
        return self._run_engine("hashcat", batch)

    def run_darkling_engine(self, batch: dict) -> list[str]:
        """Placeholder for the specialized darkling engine.

        This requires an external executable named ``darkling-engine`` to be
        installed separately. It accepts the same arguments as hashcat so the
        batch formatting is identical.
        """
        return self._run_engine("darkling-engine", batch)

