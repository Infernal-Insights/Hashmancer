import os
import time
import threading
import redis
import logging

try:
    from redis.exceptions import RedisError
except Exception:  # fallback for bundled stub
    from redis import RedisError
import requests
import json
import subprocess
import base64
import gzip
from pathlib import Path

from .crypto_utils import sign_message
from darkling import statistics
from hash_algos import HASHCAT_ALGOS

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_SSL = os.getenv("REDIS_SSL", "0")
REDIS_SSL_CERT = os.getenv("REDIS_SSL_CERT")
REDIS_SSL_KEY = os.getenv("REDIS_SSL_KEY")
REDIS_SSL_CA_CERT = os.getenv("REDIS_SSL_CA_CERT")

redis_opts: dict[str, str | int | bool] = {
    "host": REDIS_HOST,
    "port": REDIS_PORT,
    "decode_responses": True,
}
if REDIS_PASSWORD:
    redis_opts["password"] = REDIS_PASSWORD
if str(REDIS_SSL).lower() in {"1", "true", "yes"}:
    redis_opts["ssl"] = True
    if REDIS_SSL_CA_CERT:
        redis_opts["ssl_ca_certs"] = REDIS_SSL_CA_CERT
    if REDIS_SSL_CERT:
        redis_opts["ssl_certfile"] = REDIS_SSL_CERT
    if REDIS_SSL_KEY:
        redis_opts["ssl_keyfile"] = REDIS_SSL_KEY

r = redis.Redis(**redis_opts)


def _safe_redis_call(func, *args, default=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except RedisError:
        return default


# Maximum number of digests the darkling engine can accept in a single
# launch.  This must match the constant defined in the CUDA/CL sources.
MAX_HASHES = 2048


class DarklingContext:
    """Per-GPU context keeping the darkling engine state."""

    def __init__(self):
        self.charsets_json: str | None = None

    def load(self, charsets: dict):
        """Record the loaded charsets. In a real setup this would preload them
        on the GPU."""
        self.charsets_json = json.dumps(charsets, sort_keys=True)

    def matches(self, charsets: dict) -> bool:
        return self.charsets_json == json.dumps(charsets, sort_keys=True)

    def cleanup(self):
        self.charsets_json = None


class GPUSidecar(threading.Thread):
    """Background thread that fetches and executes jobs via the HTTP API."""

    def __init__(
        self,
        worker_id: str,
        gpu: dict,
        server_url: str,
        probabilistic_order: bool = False,
        markov_lang: str = "english",
        inverse_order: bool = False,
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.gpu = gpu
        self.server_url = server_url
        self.probabilistic_order = probabilistic_order
        self.markov_lang = markov_lang
        self.inverse_order = inverse_order
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
        self.darkling_ctx = DarklingContext()

    def stop(self):
        """Signal the sidecar thread to terminate."""
        self.running = False

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

        # value specified as percent for AMD GPUs
        if isinstance(limit, str) and "%" in limit:
            value = limit.strip().rstrip("%")
            sign = ""
            if value.startswith(("+", "-")):
                sign, value = value[0], value[1:]
            commands = [["rocm-smi", "-d", index, "--setpoweroverdrive", sign + value]]
        else:
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
                logging.warning(
                    "Failed to set power limit using %s on %s: %s",
                    " ".join(cmd),
                    self.gpu.get("uuid"),
                    e,
                )
                return

    def run(self):
        try:
            while self.running:
                try:
                    ts = int(time.time())
                    params = {
                        "worker_id": self.worker_id,
                        "timestamp": ts,
                        "signature": sign_message(self.worker_id, ts),
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
                    logging.warning("Sidecar error on %s: %s", self.gpu['uuid'], e)
                    time.sleep(5)
        finally:
            self.darkling_ctx.cleanup()

    def execute_job(self, batch: dict):
        """Run hashcat for the provided batch and submit the results."""
        batch_id = batch["batch_id"]
        job_id = batch.get("job_id", batch_id)
        self.current_job = batch_id
        self.hashrate = 0.0
        self.progress = 0.0

        _safe_redis_call(r.hset, f"job:{job_id}", mapping=batch)

        if self.gpu.get("pci_width", 16) <= 4:
            _safe_redis_call(
                r.hset,
                f"vram:{self.gpu['uuid']}:{job_id}",
                mapping={"payload": json.dumps(batch)},
            )
            if batch.get("wordlist"):
                try:
                    with open(batch["wordlist"], "rb") as f:
                        _safe_redis_call(
                            r.set,
                            f"vram:{self.gpu['uuid']}:{job_id}:wordlist",
                            f.read(),
                        )
                except Exception:
                    pass

        logging.info("GPU %s processing %s", self.gpu['uuid'], batch_id)
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
                "job_id": job_id,
                "msg_id": batch.get("msg_id"),
                "founds": founds,
                "timestamp": int(time.time()),
                "signature": None,
            }
            payload["signature"] = sign_message(json.dumps(founds), payload["timestamp"])
            endpoint = "submit_founds"
        else:
            payload = {
                "worker_id": self.worker_id,
                "batch_id": batch_id,
                "job_id": job_id,
                "msg_id": batch.get("msg_id"),
                "timestamp": int(time.time()),
                "signature": None,
            }
            payload["signature"] = sign_message(batch_id, payload["timestamp"])
            endpoint = "submit_no_founds"

        try:
            requests.post(f"{self.server_url}/{endpoint}", json=payload, timeout=10)
        except Exception as e:
            logging.warning("Result submission failed: %s", e)

        self.current_job = None

    def _run_engine(
        self,
        engine: str,
        batch: dict,
        range_start: int | None = None,
        range_end: int | None = None,
        skip_charsets: bool = False,
    ) -> list[str]:
        """Execute the given cracking engine according to the batch parameters."""
        batch_id = batch["batch_id"]
        hashes = json.loads(batch.get("hashes", "[]"))
        hash_file = Path(f"/tmp/{batch_id}.hashes")
        hash_file.write_text("\n".join(hashes))

        outfile = Path(f"/tmp/{batch_id}.out")
        restore = Path(f"/tmp/{batch_id}.restore")

        wordlist_path = batch.get("wordlist")

        founds: list[str] = []
        try:
            attack = batch.get("attack_mode", "mask")
            cmd = [engine, "-m", batch.get("hash_mode", "0"), str(hash_file)]

            workload = os.getenv("HASHCAT_WORKLOAD")
            if workload:
                cmd += ["-w", workload]
            if os.getenv("HASHCAT_OPTIMIZED", "0") == "1":
                cmd.append("-O")

            if not wordlist_path and batch.get("wordlist_key"):
                data_b64 = _safe_redis_call(r.get, f"wlcache:{batch['wordlist_key']}")
                if data_b64:
                    tmp = Path(f"/tmp/{batch_id}.wl")
                    tmp.write_bytes(gzip.decompress(base64.b64decode(data_b64.encode())))
                    wordlist_path = str(tmp)

            mask_charsets = batch.get("mask_charsets")
            if mask_charsets and not skip_charsets:
                try:
                    cs_map = json.loads(mask_charsets)
                except Exception:
                    cs_map = {}
                for key, charset in sorted(cs_map.items()):
                    if key.startswith("?") and len(key) == 2 and key[1].isdigit():
                        cmd += [f"-{key[1]}", charset]

            if attack == "mask" and batch.get("mask"):
                cmd += ["-a", "3", batch["mask"]]
            elif attack == "dict" and wordlist_path:
                cmd += ["-a", "0", wordlist_path]
            elif attack == "hybrid" and wordlist_path and batch.get("mask"):
                cmd += ["-a", "6", wordlist_path, batch["mask"]]

            if engine == "darkling-engine":
                if range_start is not None:
                    cmd += ["--start", str(range_start)]
                if range_end is not None:
                    cmd += ["--end", str(range_end)]

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
            if engine == "darkling-engine":
                grid = self.gpu.get("darkling_grid")
                block = self.gpu.get("darkling_block")
                if grid:
                    env["DARKLING_GRID"] = str(grid)
                if block:
                    env["DARKLING_BLOCK"] = str(block)
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
                                ts = int(time.time())
                                requests.post(
                                    f"{self.server_url}/submit_hashrate",
                                    json={
                                        "worker_id": self.worker_id,
                                        "gpu_uuid": self.gpu.get("uuid"),
                                        "hashrate": self.hashrate,
                                        "timestamp": ts,
                                        "signature": sign_message(self.worker_id, ts),
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
                founds = [
                    line.strip()
                    for line in outfile.read_text().splitlines()
                    if line.strip()
                ]

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
        finally:
            try:
                hash_file.unlink(missing_ok=True)
                outfile.unlink(missing_ok=True)
                restore.unlink(missing_ok=True)
                if (
                    wordlist_path
                    and wordlist_path.startswith("/tmp/")
                    and wordlist_path.endswith(".wl")
                ):
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

        def _count_mask(mask: str) -> int:
            count = 0
            i = 0
            while i < len(mask):
                if mask[i] == "?" and i + 1 < len(mask):
                    i += 2
                else:
                    i += 1
                count += 1
            return count

        if _count_mask(batch.get("mask", "")) >= 56:
            raise ValueError("darkling-engine supports masks <56 characters")

        mask_charsets = batch.get("mask_charsets")
        cs_map = {}
        if mask_charsets:
            try:
                cs_map = json.loads(mask_charsets)
            except Exception:
                cs_map = {}

        reload_cs = not self.darkling_ctx.matches(cs_map)
        if reload_cs:
            self.darkling_ctx.load(cs_map)

        hashes = json.loads(batch.get("hashes", "[]"))

        hash_chunks = [
            hashes[i : i + MAX_HASHES] for i in range(0, len(hashes), MAX_HASHES)
        ]

        results: list[str] = []
        first = True

        indices: list[int] | None = None
        if self.probabilistic_order:
            markov = statistics.load_markov(lang=self.markov_lang)
            indices = statistics.probability_index_order(
                batch.get("mask", ""), cs_map, markov, inverse=self.inverse_order
            )

        for chunk in hash_chunks:
            sub = batch.copy()
            sub["hashes"] = json.dumps(chunk)
            skip = not reload_cs if first else True

            if indices is not None:
                for idx in indices:
                    results.extend(
                        self._run_engine(
                            "darkling-engine",
                            sub,
                            range_start=idx,
                            range_end=idx + 1,
                            skip_charsets=skip,
                        )
                    )
                    skip = True
            else:
                results.extend(
                    self._run_engine(
                        "darkling-engine",
                        sub,
                        range_start=batch.get("start"),
                        range_end=batch.get("end"),
                        skip_charsets=skip,
                    )
                )

            first = False

        return results


def run_hashcat_benchmark(gpu: dict, engine: str = "hashcat") -> dict[str, float]:
    """Run a short benchmark for the given GPU and return hashrates.

    Parameters
    ----------
    gpu : dict
        GPU descriptor with at least an ``index`` key.
    engine : str, optional
        Executable to run, defaults to ``hashcat``. ``darkling-engine`` can be
        used if available.

    Returns
    -------
    dict[str, float]
        Mapping of algorithm name (MD5, SHA1, NTLM) to hashrate in H/s.
    """

    modes = [
        (HASHCAT_ALGOS.get("MD5", 0), "MD5"),
        (HASHCAT_ALGOS.get("SHA1", 100), "SHA1"),
        (HASHCAT_ALGOS.get("NTLM", 1000), "NTLM"),
    ]
    index = str(gpu.get("index", 0))
    results: dict[str, float] = {}

    unit_map = {
        "H/s": 1,
        "kH/s": 1e3,
        "KH/s": 1e3,
        "MH/s": 1e6,
        "GH/s": 1e9,
        "TH/s": 1e12,
        "PH/s": 1e15,
    }

    for mode, name in modes:
        cmd = [engine, "--benchmark", "-m", str(mode), "-d", index]
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            logging.warning(
                "Benchmark failed for %s mode %s: %s",
                gpu.get("uuid"),
                mode,
                e,
            )
            results[name] = 0.0
            continue

        rate = 0.0
        for line in output.splitlines():
            if line.strip().startswith("Speed.#"):
                try:
                    parts = line.split(":", 1)[1].strip().split()
                    value = float(parts[0])
                    unit = parts[1] if len(parts) > 1 else "H/s"
                    rate = value * unit_map.get(unit, 1)
                except Exception:
                    rate = 0.0
                break

        results[name] = rate

    return results


def run_darkling_benchmark(gpu: dict) -> dict[str, float]:
    """Run a short benchmark using the darkling engine."""
    modes = [
        (HASHCAT_ALGOS.get("MD5", 0), "MD5"),
        (HASHCAT_ALGOS.get("SHA1", 100), "SHA1"),
        (HASHCAT_ALGOS.get("NTLM", 1000), "NTLM"),
    ]
    index = str(gpu.get("index", 0))
    mask = "?a?a?a?a?a?a?a?a"
    results: dict[str, float] = {}
    for mode, name in modes:
        cmd = [
            "darkling-engine",
            "-m",
            str(mode),
            "-a",
            "3",
            mask,
            "--start",
            "0",
            "--end",
            "10000",
            "--quiet",
            "--status",
            "--status-json",
            "--status-timer",
            "1",
            "-d",
            index,
        ]
        rate = 0.0
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while proc.poll() is None:
                line = proc.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                try:
                    status = json.loads(line.strip())
                    speeds = status.get("speed", [])
                    if speeds:
                        rate = float(speeds[0])
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logging.warning(
                "Benchmark failed for %s mode %s: %s",
                gpu.get("uuid"),
                mode,
                e,
            )
            rate = 0.0
        results[name] = rate
    return results
