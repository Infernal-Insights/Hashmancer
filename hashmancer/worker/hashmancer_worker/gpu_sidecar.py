import os
import time
import threading
from hashmancer.utils.event_logger import log_error, log_info
from redis.exceptions import RedisError
from hashmancer.worker.redis_config import redis_from_env
from hashmancer.utils.gpu_constants import (
    MAX_HASHES,
    MAX_MASK_LEN,
    MAX_RESULT_BUFFER,
)
import requests
from hashmancer.utils.http_utils import post_with_retry, get_with_retry
import json
import subprocess
import base64
import gzip
import tempfile
from pathlib import Path

from .crypto_utils import sign_message
from darkling import statistics
from hashmancer.hash_algos import HASHCAT_ALGOS

r = redis_from_env()


def _safe_redis_call(func, *args, default=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except RedisError:
        return default


# Maximum number of digests the darkling engine can accept in a single
# launch.  This is loaded from gpu_shared_types.h along with other limits.
MAX_CHARSETS = 16


from hashmancer.darkling import gpu_helpers as gpu


class DarklingContext:
    """Per-GPU context for the darkling engine state.

    The context caches mask charsets and preloads them into device memory so
    subsequent batches can skip the upload step.  When GPU libraries are
    unavailable the allocation helpers fall back to host memory which keeps the
    logic testable without requiring hardware.
    """

    def __init__(self):
        self.charsets_json: str | None = None
        self.gpu = gpu.GPUContext()
        self.cs_buf = None
        self.len_buf = None

    def load(self, charsets: dict):
        """Upload charset tables when they differ from the cached version."""
        serialized = json.dumps(charsets, sort_keys=True)
        if serialized == self.charsets_json:
            return

        self.cleanup()
        self.charsets_json = serialized

        flat = bytearray()
        lengths: list[int] = []
        for key in sorted(charsets):
            data = charsets[key].encode()
            flat.extend(data)
            lengths.append(len(data))

        if flat:
            self.cs_buf = self.gpu.alloc(len(flat))
            self.gpu.copy_from_host(self.cs_buf, bytes(flat))
        if lengths:
            import array

            arr = array.array("I", lengths)
            self.len_buf = self.gpu.alloc(arr.itemsize * len(arr))
            self.gpu.copy_from_host(self.len_buf, arr.tobytes())

    def matches(self, charsets: dict) -> bool:
        return self.charsets_json == json.dumps(charsets, sort_keys=True)

    def cleanup(self):
        self.charsets_json = None
        if self.cs_buf is not None:
            self.gpu.free(self.cs_buf)
            self.cs_buf = None
        if self.len_buf is not None:
            self.gpu.free(self.len_buf)
            self.len_buf = None


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
            resp = get_with_retry(f"{self.server_url}/server_status", timeout=5)
            data = resp.json()
            self.low_bw_engine = data.get("low_bw_engine", "hashcat")
        except (requests.RequestException, json.JSONDecodeError) as e:
            log_error("sidecar", self.worker_id, "W003", "Failed to get server status", e)
        self.darkling_ctx = DarklingContext()
        self.autotune = os.getenv("DARKLING_AUTOTUNE")
        target = os.getenv("DARKLING_TARGET_POWER_LIMIT")
        try:
            self.target_power = float(target) if target else None
        except ValueError:
            self.target_power = None
        log_info("sidecar", self.worker_id, f"Sidecar start {self.gpu.get('uuid')}")

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
                limit = os.getenv("DARKLING_TARGET_POWER_LIMIT")
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

        timeout = float(os.getenv("GPU_POWER_TIMEOUT", "5"))
        for cmd in commands:
            try:
                subprocess.check_call(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout
                )
                return
            except FileNotFoundError:
                continue
            except subprocess.CalledProcessError as e:
                log_error(
                    "sidecar",
                    self.worker_id,
                    "W002",
                    f"Power limit command failed: {' '.join(cmd)}",
                    e,
                )
                return
            except OSError as e:
                log_error(
                    "sidecar",
                    self.worker_id,
                    "W002",
                    f"Failed to set power limit using {' '.join(cmd)} on {self.gpu.get('uuid')}",
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
                    resp = get_with_retry(
                        f"{self.server_url}/get_batch", params=params, timeout=10
                    )
                    data = resp.json()
                    if data.get("status") == "none" or "batch_id" not in data:
                        time.sleep(5)
                        continue
                    self.execute_job(data)
                except (requests.RequestException, json.JSONDecodeError) as e:
                    log_error(
                        "sidecar",
                        self.worker_id,
                        "W003",
                        f"Sidecar error on {self.gpu['uuid']}",
                        e,
                    )
                    time.sleep(5)
        finally:
            self.darkling_ctx.cleanup()
            log_info("sidecar", self.worker_id, f"Sidecar stop {self.gpu.get('uuid')}")

    def execute_job(self, batch: dict):
        """Run hashcat for the provided batch and submit the results."""
        batch_id = batch["batch_id"]
        job_id = batch.get("job_id", batch_id)
        self.current_job = batch_id
        self.hashrate = 0.0
        self.progress = 0.0

        _safe_redis_call(r.hset, f"job:{job_id}", mapping=batch)

        if self.gpu.get("pci_link_width", 16) <= 4:
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
                except OSError as e:
                    log_error("sidecar", self.worker_id, "W004", "Wordlist cache failed", e)

        log_info(
            "sidecar",
            self.worker_id,
            f"GPU {self.gpu['uuid']} processing {batch_id}",
        )
        if (
            self.gpu.get("pci_link_width", 16) <= 4
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
            post_with_retry(
                f"{self.server_url}/{endpoint}",
                json=payload,
                timeout=10,
            )
        except requests.RequestException as e:
            log_error(
                "sidecar",
                self.worker_id,
                "W004",
                "Result submission failed",
                e,
            )

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
        founds: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile("w+", delete=False, dir=tmpdir) as hf:
                hf.write("\n".join(hashes))
                hash_file = Path(hf.name)

            with tempfile.NamedTemporaryFile(delete=False, dir=tmpdir) as of:
                outfile = Path(of.name)
            with tempfile.NamedTemporaryFile(delete=False, dir=tmpdir) as rf:
                restore = Path(rf.name)

            wordlist_path = batch.get("wordlist")
            rule_path = batch.get("rules")
            job_id = batch.get("job_id", batch_id)

            if not wordlist_path or not os.path.isfile(str(wordlist_path)):
                cached = _safe_redis_call(
                    r.hget, f"vram:{self.gpu['uuid']}:{job_id}", "payload"
                )
                if cached:
                    try:
                        cached_batch = json.loads(cached)
                        for k, v in cached_batch.items():
                            batch.setdefault(k, v)
                    except json.JSONDecodeError:
                        pass
                    wordlist_path = batch.get("wordlist")

                if not wordlist_path or not os.path.isfile(str(wordlist_path)):
                    data = _safe_redis_call(
                        r.get, f"vram:{self.gpu['uuid']}:{job_id}:wordlist"
                    )
                    if data:
                        with tempfile.NamedTemporaryFile(
                            delete=False, dir=tmpdir, suffix=".wl"
                        ) as tmp:
                            if isinstance(data, str):
                                data = data.encode()
                            tmp.write(data)
                            wordlist_path = tmp.name
                            batch["wordlist"] = wordlist_path

            try:
                attack = batch.get("attack_mode", "mask")
                cmd = [engine, "-m", batch.get("hash_mode", "0"), str(hash_file)]

                workload = os.getenv("HASHCAT_WORKLOAD")
                if workload:
                    cmd += ["-w", workload]
                if os.getenv("HASHCAT_OPTIMIZED", "0") == "1":
                    cmd.append("-O")

                if not wordlist_path and batch.get("wordlist_key"):
                    data_b64 = _safe_redis_call(
                        r.get, f"wlcache:{batch['wordlist_key']}"
                    )
                    if data_b64:
                        with tempfile.NamedTemporaryFile(
                            delete=False, dir=tmpdir, suffix=".wl"
                        ) as tmp:
                            tmp.write(
                                gzip.decompress(base64.b64decode(data_b64.encode()))
                            )
                            wordlist_path = tmp.name

                mask_charsets = batch.get("mask_charsets")
                if mask_charsets and not skip_charsets:
                    try:
                        cs_map = json.loads(mask_charsets)
                    except json.JSONDecodeError:
                        cs_map = {}
                    for key, charset in sorted(cs_map.items()):
                        if key.startswith("?") and len(key) == 2 and key[1].isdigit():
                            cmd += [f"-{key[1]}", charset]

                if attack == "mask" and batch.get("mask"):
                    cmd += ["-a", "3", batch["mask"]]
                elif attack == "dict" and wordlist_path:
                    cmd += ["-a", "0", wordlist_path]
                elif attack == "dict_rules" and wordlist_path:
                    cmd += ["-a", "0", wordlist_path]
                    if rule_path:
                        if engine == "darkling-engine":
                            cmd += ["--rules", rule_path]
                        else:
                            cmd += ["-r", rule_path]
                elif attack == "ext_rules" and wordlist_path and rule_path:
                    cmd += ["-a", "0", wordlist_path, "-r", rule_path]

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
                    if rule_path:
                        env["DARKLING_RULES"] = str(rule_path)
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
                                    post_with_retry(
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
                                except requests.RequestException:
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
                            post_with_retry(
                                f"{self.server_url}/upload_restore",
                                files={"file": (os.path.basename(str(restore)), f)},
                                timeout=5,
                            )
                    except requests.RequestException:
                        pass
            finally:
                try:
                    _safe_redis_call(
                        r.delete,
                        f"vram:{self.gpu['uuid']}:{job_id}",
                        f"vram:{self.gpu['uuid']}:{job_id}:wordlist",
                    )
                except OSError:
                    pass

        return founds

    def run_hashcat(self, batch: dict) -> list[str]:
        """Execute hashcat with the given batch."""
        return self._run_engine("hashcat", batch)

    def _autotune_darkling(self, batch: dict) -> None:
        """Tune grid and block sizes to meet the target power limit."""
        sub = batch.copy()
        sub["start"] = 0
        sub["end"] = 1000

        saved_grid = self.gpu.pop("darkling_grid", None)
        saved_block = self.gpu.pop("darkling_block", None)

        self._run_engine("darkling-engine", sub, range_start=0, range_end=1000)

        from . import worker_agent  # imported here to avoid circular dependency

        idx = int(self.gpu.get("index", 0))
        power_vals = worker_agent.get_gpu_power()
        current = power_vals[idx] if idx < len(power_vals) else 0.0

        grid = saved_grid or 256
        block = saved_block or 256

        while current > self.target_power and (grid > 1 or block > 1):
            if grid > 1:
                grid = max(grid // 2, 1)
            elif block > 1:
                block = max(block // 2, 1)

            self.gpu["darkling_grid"] = grid
            self.gpu["darkling_block"] = block

            self._run_engine(
                "darkling-engine",
                sub,
                range_start=0,
                range_end=1000,
                skip_charsets=True,
            )
            power_vals = worker_agent.get_gpu_power()
            current = power_vals[idx] if idx < len(power_vals) else 0.0

        self.gpu["darkling_grid"] = grid
        self.gpu["darkling_block"] = block
        self.gpu["_darkling_tuned"] = True

    def run_darkling_engine(self, batch: dict) -> list[str]:
        """Execute the ``darkling-engine`` with caching and batch splitting.

        Supports both mask attacks and dictionary+rules mode. For mask attacks
        the engine preloads mask charsets and optionally orders candidates using
        probabilistic indexing. For dictionary+rules jobs the darkling rule
        engine is invoked without mask handling.
        """

        attack = batch.get("attack_mode", "mask")
        hashes = json.loads(batch.get("hashes", "[]"))

        if attack != "mask":
            hash_chunks = [
                hashes[i : i + MAX_HASHES] for i in range(0, len(hashes), MAX_HASHES)
            ]
            results: list[str] = []
            for chunk in hash_chunks:
                sub = batch.copy()
                sub["hashes"] = json.dumps(chunk)
                results.extend(
                    self._run_engine(
                        "darkling-engine", sub, skip_charsets=True
                    )
                )
            return results

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

        if _count_mask(batch.get("mask", "")) >= MAX_MASK_LEN:
            raise ValueError(
                f"darkling-engine supports masks <{MAX_MASK_LEN} characters"
            )

        if self.autotune and not self.gpu.get("_darkling_tuned") and self.target_power:
            try:
                self._autotune_darkling(batch)
            except Exception as e:  # pragma: no cover - tuning failures are non-fatal
                log_error("sidecar", self.worker_id, "W002", "Darkling autotune failed", e)

        mask_charsets = batch.get("mask_charsets")
        cs_map = {}
        if mask_charsets:
            try:
                cs_map = json.loads(mask_charsets)
            except json.JSONDecodeError:
                cs_map = {}

        reload_cs = not self.darkling_ctx.matches(cs_map)
        if reload_cs:
            self.darkling_ctx.load(cs_map)

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
        (HASHCAT_ALGOS.get("MD5"), "MD5"),
        (HASHCAT_ALGOS.get("SHA1"), "SHA1"),
        (HASHCAT_ALGOS.get("NTLM"), "NTLM"),
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
        if mode is None:
            results[name] = 0.0
            continue
        cmd = [engine, "--benchmark", "-m", str(mode), "-d", index]
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as e:
            log_error(
                "sidecar",
                gpu.get("uuid", ""),
                "W005",
                f"Benchmark failed for {gpu.get('uuid')} mode {mode}",
                e,
            )
            results[name] = 0.0
            continue
        except (OSError, subprocess.SubprocessError) as e:
            log_error(
                "sidecar",
                gpu.get("uuid", ""),
                "W005",
                f"Benchmark failed for {gpu.get('uuid')} mode {mode}",
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
                except (ValueError, IndexError):
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
        except (OSError, subprocess.SubprocessError) as e:
            log_error(
                "sidecar",
                gpu.get("uuid", ""),
                "W005",
                f"Benchmark failed for {gpu.get('uuid')} mode {mode}",
                e,
            )
            rate = 0.0
        results[name] = rate
    return results
