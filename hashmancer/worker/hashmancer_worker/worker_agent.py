import os
import json
import subprocess
import time
import uuid
import redis
import logging
import hashmancer.utils.event_logger as event_logger
import shutil
from .gpu_types import GPUInfo

try:
    from redis.exceptions import RedisError
except ImportError:  # fallback for bundled stub
    from redis import RedisError
import requests
from hashmancer.utils.http_utils import post_with_retry, get_with_retry
from pathlib import Path
import glob
import socket

from .gpu_sidecar import (
    GPUSidecar,
    run_hashcat_benchmark,
    run_darkling_benchmark,
)
from .bios_flasher import GPUFlashManager
from .crypto_utils import load_public_key_pem, sign_message
from hashmancer.ascii_logo import print_logo
import argparse

# defaults used before configuration is parsed in ``main``
SERVER_URL = "http://localhost:8000"
STATUS_INTERVAL = 30
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD: str | None = None
REDIS_SSL = "0"
REDIS_SSL_CERT: str | None = None
REDIS_SSL_KEY: str | None = None
REDIS_SSL_CA_CERT: str | None = None

CONFIG_FILE = Path.home() / ".hashmancer" / "worker_config.json"
GPU_TUNING: dict[str, dict] = {}
r: redis.Redis | None = None


def _redis_write(func, *args, **kwargs):
    """Perform a Redis write with retry and limited time."""
    delay = 0.5
    total = 0.0
    while total < 60:
        try:
            return func(*args, **kwargs)
        except RedisError:
            time.sleep(delay)
            total += delay
            delay = min(delay * 2, 30)
    raise RedisError("Redis write failed after retries")


def _parse_vendor_from_lspci(line: str) -> str:
    line_u = line.upper()
    if "NVIDIA" in line_u:
        return "nvidia"
    if "AMD" in line_u or "ATI" in line_u or "ADVANCED MICRO DEVICES" in line_u:
        return "amd"
    if "INTEL" in line_u:
        return "intel"
    return "unknown"


def _detect_nvidia() -> list[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        event_logger.log_error("worker", "unassigned", "W099", "nvidia-smi not found")
        return []
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pci.bus_id,memory.total,pci.link.width.current",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except subprocess.CalledProcessError as e:
        event_logger.log_error("worker", "unassigned", "W099", "nvidia-smi failed", e)
        return []

    gpus: list[GPUInfo] = []
    for line in output.strip().splitlines():
        try:
            idx, uuid_str, name, bus, mem, width = [x.strip() for x in line.split(",")]
            gpus.append(
                GPUInfo(
                    index=int(idx),
                    uuid=uuid_str,
                    model=name,
                    pci_bus=bus,
                    memory_mb=int(mem.split()[0]),
                    pci_link_width=int(width),
                    vendor="nvidia",
                )
            )
        except (ValueError, IndexError) as e:
            event_logger.log_error(
                "worker", "unassigned", "W099", "Failed to parse nvidia-smi output", e
            )
    return gpus


def _detect_amd() -> list[GPUInfo]:
    if not shutil.which("rocm-smi"):
        event_logger.log_error("worker", "unassigned", "W099", "rocm-smi not found")
        return []
    try:
        output = subprocess.check_output(
            [
                "rocm-smi",
                "--showproductname",
                "--showbus",
                "--showuniqueid",
                "--showmeminfo",
                "vram",
            ],
            text=True,
        )
    except subprocess.CalledProcessError as e:
        event_logger.log_error("worker", "unassigned", "W099", "rocm-smi failed", e)
        return []

    gpus: list[GPUInfo] = []
    current: dict | None = None
    for line in output.splitlines():
        if line.startswith("GPU") and "Unique ID" in line:
            if current:
                gpus.append(
                    GPUInfo(
                        index=current.get("index", 0),
                        uuid=current.get("uuid", ""),
                        model="AMD GPU",
                        pci_bus=current.get("pci_bus", ""),
                        memory_mb=current.get("memory_mb", 0),
                        pci_link_width=current.get("pci_link_width", 16),
                        vendor="amd",
                    )
                )
            parts = line.split()
            idx = int(parts[0].split("[")[1].split("]")[0])
            uuid_str = parts[-1]
            current = {
                "index": idx,
                "uuid": uuid_str,
                "pci_bus": "",
                "memory_mb": 0,
                "pci_link_width": 16,
            }
        elif current and line.startswith("GPU") and "PCI Bus" in line:
            current["pci_bus"] = line.split()[-1]
        elif current and line.startswith("GPU") and "VRAM Total" in line:
            try:
                current["memory_mb"] = int(line.split()[-2])
            except (ValueError, IndexError) as e:
                event_logger.log_error(
                    "worker", "unassigned", "W099", "rocm-smi parse error", e
                )
    if current:
        gpus.append(
            GPUInfo(
                index=current.get("index", 0),
                uuid=current.get("uuid", ""),
                model="AMD GPU",
                pci_bus=current.get("pci_bus", ""),
                memory_mb=current.get("memory_mb", 0),
                pci_link_width=current.get("pci_link_width", 16),
                vendor="amd",
            )
        )
    return gpus


def _detect_sysfs() -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    try:
        cards = sorted(glob.glob("/sys/class/drm/card[0-9]*"))
    except OSError as e:
        event_logger.log_error("worker", "unassigned", "W099", "sysfs scan failed", e)
        return []

    for idx, card in enumerate(cards):
        device_path = os.path.join(card, "device")
        bus = os.path.basename(os.path.realpath(device_path))
        model = "GPU"
        vendor = "unknown"
        if shutil.which("lspci"):
            try:
                out = subprocess.check_output(["lspci", "-s", bus], text=True)
                model = out.split(":", 2)[-1].strip()
                vendor = _parse_vendor_from_lspci(out)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                event_logger.log_error(
                    "worker", "unassigned", "W099", "lspci parse error", e
                )

        mem_mb = 0
        for name in ("mem_info_vram_total", "mem_info_total", "local_memory_bytes"):
            path = os.path.join(device_path, name)
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        val = int(f.read().strip())
                    mem_mb = val // (1024 * 1024)
                    break
                except (OSError, ValueError) as e:
                    event_logger.log_error(
                        "worker", "unassigned", "W099", "sysfs memory read error", e
                    )

        width = 0
        width_path = os.path.join(device_path, "current_link_width")
        if os.path.isfile(width_path):
            try:
                with open(width_path) as f:
                    width = int(f.read().strip())
            except (OSError, ValueError) as e:
                event_logger.log_error(
                    "worker", "unassigned", "W099", "sysfs width read error", e
                )

        gpus.append(
            GPUInfo(
                index=idx,
                uuid=bus,
                model=model,
                pci_bus=bus,
                memory_mb=mem_mb,
                pci_link_width=width or 16,
                vendor=vendor,
            )
        )
    return gpus


def _detect_lspci() -> list[GPUInfo]:
    if not shutil.which("lspci"):
        event_logger.log_error("worker", "unassigned", "W099", "lspci not found")
        return []
    try:
        output = subprocess.check_output(["lspci"], text=True)
    except subprocess.CalledProcessError as e:
        event_logger.log_error("worker", "unassigned", "W099", "lspci detection failed", e)
        return []

    gpus: list[GPUInfo] = []
    for line in output.splitlines():
        if "VGA compatible controller" in line or "3D controller" in line:
            bus = line.split()[0]
            model = line.split(":", 2)[-1].strip()
            vendor = _parse_vendor_from_lspci(line)
            gpus.append(
                GPUInfo(
                    index=len(gpus),
                    uuid=bus,
                    model=model,
                    pci_bus=bus,
                    memory_mb=0,
                    pci_link_width=0,
                    vendor=vendor,
                )
            )
    return gpus


def detect_gpus() -> list[GPUInfo]:
    """Return a list of detected GPUs."""
    for detector in (_detect_nvidia, _detect_amd, _detect_sysfs, _detect_lspci):
        gpus = detector()
        if gpus:
            return gpus

    event_logger.log_error(
        "worker",
        "unassigned",
        "W000",
        "No GPUs detected. Ensure drivers are installed and accessible",
    )
    return []


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
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        event_logger.log_error("worker", "unassigned", "W100", "Temperature read failed", e)

    temps = []
    for path in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/temp*_input"):
        try:
            with open(path) as f:
                val = int(f.read().strip())
                temps.append(val // 1000)
        except (OSError, ValueError) as e:
            event_logger.log_error("worker", "unassigned", "W100", "Temperature file error", e)
            continue
    return temps


def get_gpu_power() -> list[float]:
    """Return GPU power draw in watts if available."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader",
            ],
            text=True,
        )
        return [float(p.split()[0]) for p in output.strip().splitlines()]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        event_logger.log_error("worker", "unassigned", "W101", "Power read failed", e)

    power = []
    for path in glob.glob(
        "/sys/class/drm/card*/device/hwmon/hwmon*/power*_average"
    ):
        try:
            with open(path) as f:
                val = int(f.read().strip())
                power.append(val / 1_000_000)
        except (OSError, ValueError) as e:
            event_logger.log_error("worker", "unassigned", "W101", "Power file error", e)
            continue
    if not power:
        for path in glob.glob(
            "/sys/class/drm/card*/device/hwmon/hwmon*/power*_input"
        ):
            try:
                with open(path) as f:
                    val = int(f.read().strip())
                    power.append(val / 1_000_000)
            except (OSError, ValueError) as e:
                event_logger.log_error("worker", "unassigned", "W101", "Power file error", e)
                continue
    return power


def get_gpu_utilization() -> list[int]:
    """Return GPU utilization percentage if available."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        )
        return [int(u.split()[0]) for u in output.strip().splitlines()]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        event_logger.log_error("worker", "unassigned", "W102", "Utilization read failed", e)

    util = []
    for path in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
        try:
            with open(path) as f:
                util.append(int(f.read().strip()))
        except (OSError, ValueError) as e:
            event_logger.log_error("worker", "unassigned", "W102", "Utilization file error", e)
            continue
    return util


def register_worker(worker_id: str, gpus: list[GPUInfo | dict], pin: str | None = None):
    ip = socket.gethostbyname(socket.gethostname())
    ts = int(time.time())
    _redis_write(
        r.hset,
        f"worker:{worker_id}",
        mapping={"ip": ip, "status": "idle", "last_seen": ts},
    )
    _redis_write(r.sadd, "workers", worker_id)
    normalized: list[GPUInfo] = []
    for g in gpus:
        ginfo = GPUInfo(**g) if isinstance(g, dict) else g
        _redis_write(
            r.hset,
            f"gpu:{ginfo.uuid}",
            mapping={
                "model": ginfo.model,
                "pci_bus": ginfo.pci_bus,
                "memory_mb": ginfo.memory_mb,
                "pci_link_width": ginfo.pci_link_width,
                "worker": worker_id,
            },
        )
        _redis_write(r.sadd, f"worker:{worker_id}:gpus", ginfo.uuid)
        normalized.append(ginfo)

    payload = {
        "worker_id": worker_id,
        "hardware": {"gpus": [g.__dict__ for g in normalized]},
        "pubkey": load_public_key_pem(),
        "timestamp": int(time.time()),
        "signature": None,
    }
    payload["pin"] = pin
    payload["signature"] = sign_message(worker_id, payload["timestamp"])

    name = None
    try:
        resp = post_with_retry(
            f"{SERVER_URL}/register_worker",
            json=payload,
            timeout=5,
        )
        data = resp.json()
        if data.get("status") == "ok":
            name = data.get("waifu")
        else:
            name = data.get("message")
    except requests.RequestException as e:
        event_logger.log_error("worker", worker_id, "W001", "Worker registration failed", e)
        name = None

    if name:
        _redis_write(r.set, "worker_name", name)

    return name


def perform_command(cmd: str) -> None:
    """Execute a management command such as upgrade or restart."""
    root_dir = Path(__file__).resolve().parents[2]
    if cmd == "upgrade":
        subprocess.run(
            ["python3", "setup.py", "--upgrade", "--worker"],
            cwd=root_dir,
            check=False,
        )
        subprocess.run(
            ["sudo", "systemctl", "restart", "hashmancer-worker.service"],
            check=False,
        )
    elif cmd == "restart":
        subprocess.run(
            ["sudo", "systemctl", "restart", "hashmancer-worker.service"],
            check=False,
        )


def check_worker_command(name: str) -> None:
    """Poll the server for upgrade/restart commands."""
    ts = int(time.time())
    try:
        resp = get_with_retry(
            f"{SERVER_URL}/get_worker_command",
            params={
                "worker_id": name,
                "timestamp": ts,
                "signature": sign_message(name, ts),
            },
            timeout=5,
        )
        data = resp.json()
    except requests.RequestException as e:
        event_logger.log_error("worker", name, "W002", "Command check failed", e)
        return
    if data.get("status") != "ok":
        return
    cmd = data.get("command")
    if cmd:
        perform_command(cmd)


def main(argv: list[str] | None = None):
    if argv is None:
        argv = []
    logging.basicConfig(level=logging.INFO)
    global r, SERVER_URL, STATUS_INTERVAL, REDIS_HOST, REDIS_PORT
    global REDIS_PASSWORD, REDIS_SSL, REDIS_SSL_CERT, REDIS_SSL_KEY, REDIS_SSL_CA_CERT
    global GPU_TUNING

    worker_pin = os.getenv("WORKER_PIN")

    SERVER_URL = os.getenv("SERVER_URL", SERVER_URL)
    STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", str(STATUS_INTERVAL)))
    REDIS_HOST = os.getenv("REDIS_HOST", REDIS_HOST)
    REDIS_PORT = int(os.getenv("REDIS_PORT", str(REDIS_PORT)))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", REDIS_PASSWORD)
    REDIS_SSL = os.getenv("REDIS_SSL", REDIS_SSL)
    REDIS_SSL_CERT = os.getenv("REDIS_SSL_CERT", REDIS_SSL_CERT)
    REDIS_SSL_KEY = os.getenv("REDIS_SSL_KEY", REDIS_SSL_KEY)
    REDIS_SSL_CA_CERT = os.getenv("REDIS_SSL_CA_CERT", REDIS_SSL_CA_CERT)

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            SERVER_URL = os.getenv("SERVER_URL", cfg.get("server_url", SERVER_URL))
            REDIS_HOST = os.getenv("REDIS_HOST", cfg.get("redis_host", REDIS_HOST))
            REDIS_PORT = int(os.getenv("REDIS_PORT", cfg.get("redis_port", REDIS_PORT)))
            REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", cfg.get("redis_password", REDIS_PASSWORD))
            REDIS_SSL = os.getenv("REDIS_SSL", str(cfg.get("redis_ssl", REDIS_SSL)))
            REDIS_SSL_CERT = os.getenv("REDIS_SSL_CERT", cfg.get("redis_ssl_cert", REDIS_SSL_CERT))
            REDIS_SSL_KEY = os.getenv("REDIS_SSL_KEY", cfg.get("redis_ssl_key", REDIS_SSL_KEY))
            REDIS_SSL_CA_CERT = os.getenv("REDIS_SSL_CA_CERT", cfg.get("redis_ssl_ca_cert", REDIS_SSL_CA_CERT))
            GPU_TUNING = cfg.get("darkling_tuning", {})
        except (OSError, json.JSONDecodeError) as e:
            event_logger.log_error("worker", "unassigned", "W099", "Failed to load config", e)

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
    event_logger.r = r
    parser = argparse.ArgumentParser(description="Hashmancer worker agent")
    parser.add_argument(
        "--probabilistic-order",
        action="store_true",
        help="use Markov tables for probabilistic candidate ordering",
    )
    parser.add_argument(
        "--inverse-prob-order",
        action="store_true",
        help="iterate candidates from least likely first",
    )
    parser.add_argument("--pin", help="worker registration PIN")
    args = parser.parse_args(argv)

    if args.pin:
        worker_pin = args.pin

    probabilistic_order = False
    markov_lang = "english"
    inverse_prob_order = False

    print_logo()
    worker_id = os.getenv("WORKER_ID", str(uuid.uuid4()))
    gpus = [GPUInfo(**g) if isinstance(g, dict) else g for g in detect_gpus()]
    for g in gpus:
        gdict = g.__dict__
        params = GPU_TUNING.get(gdict.get("model"), {})
        if params:
            if "grid" in params:
                gdict["darkling_grid"] = params["grid"]
            if "block" in params:
                gdict["darkling_block"] = params["block"]
    name = register_worker(worker_id, gpus, worker_pin)
    # benchmark GPUs once before starting normal job processing
    low_bw_engine = "hashcat"
    try:
        resp = get_with_retry(f"{SERVER_URL}/server_status", timeout=5)
        data = resp.json()
        low_bw_engine = data.get("low_bw_engine", "hashcat")
        if not args.probabilistic_order:
            probabilistic_order = data.get("probabilistic_order", False)
        else:
            probabilistic_order = True
        if not args.inverse_prob_order:
            inverse_prob_order = data.get("inverse_prob_order", False)
        else:
            inverse_prob_order = True
        markov_lang = data.get("markov_lang", "english")
    except (requests.RequestException, json.JSONDecodeError) as e:
        probabilistic_order = args.probabilistic_order
        inverse_prob_order = args.inverse_prob_order
        event_logger.log_error("worker", worker_id, "W001", "Failed to fetch server status", e)

    for gpu in gpus:
        engine = "hashcat"
        if (
            gpu.pci_link_width <= 4
            and low_bw_engine == "darkling"
        ):
            engine = "darkling-engine"
            rates = run_darkling_benchmark(gpu.__dict__)
        else:
            rates = run_hashcat_benchmark(gpu.__dict__, engine)
        payload = {
            "worker_id": name,
            "gpu_uuid": gpu.uuid,
            "engine": engine,
            "hashrates": rates,
            "timestamp": int(time.time()),
            "signature": None,
        }
        payload["signature"] = sign_message(name, payload["timestamp"])
        try:
            post_with_retry(
                f"{SERVER_URL}/submit_benchmark", json=payload, timeout=10
            )
        except requests.RequestException as e:
            event_logger.log_error(
                "worker",
                name,
                "W001",
                f"Failed to submit benchmark for {gpu.uuid}",
                e,
            )

    threads = []
    for gpu in gpus:
        threads.append(
            GPUSidecar(
                name,
                gpu.__dict__,
                SERVER_URL,
                probabilistic_order=probabilistic_order,
                markov_lang=markov_lang,
                inverse_order=inverse_prob_order,
            )
        )
    for t in threads:
        t.start()
    flash_mgr = GPUFlashManager(name, SERVER_URL, [g.__dict__ for g in gpus])
    flash_mgr.start()
    event_logger.log_info("worker", name, f"Worker {name} started with {len(gpus)} GPUs")
    try:
        while True:
            temps = get_gpu_temps()
            power = get_gpu_power()
            utilization = get_gpu_utilization()
            progress = {}
            for t in threads:
                if t.current_job and isinstance(t.gpu, dict) and "uuid" in t.gpu:
                    progress[t.gpu["uuid"]] = t.progress
            try:
                ts = int(time.time())
                post_with_retry(
                    f"{SERVER_URL}/worker_status",
                    json={
                        "name": name,
                        "status": "online",
                        "temps": temps,
                        "power": power,
                        "utilization": utilization,
                        "progress": progress,
                        "timestamp": ts,
                        "signature": sign_message(name, ts),
                    },
                    timeout=5,
                )
            except requests.RequestException as e:
                event_logger.log_error("worker", name, "W002", "Failed to send status", e)
            check_worker_command(name)
            time.sleep(STATUS_INTERVAL)
    except KeyboardInterrupt:
        event_logger.log_info("worker", name, "Stopping worker...")
        for t in threads:
            t.running = False
        flash_mgr.running = False
        for t in threads:
            t.join()
        flash_mgr.join()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
