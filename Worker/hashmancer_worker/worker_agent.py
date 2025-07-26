import os
import json
import subprocess
import time
import uuid
import redis
import logging

try:
    from redis.exceptions import RedisError
except Exception:  # fallback for bundled stub
    from redis import RedisError
import requests
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
from ascii_logo import print_logo
import argparse

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_SSL = os.getenv("REDIS_SSL", "0")
REDIS_SSL_CERT = os.getenv("REDIS_SSL_CERT")
REDIS_SSL_KEY = os.getenv("REDIS_SSL_KEY")
REDIS_SSL_CA_CERT = os.getenv("REDIS_SSL_CA_CERT")
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
        REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", cfg.get("redis_password", REDIS_PASSWORD))
        REDIS_SSL = os.getenv("REDIS_SSL", str(cfg.get("redis_ssl", REDIS_SSL)))
        REDIS_SSL_CERT = os.getenv("REDIS_SSL_CERT", cfg.get("redis_ssl_cert", REDIS_SSL_CERT))
        REDIS_SSL_KEY = os.getenv("REDIS_SSL_KEY", cfg.get("redis_ssl_key", REDIS_SSL_KEY))
        REDIS_SSL_CA_CERT = os.getenv("REDIS_SSL_CA_CERT", cfg.get("redis_ssl_ca_cert", REDIS_SSL_CA_CERT))
    except Exception:
        pass

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


def _redis_write(func, *args, **kwargs):
    """Perform a Redis write with retry and backoff."""
    delay = 0.5
    while True:
        try:
            return func(*args, **kwargs)
        except RedisError:
            time.sleep(delay)
            delay = min(delay * 2, 30)


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
            idx, uuid_str, name, bus, mem, width = [x.strip() for x in line.split(",")]
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
    _redis_write(
        r.hset,
        f"worker:{worker_id}",
        mapping={"ip": ip, "status": "idle", "last_seen": ts},
    )
    _redis_write(r.sadd, "workers", worker_id)
    for g in gpus:
        _redis_write(
            r.hset,
            f"gpu:{g['uuid']}",
            mapping={
                "model": g["model"],
                "pci_bus": g["pci_bus"],
                "memory_mb": g["memory_mb"],
                "pci_link_width": g.get("pci_link_width", 0),
                "worker": worker_id,
            },
        )
        _redis_write(r.sadd, f"worker:{worker_id}:gpus", g["uuid"])

    payload = {
        "worker_id": worker_id,
        "hardware": {"gpus": gpus},
        "pubkey": load_public_key_pem(),
        "timestamp": int(time.time()),
        "signature": None,
    }
    payload["signature"] = sign_message(worker_id, payload["timestamp"])

    name = None
    try:
        resp = requests.post(f"{SERVER_URL}/register_worker", json=payload, timeout=5)
        data = resp.json()
        if data.get("status") == "ok":
            name = data.get("waifu")
        else:
            name = data.get("message")
    except Exception:
        name = None

    if name:
        _redis_write(r.set, "worker_name", name)

    return name


def main(argv: list[str] | None = None):
    if argv is None:
        argv = []
    logging.basicConfig(level=logging.INFO)
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
    args = parser.parse_args(argv)

    probabilistic_order = False
    markov_lang = "english"
    inverse_prob_order = False

    print_logo()
    worker_id = os.getenv("WORKER_ID", str(uuid.uuid4()))
    gpus = detect_gpus()
    name = register_worker(worker_id, gpus)
    # benchmark GPUs once before starting normal job processing
    low_bw_engine = "hashcat"
    try:
        resp = requests.get(f"{SERVER_URL}/server_status", timeout=5)
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
    except Exception:
        probabilistic_order = args.probabilistic_order
        inverse_prob_order = args.inverse_prob_order
        pass

    for gpu in gpus:
        engine = "hashcat"
        if (
            gpu.get("pci_link_width", gpu.get("pci_width", 16)) <= 4
            and low_bw_engine == "darkling"
        ):
            engine = "darkling-engine"
            rates = run_darkling_benchmark(gpu)
        else:
            rates = run_hashcat_benchmark(gpu, engine)
        payload = {
            "worker_id": name,
            "gpu_uuid": gpu.get("uuid"),
            "engine": engine,
            "hashrates": rates,
            "timestamp": int(time.time()),
            "signature": None,
        }
        payload["signature"] = sign_message(name, payload["timestamp"])
        try:
            requests.post(f"{SERVER_URL}/submit_benchmark", json=payload, timeout=10)
        except Exception as e:
            logging.warning(
                "Failed to submit benchmark for %s: %s",
                gpu.get("uuid"),
                e,
            )

    threads = []
    for gpu in gpus:
        try:
            threads.append(
                GPUSidecar(
                    name,
                    gpu,
                    SERVER_URL,
                    probabilistic_order=probabilistic_order,
                    markov_lang=markov_lang,
                    inverse_order=inverse_prob_order,
                )
            )
        except TypeError:
            threads.append(
                GPUSidecar(
                    name,
                    gpu,
                    SERVER_URL,
                    probabilistic_order=probabilistic_order,
                    markov_lang=markov_lang,
                    inverse_order=inverse_prob_order,
                )
            )
    for t in threads:
        t.start()
    flash_mgr = GPUFlashManager(name, SERVER_URL, gpus)
    flash_mgr.start()
    logging.info("Worker %s started with %d GPUs", name, len(gpus))
    try:
        while True:
            temps = get_gpu_temps()
            progress = {t.gpu.get("uuid"): t.progress for t in threads if t.current_job}
            try:
                ts = int(time.time())
                requests.post(
                    f"{SERVER_URL}/worker_status",
                    json={
                        "name": name,
                        "status": "online",
                        "temps": temps,
                        "progress": progress,
                        "timestamp": ts,
                        "signature": sign_message(name, ts),
                    },
                    timeout=5,
                )
            except Exception:
                pass
            time.sleep(STATUS_INTERVAL)
    except KeyboardInterrupt:
        logging.info("Stopping worker...")
        for t in threads:
            t.running = False
        flash_mgr.running = False
        for t in threads:
            t.join()
        flash_mgr.join()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
