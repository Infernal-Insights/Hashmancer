from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import subprocess
from datetime import datetime
import os
import redis
from redis_utils import get_redis
import json
import uuid
import socket
import asyncio
import glob
import sys
import redis_manager
import orchestrator_agent
from event_logger import log_error, log_info
from pathlib import Path
import learn_trends

from waifus import assign_waifu
from auth_utils import verify_signature, verify_signature_with_key
from pydantic import BaseModel
from ascii_logo import print_logo
from pattern_to_mask import get_top_masks

app = FastAPI()

r = get_redis()

JOB_STREAM = os.getenv("JOB_STREAM", "jobs")
HTTP_GROUP = os.getenv("HTTP_GROUP", "http-workers")
LOW_BW_JOB_STREAM = os.getenv("LOW_BW_JOB_STREAM", "darkling-jobs")
LOW_BW_GROUP = os.getenv("LOW_BW_GROUP", "darkling-workers")

CONFIG_FILE = Path.home() / ".hashmancer" / "server_config.json"
try:
    with open(CONFIG_FILE) as cfg:
        CONFIG = json.load(cfg)
except Exception:
    CONFIG = {}

WORDLISTS_DIR = Path(CONFIG.get("wordlists_dir", "/opt/hashmancer/wordlists"))
MASKS_DIR = Path(CONFIG.get("masks_dir", "/opt/hashmancer/masks"))
RULES_DIR = Path(CONFIG.get("rules_dir", "/opt/hashmancer/rules"))
RESTORE_DIR = Path(CONFIG.get("restore_dir", "/opt/hashmancer/restores"))

# API key used to protect the portal and legacy dashboard pages. When not set
# these routes remain publicly accessible.
PORTAL_KEY = CONFIG.get("portal_key")

# select which cracking engine low bandwidth workers should use. The
# specialized option is called "darkling".
LOW_BW_ENGINE = CONFIG.get("low_bw_engine", "hashcat")

# broadcast settings
BROADCAST_ENABLED = bool(CONFIG.get("broadcast_enabled", True))
BROADCAST_PORT = int(CONFIG.get("broadcast_port", 50000))
BROADCAST_INTERVAL = int(CONFIG.get("broadcast_interval", 30))

# hashes.com integration settings
HASHES_POLL_INTERVAL = int(CONFIG.get("hashes_poll_interval", 1800))
HASHES_ALGORITHMS = [a.lower() for a in CONFIG.get("hashes_algorithms", [])]

# Markov and candidate ordering settings
PROBABILISTIC_ORDER = bool(CONFIG.get("probabilistic_order", False))
MARKOV_LANG = CONFIG.get("markov_lang", "english")


def save_config():
    """Persist the CONFIG dictionary to disk."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(CONFIG, f, indent=2)
    except Exception as e:
        log_error("server", "system", "S740", "Failed to save config", e)

# baseline undervolt/flash settings per GPU model
FLASH_PRESETS_FILE = Path(__file__).with_name("flash_presets.json")
if FLASH_PRESETS_FILE.exists():
    try:
        with FLASH_PRESETS_FILE.open() as f:
            FLASH_PRESETS = json.load(f)
    except Exception:
        FLASH_PRESETS = {}
else:
    FLASH_PRESETS = {
        "nvidia": {
            "rtx 3080": {"power_limit": 220, "core_clock": 1350, "mem_clock": 4500},
            "rtx 3070": {"power_limit": 180, "core_clock": 1200, "mem_clock": 4000},
            "gtx 1050": {"power_limit": 60, "core_clock": 1350, "mem_clock": 3500},
            "gtx 1050 ti": {"power_limit": 70, "core_clock": 1400, "mem_clock": 3500},
            "gtx 1060": {"power_limit": 90, "core_clock": 1500, "mem_clock": 4000},
            "gtx 1070": {"power_limit": 110, "core_clock": 1600, "mem_clock": 4200},
            "gtx 1080": {"power_limit": 140, "core_clock": 1700, "mem_clock": 4800},
            "gtx 1080 ti": {"power_limit": 180, "core_clock": 1700, "mem_clock": 5000},
            "gtx 2080": {"power_limit": 170, "core_clock": 1800, "mem_clock": 6500},
            "gtx 2080 ti": {"power_limit": 200, "core_clock": 1800, "mem_clock": 7000},
            "default": {"power_limit": 150, "core_clock": 1100, "mem_clock": 3000},
        },
        "amd": {
            "rx 6800": {"power_limit": 170, "core_clock": 1200, "mem_clock": 2100},
            "rx 6700": {"power_limit": 150, "core_clock": 1100, "mem_clock": 1900},
            "rx 470": {"power_limit": 110, "core_clock": 1200, "mem_clock": 1750},
            "rx 480": {"power_limit": 120, "core_clock": 1250, "mem_clock": 1750},
            "rx 570": {"power_limit": 110, "core_clock": 1200, "mem_clock": 1750},
            "rx 580": {"power_limit": 125, "core_clock": 1250, "mem_clock": 2000},
            "default": {"power_limit": 120, "core_clock": 1000, "mem_clock": 1800},
        },
    }


def get_flash_settings(model: str) -> dict:
    name = model.lower()
    vendor = "nvidia"
    if "amd" in name or "radeon" in name or "rx" in name:
        vendor = "amd"
    presets = FLASH_PRESETS[vendor]
    for key, val in presets.items():
        if key != "default" and key in name:
            data = dict(val)
            data["vendor"] = vendor
            return data
    data = dict(presets["default"])
    data["vendor"] = vendor
    return data


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PortalAuthMiddleware:
    """Simple ASGI middleware enforcing an API key for portal routes."""

    def __init__(self, app, key: str | None):
        self.app = app
        self.key = key

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and self.key:
            path = scope.get("path", "")
            if (
                path.startswith("/portal")
                or path.startswith("/glyph")
                or path.startswith("/admin")
            ):
                headers = {
                    k.decode().lower(): v.decode() for k, v in scope.get("headers", [])
                }
                if headers.get("x-api-key") != self.key:
                    response = HTMLResponse("Unauthorized", status_code=401)
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


app.add_middleware(PortalAuthMiddleware, key=PORTAL_KEY)


class RegisterWorkerRequest(BaseModel):
    worker_id: str
    signature: str
    pubkey: str
    mode: str = "eco"
    provider: str = "on-prem"
    hardware: dict = {}


class WorkerStatusRequest(BaseModel):
    name: str
    status: str
    signature: str
    temps: list[int] | None = None
    progress: dict | None = None


class SubmitHashrateRequest(BaseModel):
    worker_id: str
    gpu_uuid: str | None = None
    hashrate: float
    signature: str


class SubmitBenchmarkRequest(BaseModel):
    worker_id: str
    gpu_uuid: str
    engine: str
    hashrates: dict[str, float]
    signature: str


async def broadcast_presence():
    """Periodically broadcast the server URL over UDP."""
    base = CONFIG.get("server_url", "http://localhost")
    port = CONFIG.get("server_port", 8000)
    url = f"{base}:{port}"
    payload = json.dumps({"server_url": url}).encode()
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                s.sendto(payload, ("255.255.255.255", BROADCAST_PORT))
        except Exception as e:
            log_error("server", "system", "S716", "Broadcast failed", e)
        await asyncio.sleep(BROADCAST_INTERVAL)


async def fetch_and_store_jobs():
    """Fetch jobs from hashes.com and store filtered results in Redis."""
    try:
        from hashescom_client import fetch_jobs

        jobs = fetch_jobs()
        for job in jobs:
            algo = str(job.get("algorithmName", "")).lower()
            if HASHES_ALGORITHMS and algo not in HASHES_ALGORITHMS:
                continue
            if job.get("currency") != "BTC":
                continue
            try:
                price = float(job.get("pricePerHash", 0))
            except (TypeError, ValueError):
                price = 0.0
            if price <= 0:
                continue
            job_id = job.get("id")
            if job_id is not None:
                r.hset(f"hashes_job:{job_id}", mapping=job)
    except Exception as e:
        log_error("server", "system", "S741", "Hashes.com fetch failed", e)


async def poll_hashes_jobs():
    """Background loop to periodically poll hashes.com for jobs."""
    while True:
        await fetch_and_store_jobs()
        await asyncio.sleep(HASHES_POLL_INTERVAL)


async def process_hashes_jobs():
    """Queue batches for jobs fetched from hashes.com."""
    while True:
        try:
            for key in r.scan_iter("hashes_job:*"):
                job = r.hgetall(key)
                if job.get("status") == "processed":
                    continue

                try:
                    hashes = json.loads(job.get("hashes", "[]"))
                except Exception:
                    hashes = []

                mask = job.get("mask", "")
                wordlist = job.get("wordlist", "")

                batch_id = redis_manager.store_batch(hashes, mask=mask, wordlist=wordlist)
                if batch_id:
                    r.hset(key, mapping={"status": "processed", "batch_id": batch_id})
        except redis.exceptions.RedisError as e:
            log_error("server", "system", "SRED", "Redis unavailable", e)
        except Exception as e:
            log_error("server", "system", "S742", "Failed to process hashes jobs", e)

        await asyncio.sleep(30)


async def dispatch_loop():
    """Periodically dispatch queued batches to workers."""
    while True:
        try:
            orchestrator_agent.dispatch_batches()
        except redis.exceptions.RedisError as e:
            log_error("server", "system", "SRED", "Redis unavailable", e)
        except Exception as e:
            log_error("server", "system", "S743", "Dispatch loop failed", e)
        await asyncio.sleep(5)


@app.on_event("startup")
async def start_broadcast():
    print_logo()
    if BROADCAST_ENABLED:
        asyncio.create_task(broadcast_presence())
    asyncio.create_task(poll_hashes_jobs())
    asyncio.create_task(process_hashes_jobs())
    asyncio.create_task(dispatch_loop())


@app.post("/register_worker")
async def register_worker(info: RegisterWorkerRequest):
    try:
        worker_id = info.worker_id
        if not verify_signature_with_key(info.pubkey, worker_id, info.signature):
            raise HTTPException(status_code=401, detail="unauthorized")

        specs = {
            "mode": info.mode,
            "provider": info.provider,
            "hardware": info.hardware,
        }
        pubkey = info.pubkey
        waifu_name = assign_waifu(r.smembers("waifu:names"))

        r.sadd("waifu:names", waifu_name)
        r.hset(
            f"worker:{waifu_name}",
            mapping={
                "id": worker_id,
                "specs": json.dumps(specs),
                "pubkey": pubkey,
                "last_seen": int(datetime.utcnow().timestamp()),
                "status": "idle",
                "low_bw_engine": LOW_BW_ENGINE,
            },
        )
        # queue GPUs for flashing
        gpus = info.hardware.get("gpus", []) if isinstance(info.hardware, dict) else []
        for g in gpus:
            settings = get_flash_settings(g.get("model", ""))
            data = json.dumps(
                {
                    "gpu_uuid": g.get("uuid"),
                    "index": g.get("index", 0),
                    "settings": settings,
                }
            )
            r.rpush(f"flash:{waifu_name}", data)
            r.hset(f"flash:settings:{g.get('uuid')}", mapping=settings)
            r.hset(f"gpu:{g.get('uuid')}", mapping={"crashes": 0})
        return {"status": "ok", "waifu": waifu_name}
    except redis.exceptions.RedisError as e:
        log_error("server", "unassigned", "SRED", "Redis unavailable", e)
        raise HTTPException(status_code=500, detail="redis unavailable")
    except Exception as e:
        log_error("server", "unassigned", "S700", "Worker registration failed", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_batch")
async def get_batch(worker_id: str, signature: str):
    try:
        if not verify_signature(worker_id, worker_id, signature):
            return {"status": "unauthorized"}

        info = r.hgetall(f"worker:{worker_id}")
        stream = JOB_STREAM
        group = HTTP_GROUP
        if info.get("low_bw_engine") == "darkling":
            stream = LOW_BW_JOB_STREAM
            group = LOW_BW_GROUP

        try:
            r.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        messages = r.xreadgroup(group, worker_id, {stream: ">"}, count=1, block=1000)
        if not messages:
            return {"status": "none"}

        msg_id = None
        job_id = None
        for _stream, entries in messages:
            for mid, data in entries:
                msg_id = mid
                job_id = data.get("job_id")
                break

        if not job_id:
            if msg_id:
                r.xack(stream, group, msg_id)
            return {"status": "none"}

        batch = r.hgetall(f"job:{job_id}")
        batch_id = batch.get("batch_id", job_id)
        batch["batch_id"] = batch_id
        batch["job_id"] = job_id
        batch["msg_id"] = msg_id
        batch["stream"] = stream
        r.hset(f"job:{job_id}", mapping={"msg_id": msg_id, "stream": stream})
        r.hset(
            f"worker:{worker_id}",
            mapping={
                "last_batch": batch_id,
                "last_seen": int(datetime.utcnow().timestamp()),
                "status": "processing",
            },
        )
        return batch
    except redis.exceptions.RedisError as e:
        log_error("server", worker_id, "SRED", "Redis unavailable", e)
        return {"status": "error", "message": "redis unavailable"}
    except Exception as e:
        log_error("server", worker_id, "S002", "Failed to assign batch", e)
        return {"status": "error"}


@app.post("/submit_founds")
async def submit_founds(payload: dict):
    try:
        if not verify_signature(
            payload["worker_id"], json.dumps(payload["founds"]), payload["signature"]
        ):
            return {"status": "unauthorized"}

        for line in payload["founds"]:
            r.rpush("found:results", f"{payload['batch_id']}:{line}")

        job_id = payload.get("job_id", payload.get("batch_id"))
        info = r.hgetall(f"job:{job_id}")
        msg_id = payload.get("msg_id") or info.get("msg_id")
        stream = info.get("stream", JOB_STREAM)
        group = HTTP_GROUP if stream == JOB_STREAM else LOW_BW_GROUP
        if msg_id:
            r.xack(stream, group, msg_id)

        r.hset(f"worker:{payload['worker_id']}", "status", "idle")
        return {"status": "ok", "received": len(payload["founds"])}
    except redis.exceptions.RedisError as e:
        log_error(
            "server", payload.get("worker_id", "system"), "SRED", "Redis unavailable", e
        )
        return {"status": "error", "message": "redis unavailable"}
    except Exception as e:
        log_error("server", payload["worker_id"], "S003", "Failed to accept founds", e)
        return {"status": "error"}


@app.post("/submit_no_founds")
async def submit_no_founds(payload: dict):
    try:
        if not verify_signature(
            payload["worker_id"], payload["batch_id"], payload["signature"]
        ):
            return {"status": "unauthorized"}

        r.rpush("found:none", payload["batch_id"])

        job_id = payload.get("job_id", payload.get("batch_id"))
        info = r.hgetall(f"job:{job_id}")
        msg_id = payload.get("msg_id") or info.get("msg_id")
        stream = info.get("stream", JOB_STREAM)
        group = HTTP_GROUP if stream == JOB_STREAM else LOW_BW_GROUP
        if msg_id:
            r.xack(stream, group, msg_id)

        r.hset(f"worker:{payload['worker_id']}", "status", "idle")
        return {"status": "ok"}
    except redis.exceptions.RedisError as e:
        log_error(
            "server", payload.get("worker_id", "system"), "SRED", "Redis unavailable", e
        )
        return {"status": "error", "message": "redis unavailable"}
    except Exception as e:
        log_error(
            "server", payload["worker_id"], "S004", "Failed to record empty result", e
        )
        return {"status": "error"}


@app.get("/wordlists")
async def list_wordlists():
    try:
        return [f.name for f in WORDLISTS_DIR.iterdir() if f.is_file()]
    except Exception as e:
        log_error("server", "system", "S705", "Failed to list wordlists", e)
        return []


@app.get("/masks")
async def list_masks():
    try:
        return [f.name for f in MASKS_DIR.iterdir() if f.is_file()]
    except Exception as e:
        log_error("server", "system", "S706", "Failed to list masks", e)
        return []


@app.get("/top_masks")
async def export_top_masks(limit: int = 10):
    """Return the top password masks derived from stored patterns."""
    try:
        return get_top_masks(limit)
    except Exception as e:
        log_error("server", "system", "S736", "Failed to export masks", e)
        return []


@app.get("/rules")
async def list_rules():
    try:
        return [f.name for f in RULES_DIR.iterdir() if f.is_file()]
    except Exception as e:
        log_error("server", "system", "S707", "Failed to list rules", e)
        return []


def get_gpu_temps():
    """Return a list of GPU temperatures across vendors if available."""
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


@app.get("/server_status")
async def server_status():
    """Return basic server and GPU metrics."""
    try:
        status = {
            "worker_count": r.scard("waifu:names"),
            "queue_length": r.llen("batch:queue"),
            "found_results": r.llen("found:results"),
            "gpu_temps": get_gpu_temps(),
            "low_bw_engine": LOW_BW_ENGINE,
            "probabilistic_order": PROBABILISTIC_ORDER,
            "markov_lang": MARKOV_LANG,
        }
        return status
    except redis.exceptions.RedisError as e:
        log_error("server", "system", "SRED", "Redis unavailable", e)
        return {"status": "error", "message": "redis unavailable"}
    except Exception as e:
        log_error("server", "system", "S710", "Failed to fetch status", e)
        return {"error": str(e)}


@app.get("/glyph", response_class=HTMLResponse)
async def glyph_page():
    """Serve the Glyph dashboard page."""
    try:
        html_path = Path(__file__).parent / "glyph.html"
        return html_path.read_text()
    except Exception as e:
        log_error("server", "system", "S711", "Failed to load Glyph dashboard", e)
        return HTMLResponse("<h1>Dashboard not available</h1>", status_code=500)


@app.get("/workers")
async def list_workers():
    """Return basic info about all registered workers."""
    workers = []
    try:
        for key in r.scan_iter("worker:*"):
            name = key.split(":", 1)[1]
            info = r.hgetall(key)
            workers.append(
                {
                    "name": name,
                    "status": info.get("status", "unknown"),
                    "last_seen": int(info.get("last_seen", 0)),
                }
            )
        return workers
    except redis.exceptions.RedisError as e:
        log_error("server", "system", "SRED", "Redis unavailable", e)
        return {"status": "error", "message": "redis unavailable"}
    except Exception as e:
        log_error("server", "system", "S712", "Failed to list workers", e)
        return []


@app.post("/worker_status")
async def set_worker_status(data: WorkerStatusRequest):
    """Update a worker's status string."""
    name = data.name
    status = data.status
    signature = data.signature
    if not name or status is None:
        raise HTTPException(status_code=400, detail="name and status required")
    if not verify_signature(name, name, signature):
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        r.hset(f"worker:{name}", "status", status)
        return {"status": "ok"}
    except redis.exceptions.RedisError as e:
        log_error("server", name or "system", "SRED", "Redis unavailable", e)
        raise HTTPException(status_code=500, detail="redis unavailable")
    except Exception as e:
        log_error("server", name, "S713", "Failed to set status", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit_hashrate")
async def submit_hashrate(payload: SubmitHashrateRequest):
    """Store the latest hashrate for a worker and update total history."""
    worker = payload.worker_id
    gpu = payload.gpu_uuid
    rate = payload.hashrate
    if not verify_signature(worker, worker, payload.signature):
        raise HTTPException(status_code=401, detail="unauthorized")
    if rate is None:
        raise HTTPException(status_code=400, detail="worker_id and hashrate required")
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid rate")
    try:
        ts = int(datetime.utcnow().timestamp())
        r.hset(f"worker:{worker}", "hashrate", rate)
        if gpu:
            r.hset(f"gpu:{gpu}", "hashrate", rate)
        r.rpush(
            f"hashrate_history:{worker}",
            json.dumps({"ts": ts, "rate": rate}),
        )
        r.ltrim(f"hashrate_history:{worker}", -50, -1)

        total = 0.0
        for key in r.scan_iter("worker:*"):
            try:
                total += float(r.hget(key, "hashrate") or 0)
            except (TypeError, ValueError):
                pass
        r.rpush("hashrate_history:total", json.dumps({"ts": ts, "rate": total}))
        r.ltrim("hashrate_history:total", -50, -1)
        return {"status": "ok"}
    except redis.exceptions.RedisError as e:
        log_error("server", worker or "system", "SRED", "Redis unavailable", e)
        raise HTTPException(status_code=500, detail="redis unavailable")
    except Exception as e:
        log_error("server", worker, "S714", "Failed to record hashrate", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit_benchmark")
async def submit_benchmark(data: SubmitBenchmarkRequest):
    """Record benchmark results for a GPU and aggregate per-worker speed."""
    worker = data.worker_id
    gpu = data.gpu_uuid
    if not verify_signature(worker, worker, data.signature):
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        rates = {alg: float(val) for alg, val in data.hashrates.items()}
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid hashrates")
    try:
        r.hset(
            f"benchmark:{gpu}",
            mapping={"engine": data.engine, **rates},
        )

        totals = {"MD5": 0.0, "SHA1": 0.0, "NTLM": 0.0}
        specs_raw = r.hget(f"worker:{worker}", "specs")
        gpu_list = []
        if specs_raw:
            try:
                info = json.loads(specs_raw)
                gpu_list = (
                    info.get("hardware", {}).get("gpus", [])
                    if isinstance(info.get("hardware"), dict)
                    else []
                )
            except Exception:
                gpu_list = []
        if not gpu_list:
            gpu_list = [{"uuid": gpu}]

        for g in gpu_list:
            bench = r.hgetall(f"benchmark:{g.get('uuid')}")
            for alg in totals:
                try:
                    totals[alg] += float(bench.get(alg) or 0)
                except (TypeError, ValueError):
                    pass
        r.hset(f"benchmark_total:{worker}", mapping=totals)
        return {"status": "ok"}
    except redis.exceptions.RedisError as e:
        log_error("server", worker or "system", "SRED", "Redis unavailable", e)
        raise HTTPException(status_code=500, detail="redis unavailable")
    except Exception as e:
        log_error("server", worker, "S742", "Failed to record benchmark", e)
        raise HTTPException(status_code=500, detail="error")


@app.get("/hashrate")
async def get_hashrate(worker: str | None = None):
    """Return hashrate history for a worker or total."""
    key = "hashrate_history:total" if not worker else f"hashrate_history:{worker}"
    try:
        data = [json.loads(x) for x in r.lrange(key, 0, -1)]
        return data
    except redis.exceptions.RedisError as e:
        log_error("server", worker or "system", "SRED", "Redis unavailable", e)
        return {"status": "error", "message": "redis unavailable"}
    except Exception as e:
        log_error("server", worker or "system", "S715", "Failed to get hashrate", e)
        return []


@app.get("/get_flash_task")
async def get_flash_task(worker_id: str, signature: str):
    """Pop the next flash task for a worker."""
    if not verify_signature(worker_id, worker_id, signature):
        return {"status": "unauthorized"}
    try:
        data = r.lpop(f"flash:{worker_id}")
        if not data:
            return {"status": "none"}
        task = json.loads(data)
        return {
            "status": "ok",
            "gpu_uuid": task.get("gpu_uuid"),
            "settings": task.get("settings", {}),
        }
    except redis.exceptions.RedisError as e:
        log_error("server", worker_id, "SRED", "Redis unavailable", e)
        return {"status": "error", "message": "redis unavailable"}
    except Exception as e:
        log_error("server", worker_id, "S740", "Failed to get flash task", e)
        return {"status": "error"}


class FlashResult(BaseModel):
    worker_id: str
    gpu_uuid: str
    success: bool
    signature: str


@app.post("/flash_result")
async def flash_result(res: FlashResult):
    """Record result of a flash attempt."""
    if not verify_signature(res.worker_id, res.worker_id, res.signature):
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        if not res.success:
            crashes = r.hincrby(f"gpu:{res.gpu_uuid}", "crashes", 1)
            if crashes >= 3:
                # increase power limit slightly and retry
                settings = r.hgetall(f"flash:settings:{res.gpu_uuid}")
                if settings.get("power_limit"):
                    try:
                        val = int(settings["power_limit"]) + 10
                        settings["power_limit"] = val
                    except ValueError:
                        pass
                r.hset(f"flash:settings:{res.gpu_uuid}", mapping=settings)
            r.rpush(
                f"flash:{res.worker_id}",
                json.dumps(
                    {
                        "gpu_uuid": res.gpu_uuid,
                        "settings": r.hgetall(f"flash:settings:{res.gpu_uuid}") or {},
                    }
                ),
            )
        else:
            r.hset(f"gpu:{res.gpu_uuid}", "flashed", 1)
        return {"status": "ok"}
    except redis.exceptions.RedisError as e:
        log_error("server", res.worker_id, "SRED", "Redis unavailable", e)
        raise HTTPException(status_code=500, detail="redis unavailable")
    except Exception as e:
        log_error("server", res.worker_id, "S741", "Failed to record flash result", e)
        raise HTTPException(status_code=500, detail="error")


@app.post("/upload_wordlist")
async def upload_wordlist(file: UploadFile = File(...)):
    """Upload a new dictionary file to WORDLISTS_DIR."""
    try:
        filename = Path(file.filename).name
        dest = (WORDLISTS_DIR / filename).resolve()
        if dest.parent != WORDLISTS_DIR.resolve():
            raise HTTPException(status_code=400, detail="invalid filename")
        with dest.open("wb") as f:
            while True:
                chunk = await file.read(4096)
                if not chunk:
                    break
                f.write(chunk)
        return {"status": "ok"}
    except Exception as e:
        log_error("server", "system", "S720", "Failed to upload wordlist", e)
        raise HTTPException(status_code=500, detail="upload failed")


class TrainMarkovRequest(BaseModel):
    lang: str = "english"
    directory: str | None = None


@app.post("/train_markov")
async def train_markov(req: TrainMarkovRequest):
    """Process wordlists to build Markov statistics."""
    directory = Path(req.directory) if req.directory else WORDLISTS_DIR
    try:
        learn_trends.process_wordlists(directory, lang=req.lang)
        return {"status": "ok"}
    except Exception as e:
        log_error("server", "system", "S735", "Failed to train Markov", e)
        raise HTTPException(status_code=500, detail="training failed")


@app.post("/upload_restore")
async def upload_restore(file: UploadFile = File(...)):
    """Receive a hashcat restore file and store it in RESTORE_DIR."""
    try:
        RESTORE_DIR.mkdir(parents=True, exist_ok=True)
        filename = Path(file.filename).name
        dest = (RESTORE_DIR / filename).resolve()
        if dest.parent != RESTORE_DIR.resolve():
            raise HTTPException(status_code=400, detail="invalid filename")
        with dest.open("wb") as f:
            while True:
                chunk = await file.read(4096)
                if not chunk:
                    break
                f.write(chunk)
        log_info("server", "system", f"restore uploaded: {file.filename}")
        return {"status": "ok"}
    except Exception as e:
        log_error("server", "system", "S725", "Failed to upload restore file", e)
        raise HTTPException(status_code=500, detail="upload failed")


@app.delete("/wordlist/{name}")
async def delete_wordlist(name: str):
    """Delete a dictionary by filename."""
    try:
        path = WORDLISTS_DIR / name
        if path.is_file():
            path.unlink()
            return {"status": "ok"}
        raise HTTPException(status_code=404, detail="not found")
    except Exception as e:
        log_error("server", "system", "S721", "Failed to delete wordlist", e)
        raise HTTPException(status_code=500, detail="delete failed")


@app.post("/create_mask")
async def create_mask(name: str, content: str):
    """Create a mask file with provided content."""
    try:
        filename = Path(name).name
        dest = (MASKS_DIR / filename).resolve()
        if dest.parent != MASKS_DIR.resolve():
            raise HTTPException(status_code=400, detail="invalid filename")
        dest.write_text(content)
        return {"status": "ok"}
    except Exception as e:
        log_error("server", "system", "S722", "Failed to create mask", e)
        raise HTTPException(status_code=500, detail="mask creation failed")


@app.delete("/mask/{name}")
async def delete_mask(name: str):
    """Delete a mask file."""
    try:
        filename = Path(name).name
        path = (MASKS_DIR / filename).resolve()
        if path.parent != MASKS_DIR.resolve():
            raise HTTPException(status_code=400, detail="invalid filename")
        if path.is_file():
            path.unlink()
            return {"status": "ok"}
        raise HTTPException(status_code=404, detail="not found")
    except Exception as e:
        log_error("server", "system", "S723", "Failed to delete mask", e)
        raise HTTPException(status_code=500, detail="delete failed")


@app.get("/logs")
async def get_logs(worker: str | None = None):
    """Return log entries, optionally filtered by worker."""
    try:
        entries: list[dict] = []
        if worker:
            entries = [json.loads(x) for x in r.lrange(f"error_logs:{worker}", 0, -1)]
        else:
            for key in r.scan_iter("error_logs:*"):
                entries.extend(json.loads(x) for x in r.lrange(key, 0, -1))
            entries.sort(key=lambda d: d.get("datetime", ""), reverse=True)
        return entries
    except redis.exceptions.RedisError as e:
        log_error("server", worker or "system", "SRED", "Redis unavailable", e)
        return []


@app.get("/jobs")
async def list_jobs():
    """Return information about current jobs."""
    jobs = []
    try:
        for key in r.scan_iter("job:*"):
            info = r.hgetall(key)
            info["job_id"] = key.split(":", 1)[1]
            jobs.append(info)
        return jobs
    except redis.exceptions.RedisError as e:
        log_error("server", "system", "SRED", "Redis unavailable", e)
        return []


@app.get("/found_results")
async def list_found_results(limit: int = 100):
    """Return recent found results."""
    try:
        return r.lrange("found:results", -limit, -1)
    except redis.exceptions.RedisError as e:
        log_error("server", "system", "SRED", "Redis unavailable", e)
        return []


@app.get("/restores")
async def list_restore_files():
    """Return available `.restore` files."""
    try:
        RESTORE_DIR.mkdir(parents=True, exist_ok=True)
        return [p.name for p in RESTORE_DIR.glob("*.restore")]
    except Exception as e:
        log_error("server", "system", "S732", "Failed to list restore files", e)
        return []


@app.delete("/restore/{name}")
async def delete_restore_file(name: str):
    """Delete a restore file by filename."""
    try:
        filename = Path(name).name
        path = (RESTORE_DIR / filename).resolve()
        if path.parent != RESTORE_DIR.resolve():
            raise HTTPException(status_code=400, detail="invalid filename")
        if path.is_file():
            path.unlink()
            return {"status": "ok"}
        raise HTTPException(status_code=404, detail="not found")
    except HTTPException:
        raise
    except Exception as e:
        log_error("server", "system", "S733", "Failed to delete restore", e)
        raise HTTPException(status_code=500, detail="delete failed")


@app.get("/download_restore/{name}")
async def download_restore_file(name: str):
    """Download a restore file."""
    try:
        filename = Path(name).name
        path = (RESTORE_DIR / filename).resolve()
        if path.parent != RESTORE_DIR.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)
    except HTTPException:
        raise
    except Exception as e:
        log_error("server", "system", "S734", "Failed to download restore", e)
        raise HTTPException(status_code=500, detail="download failed")


@app.get("/hashes_jobs")
async def list_hashes_jobs():
    """Return jobs fetched from hashes.com."""
    jobs = []
    try:
        for key in r.scan_iter("hashes_job:*"):
            info = r.hgetall(key)
            info["id"] = key.split(":", 1)[1]
            jobs.append(info)
        return jobs
    except redis.exceptions.RedisError as e:
        log_error("server", "system", "SRED", "Redis unavailable", e)
        return []


class ApiKeyRequest(BaseModel):
    api_key: str


@app.post("/hashes_api_key")
async def set_hashes_api_key(req: ApiKeyRequest):
    """Update the stored hashes.com API key."""
    CONFIG["hashes_api_key"] = req.api_key
    save_config()
    os.environ["HASHES_COM_API_KEY"] = req.api_key
    # refresh module level variable
    import importlib

    importlib.reload(sys.modules["Server.hashescom_client"])  # type: ignore
    return {"status": "ok"}


class AlgoRequest(BaseModel):
    algorithms: list[str]


@app.post("/hashes_algorithms")
async def set_hashes_algorithms(req: AlgoRequest):
    """Set desired algorithm filters for hashes.com jobs."""
    CONFIG["hashes_algorithms"] = req.algorithms
    global HASHES_ALGORITHMS
    HASHES_ALGORITHMS = [a.lower() for a in req.algorithms]
    save_config()
    return {"status": "ok"}


class ProbOrderRequest(BaseModel):
    enabled: bool


@app.post("/probabilistic_order")
async def set_probabilistic_order(req: ProbOrderRequest):
    """Enable or disable probabilistic candidate ordering."""
    CONFIG["probabilistic_order"] = bool(req.enabled)
    global PROBABILISTIC_ORDER
    PROBABILISTIC_ORDER = bool(req.enabled)
    save_config()
    return {"status": "ok"}


class MarkovLangRequest(BaseModel):
    lang: str


@app.post("/markov_lang")
async def set_markov_lang(req: MarkovLangRequest):
    """Set the default language for Markov statistics."""
    CONFIG["markov_lang"] = req.lang
    global MARKOV_LANG
    MARKOV_LANG = req.lang
    save_config()
    return {"status": "ok"}


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve the admin interface."""
    try:
        html_path = Path(__file__).parent / "admin.html"
        return html_path.read_text()
    except Exception as e:
        log_error("server", "system", "S724", "Failed to load admin page", e)
        return HTMLResponse("<h1>Admin page not available</h1>", status_code=500)


@app.get("/portal", response_class=HTMLResponse)
async def portal_page():
    """Serve the combined portal interface."""
    try:
        html_path = Path(__file__).parent / "portal.html"
        return html_path.read_text()
    except Exception as e:
        log_error("server", "system", "S730", "Failed to load portal page", e)
        return HTMLResponse("<h1>Portal page not available</h1>", status_code=500)


@app.websocket("/ws/portal")
async def portal_ws(ws: WebSocket):
    """Push periodic metrics and found hashes over a WebSocket."""
    await ws.accept()
    last_count = 0
    try:
        while True:
            metrics = await server_status()
            workers = await list_workers()
            total = r.llen("found:results")
            founds: list[str] = []
            if total > last_count:
                founds = r.lrange("found:results", last_count, total - 1)
                last_count = total
            await ws.send_text(
                json.dumps(
                    {
                        "metrics": metrics,
                        "workers": workers,
                        "founds": founds,
                    }
                )
            )
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log_error("server", "system", "S731", "Portal WS error", e)
