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
import logging

logging.basicConfig(level=logging.INFO)
import subprocess
from datetime import datetime
import os
import redis
from redis_utils import get_redis
import json
import uuid
import asyncio
import glob
import sys
from utils import redis_manager
from utils.event_logger import log_error, log_info
from app.background import (
    broadcast_presence,
    fetch_and_store_jobs,
    poll_hashes_jobs,
    process_hashes_jobs,
    dispatch_loop,
    start_loops,
    stop_loops,
)
try:  # optional transformers dependency
    import train_llm as _train_llm  # type: ignore
except Exception:  # pragma: no cover - optional component
    _train_llm = None
from pathlib import Path
import learn_trends
import time
import hmac
import hashlib
import csv
import io
import tempfile
from filelock import FileLock

from waifus import assign_waifu
from auth_utils import (
    verify_signature,
    verify_signature_with_key,
    fingerprint_public_key,
)
from app.api.models import (
    LoginRequest,
    LogoutRequest,
    RegisterWorkerRequest,
    WorkerStatusRequest,
    SubmitHashrateRequest,
    SubmitBenchmarkRequest,
    FlashResult,
    SubmitFoundsRequest,
    SubmitNoFoundsRequest,
    TrainMarkovRequest,
    TrainLLMRequest,
    ApiKeyRequest,
    AlgoRequest,
    AlgoParamsRequest,
    HashesSettingsRequest,
    JobPriorityRequest,
    JobConfigRequest,
    MaskListRequest,
    ProbOrderRequest,
    InverseOrderRequest,
    MarkovLangRequest,
)
from argon2 import PasswordHasher
from ascii_logo import print_logo
from pattern_to_mask import get_top_masks
import psutil
import wordlist_db
from typing import Any
from collections.abc import Coroutine
from hash_algos import HASHCAT_ALGOS
from app import config as _config
from app.config import (
    CONFIG,
    CONFIG_FILE,
    WORDLISTS_DIR,
    MASKS_DIR,
    RULES_DIR,
    RESTORE_DIR,
    STORAGE_DIR,
    WORDLIST_DB_PATH,
    TRUSTED_KEYS_FILE,
    TRUSTED_KEY_FINGERPRINTS,
    FOUNDS_FILE,
    PORTAL_KEY,
    PORTAL_PASSKEY,
    SESSION_TTL,
    MAX_IMPORT_SIZE,
    LOW_BW_ENGINE,
    BROADCAST_ENABLED,
    WATCHDOG_TOKEN,
    HASHES_SETTINGS,
    HASHES_POLL_INTERVAL,
    HASHES_ALGORITHMS,
    HASHES_DEFAULT_PRIORITY,
    PREDEFINED_MASKS,
    HASHES_ALGO_PARAMS,
    PROBABILISTIC_ORDER,
    INVERSE_PROB_ORDER,
    MARKOV_LANG,
    LLM_ENABLED,
    LLM_MODEL_PATH,
    LLM_TRAIN_EPOCHS,
    LLM_TRAIN_LEARNING_RATE,
    save_config,
    load_config,
)

# reload configuration in case HOME changed before import
load_config()
CONFIG = _config.CONFIG  # refresh local reference after reload

app = FastAPI()

r = get_redis()
password_hasher = PasswordHasher()

# store references to background tasks so they can be cancelled
BACKGROUND_TASKS: list[asyncio.Task] = []


def create_background_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """Create a task and track it for cleanup on shutdown."""
    task = asyncio.create_task(coro)
    BACKGROUND_TASKS.append(task)
    return task

JOB_STREAM = os.getenv("JOB_STREAM", "jobs")
HTTP_GROUP = os.getenv("HTTP_GROUP", "http-workers")
LOW_BW_JOB_STREAM = os.getenv("LOW_BW_JOB_STREAM", "darkling-jobs")
LOW_BW_GROUP = os.getenv("LOW_BW_GROUP", "darkling-workers")


# propagate the model path for orchestrator_agent if enabled
if LLM_ENABLED and LLM_MODEL_PATH:
    os.environ["LLM_MODEL_PATH"] = LLM_MODEL_PATH
else:
    os.environ.pop("LLM_MODEL_PATH", None)

import orchestrator_agent




def sign_session(session_id: str, expiry: int) -> str:
    """Return a signed session token using the configured passkey."""
    key = (PORTAL_PASSKEY or "").encode()
    payload = f"{session_id}|{expiry}".encode()
    sig = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return f"{session_id}|{expiry}|{sig}"


def verify_session_token(token: str) -> bool:
    """Validate a session token and confirm it's stored in Redis."""
    try:
        session_id, exp_s, sig = token.split("|")
        expiry = int(exp_s)
    except ValueError:
        return False
    if expiry < int(time.time()):
        return False
    expected = hmac.new(
        (PORTAL_PASSKEY or "").encode(), f"{session_id}|{expiry}".encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return False
    return bool(r.exists(f"session:{session_id}"))


def verify_hash(password: str, hash_str: str, algorithm: str) -> bool:
    """Return True if password hashes to hash_str for the given algorithm."""
    algo = algorithm.lower()
    try:
        if algo == "ntlm":
            digest = hashlib.new("md4", password.encode("utf-16le")).hexdigest()
        else:
            digest = hashlib.new(algo, password.encode()).hexdigest()
    except Exception:
        return False
    return digest.lower() == hash_str.lower()

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


origins = CONFIG.get("allowed_origins", ["*"])
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
                path == "/"
                or path.startswith("/portal")
                or path.startswith("/glyph")
                or path.startswith("/admin")
            ):
                headers = {
                    k.decode().lower(): v.decode() for k, v in scope.get("headers", [])
                }
                if headers.get("x-api-key") != self.key:
                    cookie_header = headers.get("cookie", "")
                    token = None
                    for part in cookie_header.split(";"):
                        if part.strip().startswith("session="):
                            token = part.strip().split("=", 1)[1]
                            break
                    if not token or not verify_session_token(token):
                        response = HTMLResponse("Unauthorized", status_code=401)
                        await response(scope, receive, send)
                        return
        await self.app(scope, receive, send)


app.add_middleware(PortalAuthMiddleware, key=PORTAL_KEY)


@app.post("/login")
async def login(req: LoginRequest):
    initial_token = CONFIG.get("initial_admin_token")
    if initial_token and req.passkey == initial_token:
        username = getattr(req, "username", None)
        password = getattr(req, "password", None)
        if not username or not password:
            raise HTTPException("setup required")
        CONFIG["admin_username"] = username
        CONFIG["admin_password_hash"] = password_hasher.hash(password)
        CONFIG.pop("initial_admin_token", None)
        save_config()
    elif not PORTAL_PASSKEY or req.passkey != PORTAL_PASSKEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    session_id = uuid.uuid4().hex
    expiry = int(time.time()) + SESSION_TTL
    token = sign_session(session_id, expiry)
    r.set(f"session:{session_id}", 1, ex=SESSION_TTL)
    return {"status": "ok", "cookie": token}


@app.post("/logout")
async def logout(req: LogoutRequest):
    """Invalidate a session token."""
    try:
        session_id = req.token.split("|")[0]
        r.delete(f"session:{session_id}")
    except Exception:
        pass
    return {"status": "ok", "cookie": ""}




@app.on_event("startup")
async def start_broadcast():
    print_logo()
    tasks = start_loops()
    BACKGROUND_TASKS.extend(tasks)


@app.on_event("shutdown")
async def shutdown_event():
    """Cancel all background tasks and wait for them to finish."""
    await stop_loops(BACKGROUND_TASKS)
    BACKGROUND_TASKS.clear()


@app.post("/register_worker")
async def register_worker(info: RegisterWorkerRequest):
    try:
        worker_id = info.worker_id
        if not verify_signature_with_key(info.pubkey, worker_id, info.timestamp, info.signature):
            raise HTTPException(status_code=401, detail="unauthorized")

        if TRUSTED_KEY_FINGERPRINTS:
            fp = fingerprint_public_key(info.pubkey)
            if fp not in TRUSTED_KEY_FINGERPRINTS:
                raise HTTPException("untrusted key")

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
    except HTTPException:
        raise
    except Exception as e:
        log_error("server", "unassigned", "S700", "Worker registration failed", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_batch")
async def get_batch(worker_id: str, timestamp: int, signature: str):
    try:
        if not verify_signature(worker_id, worker_id, timestamp, signature):
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
async def submit_founds(payload: SubmitFoundsRequest):
    try:
        if not verify_signature(
            payload.worker_id,
            json.dumps(payload.founds),
            int(getattr(payload, "timestamp", 0)),
            payload.signature,
        ):
            return {"status": "unauthorized"}

        for line in payload.founds:
            r.rpush("found:results", f"{payload.batch_id}:{line}")
            try:
                hash_str, password = line.split(":", 1)
            except ValueError:
                continue
            r.hset("found:map", hash_str, password)
        try:
            lock = FileLock(str(FOUNDS_FILE) + ".lock")
            with lock:
                with FOUNDS_FILE.open("a", encoding="utf-8") as fh:
                    for line in payload.founds:
                        fh.write(line + "\n")
        except Exception:
            pass

        job_id = payload.job_id or payload.batch_id
        info = r.hgetall(f"job:{job_id}")
        msg_id = payload.msg_id or info.get("msg_id")
        stream = info.get("stream", JOB_STREAM)
        group = HTTP_GROUP if stream == JOB_STREAM else LOW_BW_GROUP
        if msg_id:
            r.xack(stream, group, msg_id)

        try:
            start = int(info.get("start", 0))
            end = int(info.get("end", 0))
            if start or end:
                redis_manager.complete_range(payload.batch_id, start, end)
        except Exception:
            pass

        r.hset(f"worker:{payload.worker_id}", "status", "idle")
        return {"status": "ok", "received": len(payload.founds)}
    except redis.exceptions.RedisError as e:
        log_error(
            "server", payload.worker_id if hasattr(payload, "worker_id") else "system", "SRED", "Redis unavailable", e
        )
        return {"status": "error", "message": "redis unavailable"}
    except Exception as e:
        log_error("server", payload.worker_id, "S003", "Failed to accept founds", e)
        return {"status": "error"}


@app.post("/submit_no_founds")
async def submit_no_founds(payload: SubmitNoFoundsRequest):
    try:
        if not verify_signature(
            payload.worker_id,
            payload.batch_id,
            int(getattr(payload, "timestamp", 0)),
            payload.signature,
        ):
            return {"status": "unauthorized"}

        r.rpush("found:none", payload.batch_id)

        job_id = payload.job_id or payload.batch_id
        info = r.hgetall(f"job:{job_id}")
        msg_id = payload.msg_id or info.get("msg_id")
        stream = info.get("stream", JOB_STREAM)
        group = HTTP_GROUP if stream == JOB_STREAM else LOW_BW_GROUP
        if msg_id:
            r.xack(stream, group, msg_id)

        try:
            start = int(info.get("start", 0))
            end = int(info.get("end", 0))
            if start or end:
                redis_manager.complete_range(payload.batch_id, start, end)
        except Exception:
            pass

        r.hset(f"worker:{payload.worker_id}", "status", "idle")
        return {"status": "ok"}
    except redis.exceptions.RedisError as e:
        log_error(
            "server", payload.worker_id if hasattr(payload, "worker_id") else "system", "SRED", "Redis unavailable", e
        )
        return {"status": "error", "message": "redis unavailable"}
    except Exception as e:
        log_error(
            "server", payload.worker_id, "S004", "Failed to record empty result", e
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
            "inverse_prob_order": INVERSE_PROB_ORDER,
            "markov_lang": MARKOV_LANG,
            "llm_train_epochs": LLM_TRAIN_EPOCHS,
            "llm_train_learning_rate": LLM_TRAIN_LEARNING_RATE,
            "cpu_usage": None,
            "memory_utilization": None,
            "disk_space": None,
            "cpu_load": None,
            "memory_usage": None,
            "backlog_target": None,
            "pending_jobs": None,
            "queued_batches": None,
        }
        try:
            status["cpu_usage"] = psutil.cpu_percent(interval=None)
        except Exception:
            pass
        try:
            vm = psutil.virtual_memory()
            status["memory_utilization"] = vm.percent
            status["memory_usage"] = getattr(vm, "used", None)
        except Exception:
            pass
        try:
            status["disk_space"] = psutil.disk_usage("/").percent
        except Exception:
            pass
        try:
            status["cpu_load"] = psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else os.getloadavg()[0]
        except Exception:
            pass
        try:
            status["backlog_target"] = orchestrator_agent.compute_backlog_target()
        except Exception:
            pass
        try:
            status["pending_jobs"] = orchestrator_agent.pending_count()
        except Exception:
            pass
        try:
            status["queued_batches"] = r.llen("batch:queue")
        except Exception:
            pass
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


@app.get("/workers/{worker_id}/stats")
async def get_worker_stats(worker_id: str, token: str | None = None):
    """Return temperature, hashrate and status for a worker."""
    if WATCHDOG_TOKEN and token != WATCHDOG_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        info = r.hgetall(f"worker:{worker_id}")
        if not info:
            raise HTTPException(status_code=404, detail="not found")
        temps = []
        if info.get("temps"):
            try:
                temps = json.loads(info["temps"])
            except Exception:
                temps = []
        hashrate = 0.0
        try:
            hashrate = float(info.get("hashrate", 0))
        except (TypeError, ValueError):
            hashrate = 0.0
        return {
            "temps": temps,
            "hashrate": hashrate,
            "status": info.get("status", "unknown"),
        }
    except redis.exceptions.RedisError as e:
        log_error("server", worker_id or "system", "SRED", "Redis unavailable", e)
        raise HTTPException(status_code=500, detail="redis unavailable")
    except HTTPException:
        raise
    except Exception as e:
        log_error("server", worker_id, "S750", "Failed to fetch stats", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workers/{worker_id}/reboot")
async def reboot_worker(worker_id: str, token: str | None = None):
    """Queue a reboot command for the worker."""
    if WATCHDOG_TOKEN and token != WATCHDOG_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        r.rpush(f"reboot:{worker_id}", "reboot")
        return {"status": "queued"}
    except redis.exceptions.RedisError as e:
        log_error("server", worker_id or "system", "SRED", "Redis unavailable", e)
        raise HTTPException(status_code=500, detail="redis unavailable")
    except Exception as e:
        log_error("server", worker_id, "S751", "Failed to queue reboot", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/worker_status")
async def set_worker_status(data: WorkerStatusRequest):
    """Update a worker's status string."""
    name = data.name
    status = data.status
    signature = data.signature
    timestamp = data.timestamp
    if not name or status is None:
        raise HTTPException(status_code=400, detail="name and status required")
    if not verify_signature(name, name, timestamp, signature):
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        mapping = {"status": status, "last_seen": int(time.time())}
        temps = getattr(data, "temps", None)
        progress = getattr(data, "progress", None)
        if temps is not None:
            mapping["temps"] = json.dumps(temps)
        if progress is not None:
            mapping["progress"] = json.dumps(progress)
        r.hset(f"worker:{name}", mapping=mapping)
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
    if not verify_signature(worker, worker, payload.timestamp, payload.signature):
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
    if not verify_signature(worker, worker, data.timestamp, data.signature):
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
async def get_flash_task(worker_id: str, timestamp: int, signature: str):
    """Pop the next flash task for a worker."""
    if not verify_signature(worker_id, worker_id, timestamp, signature):
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


@app.post("/flash_result")
async def flash_result(res: FlashResult):
    """Record result of a flash attempt."""
    if not verify_signature(res.worker_id, res.worker_id, res.timestamp, res.signature):
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
    """Stream a dictionary upload directly into the SQLite database."""
    conn = None
    try:
        filename = Path(file.filename).name
        with tempfile.NamedTemporaryFile() as tmp:
            size = 0
            while True:
                chunk = await file.read(4096)
                if not chunk:
                    break
                tmp.write(chunk)
                size += len(chunk)
            tmp.flush()
            tmp.seek(0)
            conn = wordlist_db.connect()
            cur = conn.execute(
                "INSERT OR REPLACE INTO wordlists(name, data) VALUES(?, zeroblob(?))",
                (filename, size),
            )
            rowid = cur.lastrowid
            blob = conn.blobopen("wordlists", "data", rowid, readonly=False)
            while True:
                chunk = tmp.read(4096)
                if not chunk:
                    break
                blob.write(chunk)
            blob.close()
            conn.commit()
            conn.close()
        return {"status": "ok"}
    except Exception as e:
        if conn:
            conn.close()
        log_error("server", "system", "S720", "Failed to upload wordlist", e)
        raise HTTPException(status_code=500, detail="upload failed")


@app.post("/import_hashes")
async def import_hashes(file: UploadFile = File(...), hash_mode: str = "0"):
    """Parse a CSV of hashes and queue cracking batches."""
    try:
        queued: list[str] = []
        errors: list[str] = []
        total = 0
        buffer = b""

        # read CSV header
        while True:
            chunk = await file.read(4096)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_IMPORT_SIZE:
                raise HTTPException(400, "file too large")
            buffer += chunk
            if b"\n" in buffer:
                break
        if b"\n" not in buffer:
            raise HTTPException(400, "invalid csv")
        header_line, buffer = buffer.split(b"\n", 1)
        fieldnames = next(csv.reader([header_line.decode()]))
        line_num = 0

        async def iter_lines():
            nonlocal buffer, total
            while True:
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    yield line.decode()
                chunk = await file.read(4096)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_IMPORT_SIZE:
                    raise HTTPException(400, "file too large")
                buffer += chunk
            if buffer:
                yield buffer.decode()

        async for line in iter_lines():
            line_num += 1
            if not line.strip():
                continue
            row = next(csv.DictReader([line], fieldnames=fieldnames))
            h = (row.get("hash") or "").strip()
            if not h:
                errors.append(f"line {line_num}: missing hash")
                continue
            mask = (row.get("mask") or "").strip()
            wordlist = (row.get("wordlist") or "").strip()
            target = row.get("target") or "any"
            batch_id = redis_manager.store_batch(
                [h], mask=mask, wordlist=wordlist, rule="", target=target, hash_mode=hash_mode
            )
            if batch_id:
                queued.append(batch_id)
            else:
                errors.append(f"line {line_num}: redis error")
        return {"queued": len(queued), "errors": errors}
    except HTTPException:
        raise
    except Exception as e:
        log_error("server", "system", "S726", "Failed to import hashes", e)
        raise HTTPException(status_code=500, detail="import failed")




async def train_llm(req: TrainLLMRequest):
    """Fine-tune a local language model using transformers."""
    dataset = Path(req.dataset)
    out_dir = Path(req.output_dir)
    try:
        if _train_llm is None:
            raise RuntimeError("transformers not available")
        _train_llm.train_model(
            dataset,
            req.base_model,
            req.epochs,
            req.learning_rate,
            out_dir,
        )
        return {"status": "ok"}
    except Exception as e:
        log_error("server", "system", "S741", "Failed to train LLM", e)
        raise HTTPException(status_code=500, detail="training failed")


@app.post("/train_markov")
async def train_markov(req: TrainMarkovRequest):
    """Process wordlists to build Markov statistics."""
    directory = Path(req.directory) if req.directory else WORDLISTS_DIR
    try:
        create_background_task(
            asyncio.to_thread(learn_trends.process_wordlists, directory, lang=req.lang)
        )
        return {"status": "scheduled"}
    except Exception as e:
        log_error("server", "system", "S735", "Failed to train Markov", e)
        raise HTTPException(status_code=500, detail="training failed")


@app.post("/train_llm")
async def train_llm_endpoint(req: TrainLLMRequest):
    """Fine-tune a local language model using transformers."""
    dataset = Path(req.dataset)
    out_dir = Path(req.output_dir)
    try:
        if _train_llm is None:
            raise RuntimeError("transformers not available")
        create_background_task(
            asyncio.to_thread(
                _train_llm.train_model,
                dataset,
                req.base_model,
                req.epochs,
                req.learning_rate,
                out_dir,
            )
        )
        return {"status": "scheduled"}
    except Exception as e:
        log_error("server", "system", "S741", "Failed to train LLM", e)
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


@app.post("/hashes_algorithms")
async def set_hashes_algorithms(req: AlgoRequest):
    """Set desired algorithm filters for hashes.com jobs."""
    CONFIG["hashes_algorithms"] = req.algorithms
    global HASHES_ALGORITHMS
    HASHES_ALGORITHMS = [a.lower() for a in req.algorithms]
    save_config()
    return {"status": "ok"}


@app.get("/hashes_algo_params")
async def get_hashes_algo_params():
    return HASHES_ALGO_PARAMS


@app.post("/hashes_algo_params")
async def set_hashes_algo_params(req: AlgoParamsRequest):
    algo = req.algo.lower()
    CONFIG.setdefault("hashes_algo_params", {})[algo] = req.params
    global HASHES_ALGO_PARAMS
    HASHES_ALGO_PARAMS[algo] = req.params
    save_config()
    return {"status": "ok"}


@app.get("/hashes_settings")
async def get_hashes_settings():
    return HASHES_SETTINGS


@app.post("/hashes_settings")
async def set_hashes_settings(req: HashesSettingsRequest):
    if req.hashes_poll_interval is not None:
        HASHES_SETTINGS["hashes_poll_interval"] = int(req.hashes_poll_interval)
    if req.algo_params is not None:
        HASHES_SETTINGS.setdefault("algo_params", {}).update(req.algo_params)
    CONFIG["hashes_settings"] = HASHES_SETTINGS
    global HASHES_POLL_INTERVAL, HASHES_ALGO_PARAMS
    HASHES_POLL_INTERVAL = int(HASHES_SETTINGS.get("hashes_poll_interval", HASHES_POLL_INTERVAL))
    HASHES_ALGO_PARAMS = dict(HASHES_SETTINGS.get("algo_params", {}))
    save_config()
    return {"status": "ok"}


@app.get("/hashes_job_config")
async def get_hashes_job_config():
    cfg: dict[str, int] = {}
    try:
        for key in r.scan_iter("hashes_job:*"):
            prio = r.hget(key, "priority")
            if prio is not None:
                cfg[key.split(":", 1)[1]] = int(prio)
        return cfg
    except redis.exceptions.RedisError as e:
        log_error("server", "system", "SRED", "Redis unavailable", e)
        return {}


@app.post("/hashes_job_config")
async def set_hashes_job_config(req: JobConfigRequest):
    return await set_hashes_job_priority(req)


@app.post("/hashes_job_priority")
async def set_hashes_job_priority(req: JobPriorityRequest):
    """Set the priority for a Hashes.com job."""
    try:
        r.hset(f"hashes_job:{req.job_id}", "priority", int(req.priority))
        return {"status": "ok"}
    except redis.exceptions.RedisError as e:
        log_error("server", "system", "SRED", "Redis unavailable", e)
        return {"status": "error", "message": "redis unavailable"}


@app.get("/predefined_masks")
async def get_predefined_masks():
    """Return the list of predefined masks."""
    return list(PREDEFINED_MASKS)


@app.post("/predefined_masks")
async def set_predefined_masks(req: MaskListRequest):
    """Replace the predefined mask list."""
    CONFIG["predefined_masks"] = req.masks
    global PREDEFINED_MASKS
    PREDEFINED_MASKS = list(req.masks)
    save_config()
    return {"status": "ok"}


@app.delete("/predefined_masks")
async def clear_predefined_masks():
    """Remove all predefined masks."""
    CONFIG["predefined_masks"] = []
    global PREDEFINED_MASKS
    PREDEFINED_MASKS = []
    save_config()
    return {"status": "ok"}


@app.get("/hash_algos")
async def get_hash_algos():
    """Return mapping of hash algorithm names to hashcat mode IDs."""
    return HASHCAT_ALGOS


@app.post("/probabilistic_order")
async def set_probabilistic_order(req: ProbOrderRequest):
    """Enable or disable probabilistic candidate ordering."""
    CONFIG["probabilistic_order"] = bool(req.enabled)
    global PROBABILISTIC_ORDER
    PROBABILISTIC_ORDER = bool(req.enabled)
    save_config()
    return {"status": "ok"}


@app.post("/inverse_prob_order")
async def set_inverse_prob_order(req: InverseOrderRequest):
    """Enable or disable inverse probabilistic ordering."""
    CONFIG["inverse_prob_order"] = bool(req.enabled)
    global INVERSE_PROB_ORDER
    INVERSE_PROB_ORDER = bool(req.enabled)
    save_config()
    return {"status": "ok"}


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


@app.get("/login_page", response_class=HTMLResponse)
async def login_page():
    """Serve a simple login form for the portal."""
    try:
        html_path = Path(__file__).parent / "login.html"
        return html_path.read_text()
    except Exception as e:
        log_error("server", "system", "S729", "Failed to load login page", e)
        return HTMLResponse("<h1>Login page not available</h1>", status_code=500)


@app.get("/", response_class=HTMLResponse)
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
