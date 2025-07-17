from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import subprocess
from datetime import datetime
import os
import redis
import json
import uuid
import socket
import threading
import time
from event_logger import log_error
from pathlib import Path

from waifus import assign_waifu
from auth_utils import verify_signature

app = FastAPI()

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

JOB_STREAM = os.getenv("JOB_STREAM", "jobs")
HTTP_GROUP = os.getenv("HTTP_GROUP", "http-workers")

CONFIG_FILE = Path.home() / ".hashmancer" / "server_config.json"
try:
    with open(CONFIG_FILE) as cfg:
        CONFIG = json.load(cfg)
except Exception:
    CONFIG = {}

WORDLISTS_DIR = Path(CONFIG.get("wordlists_dir", "/opt/hashmancer/wordlists"))
MASKS_DIR = Path(CONFIG.get("masks_dir", "/opt/hashmancer/masks"))
RULES_DIR = Path(CONFIG.get("rules_dir", "/opt/hashmancer/rules"))

# broadcast settings
BROADCAST_ENABLED = bool(CONFIG.get("broadcast_enabled", True))
BROADCAST_PORT = int(CONFIG.get("broadcast_port", 50000))
BROADCAST_INTERVAL = int(CONFIG.get("broadcast_interval", 30))

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def broadcast_presence():
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
        time.sleep(BROADCAST_INTERVAL)


@app.on_event("startup")
async def start_broadcast():
    if BROADCAST_ENABLED:
        thread = threading.Thread(target=broadcast_presence, daemon=True)
        thread.start()


@app.post("/register_worker")
async def register_worker(info: dict):
    try:
        worker_id = info.get("worker_id", str(uuid.uuid4()))
        signature = info.get("signature")
        if signature and not verify_signature(worker_id, worker_id, signature):
            return {"status": "unauthorized"}

        specs = {
            "mode": info.get("mode", "eco"),
            "provider": info.get("provider", "on-prem"),
            "hardware": info.get("hardware", {}),
        }
        pubkey = info.get("pubkey")
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
            },
        )
        return {"status": "ok", "waifu": waifu_name}
    except Exception as e:
        log_error("server", "unassigned", "S700", "Worker registration failed", e)
        return {"status": "error", "message": str(e)}


@app.get("/get_batch")
async def get_batch(worker_id: str, signature: str):
    try:
        if not verify_signature(worker_id, worker_id, signature):
            return {"status": "unauthorized"}
        try:
            r.xgroup_create(JOB_STREAM, HTTP_GROUP, id="0", mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        messages = r.xreadgroup(HTTP_GROUP, worker_id, {JOB_STREAM: ">"}, count=1, block=1000)
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
                r.xack(JOB_STREAM, HTTP_GROUP, msg_id)
            return {"status": "none"}

        batch = r.hgetall(f"job:{job_id}")
        batch["job_id"] = job_id
        r.xack(JOB_STREAM, HTTP_GROUP, msg_id)
        r.hset(
            f"worker:{worker_id}",
            mapping={
                "last_batch": job_id,
                "last_seen": int(datetime.utcnow().timestamp()),
                "status": "processing",
            },
        )
        return batch
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
        r.hset(f"worker:{payload['worker_id']}", "status", "idle")
        return {"status": "ok", "received": len(payload["founds"])}
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
        r.hset(f"worker:{payload['worker_id']}", "status", "idle")
        return {"status": "ok"}
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


@app.get("/rules")
async def list_rules():
    try:
        return [f.name for f in RULES_DIR.iterdir() if f.is_file()]
    except Exception as e:
        log_error("server", "system", "S707", "Failed to list rules", e)
        return []


def get_gpu_temps():
    """Return a list of GPU temperatures using nvidia-smi if available."""
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
        return []


@app.get("/server_status")
async def server_status():
    """Return basic server and GPU metrics."""
    try:
        status = {
            "worker_count": r.scard("waifu:names"),
            "queue_length": r.llen("batch:queue"),
            "found_results": r.llen("found:results"),
            "gpu_temps": get_gpu_temps(),
        }
        return status
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
    except Exception as e:
        log_error("server", "system", "S712", "Failed to list workers", e)
        return []


@app.post("/worker_status")
async def set_worker_status(data: dict):
    """Update a worker's status string."""
    name = data.get("name")
    status = data.get("status")
    signature = data.get("signature")
    if not name or status is None:
        return {"status": "error", "message": "name and status required"}
    if signature and not verify_signature(name, name, signature):
        return {"status": "unauthorized"}
    try:
        r.hset(f"worker:{name}", "status", status)
        return {"status": "ok"}
    except Exception as e:
        log_error("server", name, "S713", "Failed to set status", e)
        return {"status": "error", "message": str(e)}


@app.post("/submit_hashrate")
async def submit_hashrate(payload: dict):
    """Store the latest hashrate for a worker and update total history."""
    worker = payload.get("worker_id")
    rate = payload.get("hashrate")
    if not worker or rate is None:
        return {"status": "error", "message": "worker_id and hashrate required"}
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        return {"status": "error", "message": "invalid rate"}
    try:
        ts = int(datetime.utcnow().timestamp())
        r.hset(f"worker:{worker}", "hashrate", rate)
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
    except Exception as e:
        log_error("server", worker, "S714", "Failed to record hashrate", e)
        return {"status": "error", "message": str(e)}


@app.get("/hashrate")
async def get_hashrate(worker: str | None = None):
    """Return hashrate history for a worker or total."""
    key = "hashrate_history:total" if not worker else f"hashrate_history:{worker}"
    try:
        data = [json.loads(x) for x in r.lrange(key, 0, -1)]
        return data
    except Exception as e:
        log_error("server", worker or "system", "S715", "Failed to get hashrate", e)
        return []
