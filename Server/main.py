from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
from event_logger import log_error, log_info
from pathlib import Path

from waifus import assign_waifu
from auth_utils import verify_signature, verify_signature_with_key
from pydantic import BaseModel
from ascii_logo import print_logo

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
            if path.startswith("/portal") or path.startswith("/glyph") or path.startswith("/admin"):
                headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
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


@app.on_event("startup")
async def start_broadcast():
    print_logo()
    if BROADCAST_ENABLED:
        asyncio.create_task(broadcast_presence())


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
        r.xack(stream, group, msg_id)
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
        r.hset(f"worker:{payload['worker_id']}", "status", "idle")
        return {"status": "ok", "received": len(payload["founds"])}
    except redis.exceptions.RedisError as e:
        log_error("server", payload.get("worker_id", "system"), "SRED", "Redis unavailable", e)
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
        r.hset(f"worker:{payload['worker_id']}", "status", "idle")
        return {"status": "ok"}
    except redis.exceptions.RedisError as e:
        log_error("server", payload.get("worker_id", "system"), "SRED", "Redis unavailable", e)
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
                json.dumps({
                    "metrics": metrics,
                    "workers": workers,
                    "founds": founds,
                })
            )
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log_error("server", "system", "S731", "Portal WS error", e)
