import redis
from redis_utils import get_redis
import json
import traceback
from datetime import datetime

r = get_redis()


def log_event(
    worker_id, code, message, level="info", details=None, component="general"
):
    event = {
        "datetime": datetime.utcnow().isoformat(),
        "worker_id": worker_id,
        "component": component,
        "code": code,
        "level": level,
        "message": message,
    }
    if details:
        event["traceback"] = traceback.format_exc(limit=2)

    try:
        r.rpush(f"error_logs:{worker_id}", json.dumps(event))
        r.ltrim(f"error_logs:{worker_id}", -100, -1)  # Keep last 100 logs
    except redis.exceptions.RedisError:
        print("Redis unavailable: log_event", event)


def log_error(component, worker_id, code, message, exception=None):
    tb = traceback.format_exc(limit=2) if exception else None
    log_event(worker_id, code, message, level="error", details=tb, component=component)


def log_info(component, worker_id, message):
    log_event(worker_id, "I001", message, level="info", component=component)


def log_watchdog_event(payload: dict):
    worker_id = payload.get("worker_id", "unknown")
    event = {
        "datetime": datetime.utcnow().isoformat(),
        "type": payload.get("type", "unknown"),
        "worker_id": worker_id,
        "uptime": payload.get("uptime", ""),
        "gpus": payload.get("gpus", []),
        "load": payload.get("load", ""),
        "notes": payload.get("notes", ""),
    }
    try:
        r.rpush(f"watchdog_events:{worker_id}", json.dumps(event))
        r.ltrim(f"watchdog_events:{worker_id}", -100, -1)
    except redis.exceptions.RedisError:
        print("Redis unavailable: log_watchdog_event", event)
