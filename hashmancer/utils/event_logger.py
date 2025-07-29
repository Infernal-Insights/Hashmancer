import os
import redis
import json
import traceback
from datetime import datetime
import logging


def get_redis() -> redis.Redis:
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD")
    use_ssl = os.getenv("REDIS_SSL", "0")
    ssl_cert = os.getenv("REDIS_SSL_CERT")
    ssl_key = os.getenv("REDIS_SSL_KEY")
    ssl_ca = os.getenv("REDIS_SSL_CA_CERT")

    opts: dict[str, str | int | bool] = {
        "host": host,
        "port": port,
        "decode_responses": True,
    }
    if password:
        opts["password"] = password
    if str(use_ssl).lower() in {"1", "true", "yes"}:
        opts["ssl"] = True
        if ssl_ca:
            opts["ssl_ca_certs"] = ssl_ca
        if ssl_cert:
            opts["ssl_certfile"] = ssl_cert
        if ssl_key:
            opts["ssl_keyfile"] = ssl_key
    return redis.Redis(**opts)

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
    if details is not None:
        event["traceback"] = details
    else:
        tb = traceback.format_exc(limit=2)
        if tb and tb != "NoneType: None\n":
            event["traceback"] = tb

    try:
        if hasattr(r, "rpush"):
            r.rpush(f"error_logs:{worker_id}", json.dumps(event))
            r.ltrim(f"error_logs:{worker_id}", -100, -1)  # Keep last 100 logs
    except redis.exceptions.RedisError:
        logging.warning("Redis unavailable: log_event %s", event)
    except AttributeError:
        # r may be a stub in tests
        pass


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
        if hasattr(r, "rpush"):
            r.rpush(f"watchdog_events:{worker_id}", json.dumps(event))
            r.ltrim(f"watchdog_events:{worker_id}", -100, -1)
    except redis.exceptions.RedisError:
        logging.warning("Redis unavailable: log_watchdog_event %s", event)
    except AttributeError:
        pass
