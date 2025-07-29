import asyncio
import os
import time
import redis

from hashmancer.utils.event_logger import log_error, log_watchdog_event

STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", "30"))


async def watchdog_loop() -> None:
    """Check worker heartbeats and mark stale workers offline."""
    while True:
        try:
            from hashmancer.server import main  # late import so tests can patch 'r'
            r = main.r
            now = int(time.time())
            threshold = 5 * STATUS_INTERVAL
            for key in r.scan_iter("worker:*"):
                info = r.hgetall(key)
                last_seen = int(info.get("last_seen", 0))
                if now - last_seen > threshold and info.get("status") != "offline":
                    r.hset(key, "status", "offline")
                    name = key.split(":", 1)[1]
                    age = now - last_seen
                    log_watchdog_event({
                        "worker_id": name,
                        "type": "offline",
                        "notes": str(age),
                    })
        except redis.exceptions.RedisError as e:
            log_error("server", "system", "SRED", "Redis unavailable", e)
        except Exception as e:  # pragma: no cover - unexpected failures
            log_error("server", "system", "S744", "Watchdog loop failed", e)
        await asyncio.sleep(60)
