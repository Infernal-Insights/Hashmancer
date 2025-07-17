import os
import json
from pathlib import Path
import redis

CONFIG_FILE = Path.home() / ".hashmancer" / "server_config.json"


def get_redis() -> redis.Redis:
    """Return a Redis connection using env vars or server_config.json."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            host = os.getenv("REDIS_HOST", cfg.get("redis_host", host))
            port = int(os.getenv("REDIS_PORT", cfg.get("redis_port", port)))
        except Exception:
            pass
    return redis.Redis(host=host, port=port, decode_responses=True)
