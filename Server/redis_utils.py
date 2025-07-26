import os
import json
import redis

from app.config import CONFIG_FILE


def get_redis() -> redis.Redis:
    """Return a Redis connection using env vars or server_config.json."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD")
    use_ssl = os.getenv("REDIS_SSL", "0")
    ssl_cert = os.getenv("REDIS_SSL_CERT")
    ssl_key = os.getenv("REDIS_SSL_KEY")
    ssl_ca = os.getenv("REDIS_SSL_CA_CERT")
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            host = os.getenv("REDIS_HOST", cfg.get("redis_host", host))
            port = int(os.getenv("REDIS_PORT", cfg.get("redis_port", port)))
            password = os.getenv(
                "REDIS_PASSWORD", cfg.get("redis_password", password)
            )
            use_ssl = os.getenv("REDIS_SSL", str(cfg.get("redis_ssl", use_ssl)))
            ssl_cert = os.getenv(
                "REDIS_SSL_CERT", cfg.get("redis_ssl_cert", ssl_cert)
            )
            ssl_key = os.getenv(
                "REDIS_SSL_KEY", cfg.get("redis_ssl_key", ssl_key)
            )
            ssl_ca = os.getenv(
                "REDIS_SSL_CA_CERT", cfg.get("redis_ssl_ca_cert", ssl_ca)
            )
        except Exception:
            pass
    options: dict[str, str | int | bool] = {
        "host": host,
        "port": port,
        "decode_responses": True,
    }
    if password:
        options["password"] = password
    if str(use_ssl).lower() in {"1", "true", "yes"}:
        options["ssl"] = True
        if ssl_ca:
            options["ssl_ca_certs"] = ssl_ca
        if ssl_cert:
            options["ssl_certfile"] = ssl_cert
        if ssl_key:
            options["ssl_keyfile"] = ssl_key
    return redis.Redis(**options)
