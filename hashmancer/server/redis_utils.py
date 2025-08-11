"""Centralized Redis connection utilities."""

import os
import json
import redis
from typing import Any, Dict
from pathlib import Path
import tempfile

from .app.config import CONFIG_FILE


def _read_secret(var: str) -> str | None:
    """Return the value of ``var`` or the contents of ``var_FILE``."""
    file_var = f"{var}_FILE"
    if file_path := os.getenv(file_var):
        try:
            return Path(file_path).read_text().strip()
        except OSError:
            return None
    return os.getenv(var)


# Global registry to track temporary SSL files for cleanup
_TEMP_SSL_FILES: list[str] = []

def _resolve_ssl_file(var: str) -> str | None:
    """Resolve a certificate/key environment variable to a file path."""
    file_var = f"{var}_FILE"
    if file_path := os.getenv(file_var):
        return file_path

    value = os.getenv(var)
    if not value:
        return None

    path = Path(value)
    if path.exists():
        return str(path)

    # assume the value is raw certificate/key data
    fd, tmp_path = tempfile.mkstemp(suffix='.pem', prefix=f'{var.lower()}_')
    with os.fdopen(fd, "w") as tmp:
        tmp.write(value)
    
    # Track temp file for cleanup
    _TEMP_SSL_FILES.append(tmp_path)
    return tmp_path


def cleanup_temp_ssl_files() -> None:
    """Clean up temporary SSL certificate files."""
    for filepath in _TEMP_SSL_FILES:
        try:
            os.unlink(filepath)
        except OSError:
            pass  # File may already be deleted
    _TEMP_SSL_FILES.clear()


def redis_options_from_env(**overrides: Any) -> Dict[str, Any]:
    """Build a dictionary of Redis keyword arguments from environment and config."""
    # Load from config file if it exists
    config_overrides = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            config_overrides = {
                "host": cfg.get("redis_host"),
                "port": cfg.get("redis_port"),
                "password": cfg.get("redis_password"),
                "ssl": cfg.get("redis_ssl"),
                "ssl_cert": cfg.get("redis_ssl_cert"),
                "ssl_key": cfg.get("redis_ssl_key"),
                "ssl_ca_cert": cfg.get("redis_ssl_ca_cert"),
            }
            # Remove None values
            config_overrides = {k: v for k, v in config_overrides.items() if v is not None}
        except Exception:
            pass

    host = overrides.get("host") or os.getenv("REDIS_HOST") or config_overrides.get("host", "localhost")
    port = int(overrides.get("port") or os.getenv("REDIS_PORT") or config_overrides.get("port", 6379))
    
    password = overrides.get("password")
    if password is None:
        password = _read_secret("REDIS_PASSWORD") or config_overrides.get("password")

    ssl = overrides.get("ssl")
    if ssl is None:
        ssl = os.getenv("REDIS_SSL") or config_overrides.get("ssl", "0")
    ssl = str(ssl).lower() in {"1", "true", "yes"}

    ssl_cert = overrides.get("ssl_cert")
    if ssl_cert is None:
        ssl_cert = _resolve_ssl_file("REDIS_SSL_CERT") or config_overrides.get("ssl_cert")
    ssl_key = overrides.get("ssl_key") 
    if ssl_key is None:
        ssl_key = _resolve_ssl_file("REDIS_SSL_KEY") or config_overrides.get("ssl_key")
    ssl_ca_cert = overrides.get("ssl_ca_cert")
    if ssl_ca_cert is None:
        ssl_ca_cert = _resolve_ssl_file("REDIS_SSL_CA_CERT") or config_overrides.get("ssl_ca_cert")

    opts: Dict[str, Any] = {
        "host": host,
        "port": port,
        "decode_responses": overrides.get("decode_responses", True),
    }
    if password:
        opts["password"] = password
    if ssl:
        opts["ssl"] = True
        if ssl_ca_cert:
            opts["ssl_ca_certs"] = ssl_ca_cert
        if ssl_cert:
            opts["ssl_certfile"] = ssl_cert
        if ssl_key:
            opts["ssl_keyfile"] = ssl_key
    return opts


def get_redis(**overrides: Any) -> redis.Redis:
    """Return a Redis connection using environment variables and config file."""
    return redis.Redis(**redis_options_from_env(**overrides))
