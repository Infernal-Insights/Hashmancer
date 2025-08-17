"""Centralized Redis connection utilities - Updated to use Unified Redis Manager."""

import os
import json
import redis
from typing import Any, Dict
from pathlib import Path
import tempfile
import logging

# Import the new unified Redis manager
from .unified_redis import get_redis_manager, redis_connection, get_redis_stats

logger = logging.getLogger(__name__)

# For backward compatibility, keep the config loading functions
try:
    from .app.config import CONFIG_FILE
except ImportError:
    CONFIG_FILE = Path("config.json")


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
    # This function is kept for compatibility but now logs a deprecation warning
    logger.warning("redis_options_from_env() is deprecated. Use unified_redis.RedisConfig.from_env() instead.")
    
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
    """Return a Redis connection using the unified Redis manager.
    
    This function now uses the unified Redis manager for better connection management,
    pooling, and error handling. The overrides parameter is ignored as configuration
    should be done through environment variables or config files.
    """
    if overrides:
        logger.warning("get_redis() overrides are deprecated. Use environment variables or config files.")
    
    try:
        manager = get_redis_manager()
        return manager.get_legacy_sync_client()
    except Exception as e:
        logger.error(f"Failed to get Redis connection: {e}")
        # Fallback to direct connection for extreme compatibility
        try:
            return redis.Redis(**redis_options_from_env(**overrides))
        except Exception as fallback_error:
            logger.error(f"Fallback Redis connection also failed: {fallback_error}")
            raise


# Add convenience functions for the new unified system
def get_redis_with_context():
    """Get Redis connection with proper context management.
    
    Example usage:
        with get_redis_with_context() as redis_conn:
            redis_conn.set("key", "value")
    """
    return redis_connection()


def get_redis_health():
    """Get Redis health status and statistics."""
    try:
        return get_redis_stats()
    except Exception as e:
        return {"error": str(e), "healthy": False}
