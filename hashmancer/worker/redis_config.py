"""Utilities for configuring Redis connections from environment variables.

This module has been updated to use the unified Redis manager for better
connection management, pooling, and error handling across the entire application.
"""

from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict

import redis

logger = logging.getLogger(__name__)

# Try to import the unified Redis manager
try:
    from hashmancer.server.unified_redis import (
        get_redis_manager, 
        redis_connection, 
        get_redis, 
        get_redis_stats
    )
    UNIFIED_REDIS_AVAILABLE = True
except ImportError:
    UNIFIED_REDIS_AVAILABLE = False
    logger.warning("Unified Redis manager not available, using legacy Redis connection")

# ---------------------------------------------------------------------------
# helper functions (kept for compatibility)

def _read_secret(var: str) -> str | None:
    """Return the value of ``var`` or the contents of ``var_FILE``.

    This is commonly used for reading passwords stored in Docker secrets or
    other mounted files.  If neither is set, ``None`` is returned.
    """

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
    """Resolve a certificate/key environment variable to a file path.

    The environment variable may either contain a path to an existing file or
    the PEM data itself.  A ``*_FILE`` variant is also supported.  When PEM data
    is provided directly, it is written to a temporary file and the path to that
    file is returned.
    """

    # First handle the *_FILE variant
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

# ---------------------------------------------------------------------------
# public API

def redis_options_from_env(**overrides: Any) -> Dict[str, Any]:
    """Build a dictionary of ``redis.Redis`` keyword arguments.
    
    NOTE: This function is deprecated. The unified Redis manager should be used instead.
    """
    logger.warning("redis_options_from_env() is deprecated. Use the unified Redis manager instead.")

    host = overrides.get("host") or os.getenv("REDIS_HOST", "localhost")
    port = int(overrides.get("port") or os.getenv("REDIS_PORT", 6379))
    password = overrides.get("password")
    if password is None:
        password = _read_secret("REDIS_PASSWORD")

    ssl = overrides.get("ssl")
    if ssl is None:
        ssl = os.getenv("REDIS_SSL", "0")
    ssl = str(ssl).lower() in {"1", "true", "yes"}

    ssl_cert = overrides.get("ssl_cert")
    if ssl_cert is None:
        ssl_cert = _resolve_ssl_file("REDIS_SSL_CERT")
    ssl_key = overrides.get("ssl_key")
    if ssl_key is None:
        ssl_key = _resolve_ssl_file("REDIS_SSL_KEY")
    ssl_ca_cert = overrides.get("ssl_ca_cert")
    if ssl_ca_cert is None:
        ssl_ca_cert = _resolve_ssl_file("REDIS_SSL_CA_CERT")

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


def redis_from_env(**overrides: Any) -> redis.Redis:
    """Return a :class:`redis.Redis` instance configured from the environment.
    
    This function now uses the unified Redis manager when available for better
    connection management and error handling.
    """
    
    if UNIFIED_REDIS_AVAILABLE:
        try:
            if overrides:
                logger.warning("redis_from_env() overrides are ignored when using unified Redis manager")
            return get_redis()
        except Exception as e:
            logger.error(f"Failed to get Redis connection from unified manager: {e}")
            # Fall back to legacy method
    
    # Legacy fallback
    logger.info("Using legacy Redis connection method")
    return redis.Redis(**redis_options_from_env(**overrides))


def get_redis_connection():
    """Get Redis connection with proper context management.
    
    Example usage:
        with get_redis_connection() as redis_conn:
            redis_conn.set("key", "value")
    """
    if UNIFIED_REDIS_AVAILABLE:
        return redis_connection()
    else:
        # Fallback context manager for legacy connections
        from contextlib import contextmanager
        
        @contextmanager
        def legacy_redis_connection():
            conn = redis_from_env()
            try:
                yield conn
            finally:
                try:
                    conn.close()
                except:
                    pass
        
        return legacy_redis_connection()


def get_redis_health() -> Dict[str, Any]:
    """Get Redis health status and statistics."""
    if UNIFIED_REDIS_AVAILABLE:
        try:
            return get_redis_stats()
        except Exception as e:
            return {"error": str(e), "healthy": False}
    else:
        # Simple health check for legacy connections
        try:
            with get_redis_connection() as conn:
                conn.ping()
            return {"healthy": True, "legacy_mode": True}
        except Exception as e:
            return {"healthy": False, "error": str(e), "legacy_mode": True}


def test_redis_connection() -> bool:
    """Test if Redis connection is working."""
    try:
        with get_redis_connection() as conn:
            conn.ping()
        return True
    except Exception as e:
        logger.error(f"Redis connection test failed: {e}")
        return False
