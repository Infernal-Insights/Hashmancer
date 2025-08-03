"""Utilities for configuring Redis connections from environment variables.

This module centralizes the logic for reading ``REDIS_*`` environment
variables including optional SSL certificates and passwords.  It exposes two
helpers:

``redis_options_from_env`` – returns a dictionary of keyword arguments that can
be passed to :class:`redis.Redis`.
``redis_from_env`` – returns a configured :class:`redis.Redis` instance.

Both helpers respect the common ``*_FILE`` variants which allow sensitive values
such as passwords or certificate PEM data to be provided via a file path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import redis

# ---------------------------------------------------------------------------
# helper functions

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
    fd, tmp_path = tempfile.mkstemp()
    with os.fdopen(fd, "w") as tmp:
        tmp.write(value)
    return tmp_path

# ---------------------------------------------------------------------------
# public API

def redis_options_from_env(**overrides: Any) -> Dict[str, Any]:
    """Build a dictionary of ``redis.Redis`` keyword arguments.

    Environment variables are used as defaults but can be overridden by keyword
    arguments.
    """

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
    """Return a :class:`redis.Redis` instance configured from the environment."""

    return redis.Redis(**redis_options_from_env(**overrides))
