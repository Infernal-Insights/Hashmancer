"""Shared utilities for Hashmancer components."""

import os
import sys

SERVER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "Server")
if SERVER_PATH not in sys.path:
    sys.path.insert(0, SERVER_PATH)

from server_utils import redis_manager  # type: ignore

__all__ = ["redis_manager"]
