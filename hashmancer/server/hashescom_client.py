import asyncio
import os
import json
from pathlib import Path
import logging

import requests

try:
    import aiohttp
except Exception:  # pragma: no cover - fallback when aiohttp unavailable
    aiohttp = None

# Prefer an environment variable but fall back to the server config
from .app.config import CONFIG_FILE

def _load_api_key() -> str | None:
    key = os.environ.get("HASHES_COM_API_KEY")
    if key:
        return key
    try:
        with CONFIG_FILE.open() as f:
            cfg = json.load(f)
        return cfg.get("hashes_api_key")
    except Exception:
        return None

HASHES_API = _load_api_key()


async def _fetch_jobs_async(url: str, headers: dict | None = None) -> list:
    assert aiohttp is not None
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
    if not data.get("success"):
        return []
    return data["list"]


def _fetch_jobs_sync(url: str, headers: dict | None = None) -> list:
    resp = requests.get(url, timeout=10, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        return []
    return data["list"]


async def fetch_jobs() -> list:
    url = "https://hashes.com/en/api/jobs"
    headers = {"X-Api-Key": HASHES_API} if HASHES_API else None
    try:
        if aiohttp is not None:
            return await _fetch_jobs_async(url, headers)
        return await asyncio.to_thread(_fetch_jobs_sync, url, headers)
    except Exception as e:
        logging.warning("[❌] Hashes.com fetch error: %s", e)
        return []


def upload_founds(algo_id, found_file) -> bool:
    """Upload found hashes and return True if the API confirms success."""
    try:
        url = "https://hashes.com/en/api/founds"
        with open(found_file, "rb") as fh:
            files = {"userfile": fh}
            data = {"algo": str(algo_id), "key": HASHES_API or ""}
            resp = requests.post(url, files=files, data=data, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if not result.get("success"):
            logging.warning("[❌] Upload rejected: %s", result)
            return False
        return True
    except requests.HTTPError as e:
        logging.warning("[❌] Upload error: %s", e)
        return False
    except Exception as e:
        logging.warning("[❌] Upload error: %s", e)
        return False
