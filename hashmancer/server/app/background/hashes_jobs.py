import asyncio
import json
import os
import redis

from hashmancer.server.server_utils import redis_manager
from hashmancer.utils.event_logger import log_error
from ..config import (
    HASHES_SETTINGS,
    HASHES_POLL_INTERVAL,
    HASHES_ALGORITHMS,
    HASHES_DEFAULT_PRIORITY,
    PREDEFINED_MASKS,
    HASHES_ALGO_PARAMS,
)


async def fetch_and_store_jobs() -> None:
    """Fetch jobs from hashes.com and store filtered results in Redis."""
    try:
        import hashescom_client
        from hashmancer.server import main  # late import for patched r in tests

        r = main.r
        hashes_algorithms = getattr(main, "HASHES_ALGORITHMS", HASHES_ALGORITHMS)
        jobs = await hashescom_client.fetch_jobs()
        for job in jobs:
            algo = str(job.get("algorithmName", "")).lower()
            if hashes_algorithms and algo not in hashes_algorithms:
                continue
            if job.get("currency") != "BTC":
                continue
            try:
                price = float(job.get("pricePerHash", 0))
            except (TypeError, ValueError):
                price = 0.0
            if price <= 0:
                continue
            job_id = job.get("id")
            if job_id is not None:
                r.hset(f"hashes_job:{job_id}", mapping=job)
    except Exception as e:  # pragma: no cover - network errors rarely tested
        log_error("server", "system", "S741", "Hashes.com fetch failed", e)


async def poll_hashes_jobs() -> None:
    """Background loop to periodically poll hashes.com for jobs."""
    while True:
        await fetch_and_store_jobs()
        from hashmancer.server import main
        settings = getattr(main, "HASHES_SETTINGS", HASHES_SETTINGS)
        interval = int(settings.get("hashes_poll_interval", HASHES_POLL_INTERVAL))
        await asyncio.sleep(interval)


async def process_hashes_jobs() -> None:
    """Queue batches for jobs fetched from hashes.com."""
    from hashmancer.server import main  # late import for patched r and verify_hash in tests

    r = main.r
    verify_hash = main.verify_hash
    algo_params = getattr(main, "HASHES_ALGO_PARAMS", HASHES_ALGO_PARAMS)
    predefined_masks = getattr(main, "PREDEFINED_MASKS", PREDEFINED_MASKS)
    default_priority = getattr(main, "HASHES_DEFAULT_PRIORITY", HASHES_DEFAULT_PRIORITY)

    while True:
        try:
            for key in r.scan_iter("hashes_job:*"):
                job = r.hgetall(key)
                if job.get("status") == "processed":
                    continue

                try:
                    hashes = json.loads(job.get("hashes", "[]"))
                except Exception:
                    hashes = []

                algorithm = job.get("algorithmName", "")
                algo_id = job.get("algorithmId")

                known: list[str] = []
                remaining: list[str] = []
                for h in hashes:
                    pw = r.hget("found:map", h)
                    if pw and verify_hash(pw, h, algorithm):
                        known.append(f"{h}:{pw}")
                    else:
                        remaining.append(h)

                mask = job.get("mask", "")
                wordlist = job.get("wordlist", "")
                params = algo_params.get(algorithm.lower(), {})
                mask_len = params.get("mask_length")
                if mask_len:
                    mask_len = int(mask_len)
                    if mask:
                        mask = mask[:mask_len]
                    else:
                        mask = "?a" * mask_len
                rule = params.get("rule", "")
                if known:
                    try:
                        import tempfile
                        import hashescom_client

                        with tempfile.NamedTemporaryFile("w", delete=False) as fh:
                            fh.write("\n".join(known))
                            temp_path = fh.name
                        hashescom_client.upload_founds(algo_id, temp_path)
                        os.unlink(temp_path)
                    except Exception:
                        pass

                batch_id = None
                if remaining:
                    priority = int(job.get("priority", default_priority))
                    batch_id = redis_manager.store_batch(
                        remaining,
                        mask=mask,
                        wordlist=wordlist,
                        rule=rule,
                        priority=priority,
                    )
                    for pm in predefined_masks:
                        redis_manager.store_batch(
                            remaining,
                            mask=pm,
                            wordlist=wordlist,
                            rule=rule,
                            priority=priority + 1,
                        )
                r.hset(key, mapping={"status": "processed", "batch_id": batch_id or ""})
        except redis.exceptions.RedisError as e:
            log_error("server", "system", "SRED", "Redis unavailable", e)
        except Exception as e:
            log_error("server", "system", "S742", "Failed to process hashes jobs", e)

        await asyncio.sleep(30)
