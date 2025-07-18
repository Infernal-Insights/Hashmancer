import os
import json
import logging
import time
import uuid
import redis
from redis_utils import get_redis
from event_logger import log_error

JOB_STREAM = os.getenv("JOB_STREAM", "jobs")
HTTP_GROUP = os.getenv("HTTP_GROUP", "http-workers")

r = get_redis()


def worker_counts():
    """Return (high_bw, low_bw) worker counts based on stored specs."""
    high = low = 0
    try:
        for key in r.scan_iter("worker:*"):
            info = r.hgetall(key)
            specs_json = info.get("specs")
            if not specs_json:
                continue
            try:
                specs = json.loads(specs_json)
                if isinstance(specs, dict) and "gpus" in specs:
                    specs = specs["gpus"]
            except json.JSONDecodeError:
                specs = []
            if any(g.get("pci_link_width", 0) >= 8 for g in specs):
                high += 1
            else:
                low += 1
    except redis.exceptions.RedisError as e:
        log_error("orchestrator", "system", "SRED", "Redis unavailable", e)
        return 0, 0
    return high, low


def pending_count() -> int:
    """Return number of unacknowledged jobs in the stream for HTTP_GROUP."""
    try:
        try:
            info = r.xpending(JOB_STREAM, HTTP_GROUP)
        except redis.exceptions.ResponseError:
            # group may not exist yet
            r.xgroup_create(JOB_STREAM, HTTP_GROUP, id="0", mkstream=True)
            info = {"pending": 0}
        if isinstance(info, dict):
            return int(info.get("pending", 0))
        return int(info[0]) if info else 0
    except redis.exceptions.RedisError as e:
        log_error("orchestrator", "system", "SRED", "Redis unavailable", e)
        return 0


def dispatch_batches():
    """Prefetch batches from batch:queue into the job stream."""
    try:
        high, low = worker_counts()
        backlog_target = high + low + 2
        pending = pending_count()
        while pending < backlog_target:
            batch_id = r.rpop("batch:queue")
            if not batch_id:
                break
            batch = r.hgetall(f"batch:{batch_id}")
            if not batch:
                continue

            attack = "mask"
            if batch.get("wordlist") and batch.get("mask"):
                attack = "hybrid"
            elif batch.get("wordlist"):
                attack = "dict"

            task_id = str(uuid.uuid4())
            r.hset(
                f"job:{task_id}",
                mapping={
                    "hashes": batch.get("hashes", "[]"),
                    "mask": batch.get("mask", ""),
                    "wordlist": batch.get("wordlist", ""),
                    "attack_mode": attack,
                    "status": "queued",
                },
            )
            r.expire(f"job:{task_id}", 3600)
            r.xadd(JOB_STREAM, {"job_id": task_id})
            pending += 1
    except redis.exceptions.RedisError as e:
        log_error("orchestrator", "system", "SRED", "Redis unavailable", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    while True:
        dispatch_batches()
        time.sleep(1)
