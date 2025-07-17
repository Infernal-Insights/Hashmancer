import os
import json
import logging
import time
import uuid
import redis
from redis_utils import get_redis

JOB_STREAM = os.getenv("JOB_STREAM", "jobs")

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
        logging.error(f"Redis unavailable: {e}")
        return 0, 0
    return high, low


def dispatch_batches():
    try:
        batch_id = r.rpop("batch:queue")
        if not batch_id:
            return
        batch = r.hgetall(f"batch:{batch_id}")
        if not batch:
            return

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
    except redis.exceptions.RedisError as e:
        logging.error(f"Redis unavailable: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    while True:
        dispatch_batches()
        time.sleep(1)
