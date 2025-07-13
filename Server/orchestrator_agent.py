import os
import json
import logging
import time
import uuid
import redis

LOW_QUEUE = os.getenv("LOW_QUEUE", "task:low")
HIGH_QUEUE = os.getenv("HIGH_QUEUE", "task:high")

r = redis.Redis(host="localhost", port=6379, decode_responses=True)


def worker_counts():
    """Return (high_bw, low_bw) worker counts based on stored specs."""
    high = low = 0
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
    return high, low


def dispatch_batches():
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

    high_workers, low_workers = worker_counts()
    high_load = r.llen(HIGH_QUEUE) / max(high_workers, 1)
    low_load = r.llen(LOW_QUEUE) / max(low_workers, 1)

    if attack in {"dict", "hybrid"}:
        queue = HIGH_QUEUE if high_workers else LOW_QUEUE
    else:
        queue = HIGH_QUEUE if high_load < low_load and high_workers else LOW_QUEUE

    task_id = str(uuid.uuid4())
    r.hset(
        f"task:{task_id}",
        mapping={
            "hashes": batch.get("hashes", "[]"),
            "mask": batch.get("mask", ""),
            "wordlist": batch.get("wordlist", ""),
            "attack_mode": attack,
        },
    )
    r.expire(f"task:{task_id}", 3600)
    r.rpush(queue, task_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    while True:
        dispatch_batches()
        time.sleep(1)
