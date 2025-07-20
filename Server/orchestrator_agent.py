import os
import json
import logging
import time
import uuid
import redis
import base64
import gzip
import hashlib
from redis_utils import get_redis
from event_logger import log_error
from pattern_stats import generate_mask, TOKEN_RE
from pattern_utils import is_valid_word
from darkling import charsets

JOB_STREAM = os.getenv("JOB_STREAM", "jobs")
HTTP_GROUP = os.getenv("HTTP_GROUP", "http-workers")
LOW_BW_JOB_STREAM = os.getenv("LOW_BW_JOB_STREAM", "darkling-jobs")
LOW_BW_GROUP = os.getenv("LOW_BW_GROUP", "darkling-workers")

r = get_redis()

# Mapping between pattern tokens produced by pattern_stats and mask charset
# identifiers used by the darkling engine.
TOKEN_TO_ID = {"$U": "?1", "$l": "?2", "$d": "?3", "$s": "?4"}

# Charsets referenced by those identifiers. These are serialized into the job
# so low-bandwidth workers can load the correct lookup tables.
ID_TO_CHARSET = {
    "?1": charsets.ENGLISH_UPPER,
    "?2": charsets.ENGLISH_LOWER,
    "?3": "0123456789",
    "?4": charsets.COMMON_SYMBOLS,
}


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


def gpu_metrics() -> list[tuple[int, float]]:
    """Return a list of (pci_width, hashrate) tuples for each GPU."""
    metrics: list[tuple[int, float]] = []
    try:
        for key in r.scan_iter("gpu:*"):
            info = r.hgetall(key)
            width = int(info.get("pci_link_width", 0))
            try:
                rate = float(info.get("hashrate", 0))
            except (TypeError, ValueError):
                rate = 0.0
            metrics.append((width, rate))
    except redis.exceptions.RedisError as e:
        log_error("orchestrator", "system", "SRED", "Redis unavailable", e)
    return metrics


def compute_backlog_target() -> int:
    """Return desired backlog depth based on GPU count and load."""
    backlog = 2  # base
    for width, rate in gpu_metrics():
        if width >= 8:
            backlog += 4
        elif width >= 4:
            backlog += 2
        else:
            backlog += 1
        if rate > 0:
            backlog += 1
    return backlog


def any_darkling_workers() -> bool:
    """Return True if any worker is configured to use the darkling engine."""
    try:
        for key in r.scan_iter("worker:*"):
            if r.hget(key, "low_bw_engine") == "darkling":
                return True
    except redis.exceptions.RedisError as e:
        log_error("orchestrator", "system", "SRED", "Redis unavailable", e)
    return False


def cache_wordlist(path: str) -> str:
    """Store a compressed copy of the wordlist in Redis and return its key."""
    key = hashlib.sha1(path.encode()).hexdigest()
    redis_key = f"wlcache:{key}"
    try:
        if not r.exists(redis_key):
            with open(path, "rb") as f:
                data = gzip.compress(f.read())
            r.set(redis_key, base64.b64encode(data).decode())
    except Exception as e:
        log_error("orchestrator", "system", "SCACHE", "Failed to cache wordlist", e)
    return key


def pending_count(stream: str = JOB_STREAM, group: str = HTTP_GROUP) -> int:
    """Return number of unacknowledged jobs in the given stream."""
    try:
        try:
            info = r.xpending(stream, group)
        except redis.exceptions.ResponseError:
            # group may not exist yet
            r.xgroup_create(stream, group, id="0", mkstream=True)
            info = {"pending": 0}
        if isinstance(info, dict):
            return int(info.get("pending", 0))
        return int(info[0]) if info else 0
    except redis.exceptions.RedisError as e:
        log_error("orchestrator", "system", "SRED", "Redis unavailable", e)
        return 0


def dispatch_batches():
    """Prefetch batches from batch:queue into one of the job streams."""
    try:
        backlog_target = compute_backlog_target()
        pending_high = pending_count(JOB_STREAM, HTTP_GROUP)
        pending_low = pending_count(LOW_BW_JOB_STREAM, LOW_BW_GROUP)
        darkling = any_darkling_workers()

        while (pending_high < backlog_target) or (darkling and pending_low < backlog_target):
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

            wordlist_key = ""
            if batch.get("wordlist"):
                wordlist_key = cache_wordlist(batch["wordlist"])

            job_data = {
                "batch_id": batch_id,
                "hashes": batch.get("hashes", "[]"),
                "mask": batch.get("mask", ""),
                "wordlist": batch.get("wordlist", ""),
                "wordlist_key": wordlist_key,
                "attack_mode": attack,
                "status": "queued",
            }

            # route job depending on attack type
            if attack == "mask" and darkling and pending_low < backlog_target:
                task_id = str(uuid.uuid4())
                job_data["start"] = 0
                job_data["end"] = 1000
                r.hset(f"job:{task_id}", mapping=job_data)
                r.expire(f"job:{task_id}", 3600)
                r.xadd(LOW_BW_JOB_STREAM, {"job_id": task_id})
                pending_low += 1
                continue

            task_id = str(uuid.uuid4())
            r.hset(f"job:{task_id}", mapping=job_data)
            r.expire(f"job:{task_id}", 3600)
            r.xadd(JOB_STREAM, {"job_id": task_id})
            pending_high += 1

            if darkling and attack != "mask" and pending_low < backlog_target:
                # transform into a basic mask attack for darkling workers
                d_id = str(uuid.uuid4())
                transformed = job_data.copy()

                mask_length = 8
                if batch.get("wordlist"):
                    try:
                        lengths = []
                        with open(batch["wordlist"], "r", encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f):
                                if i >= 100:
                                    break
                                word = line.strip()
                                if is_valid_word(word):
                                    lengths.append(len(word))
                        if lengths:
                            mask_length = round(sum(lengths) / len(lengths))
                    except Exception:
                        pass

                pattern = generate_mask(mask_length)
                tokens = TOKEN_RE.findall(pattern)
                mask = "".join(TOKEN_TO_ID.get(t, "?1") for t in tokens)

                transformed.update({
                    "mask": mask,
                    "wordlist": "",
                    "wordlist_key": "",
                    "attack_mode": "mask",
                    "mask_charsets": json.dumps(ID_TO_CHARSET),
                    "start": 0,
                    "end": 1000,
                })
                r.hset(f"job:{d_id}", mapping=transformed)
                r.expire(f"job:{d_id}", 3600)
                r.xadd(LOW_BW_JOB_STREAM, {"job_id": d_id})
                pending_low += 1
    except redis.exceptions.RedisError as e:
        log_error("orchestrator", "system", "SRED", "Redis unavailable", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    while True:
        dispatch_batches()
        time.sleep(1)
