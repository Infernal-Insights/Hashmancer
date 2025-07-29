import os
import json
import logging
import time
import uuid
import redis
import base64
import gzip
import zlib
import hashlib
import re
from .redis_utils import get_redis
from hashmancer.utils.event_logger import log_error
from . import wordlist_db
from .pattern_stats import generate_mask, TOKEN_RE
from .pattern_utils import is_valid_word
from hashmancer.darkling import charsets
from hashmancer.server.server_utils import redis_manager

try:  # optional local LLM orchestrator
    from llm_orchestrator import LLMOrchestrator  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LLMOrchestrator = None

_LLM = None
if LLMOrchestrator and os.getenv("LLM_MODEL_PATH"):
    try:
        _LLM = LLMOrchestrator(os.getenv("LLM_MODEL_PATH"))
    except Exception as e:  # pragma: no cover - optional component
        logging.warning("LLMOrchestrator disabled: %s", e)

JOB_STREAM = os.getenv("JOB_STREAM", "jobs")
HTTP_GROUP = os.getenv("HTTP_GROUP", "http-workers")
LOW_BW_JOB_STREAM = os.getenv("LOW_BW_JOB_STREAM", "darkling-jobs")
LOW_BW_GROUP = os.getenv("LOW_BW_GROUP", "darkling-workers")

r = get_redis()


def is_already_cracked(hash_str: str) -> bool:
    """Return True if the given hash exists in the found mapping."""
    try:
        hexists = getattr(r, "hexists", None)
        if hexists:
            return bool(hexists("found:map", hash_str))
        return False
    except redis.exceptions.RedisError:
        return False


def get_cracked_password(hash_str: str) -> str | None:
    """Return the password for a previously cracked hash, if any."""
    try:
        hget = getattr(r, "hget", None)
        if hget:
            return hget("found:map", hash_str)
        return None
    except redis.exceptions.RedisError:
        return None


# Mapping between pattern tokens produced by pattern_stats and mask charset
# identifiers used by the darkling engine.
TOKEN_TO_ID = {
    "$U": "?1",
    "$l": "?2",
    "$d": "?3",
    "$s": "?4",
    "$c": "?5",
    "$e": "?6",
}

# Charsets referenced by those identifiers. These are serialized into the job
# so low-bandwidth workers can load the correct lookup tables.
ID_TO_CHARSET = {
    "?1": charsets.ENGLISH_UPPER,
    "?2": charsets.ENGLISH_LOWER,
    "?3": "0123456789",
    "?4": charsets.COMMON_SYMBOLS,
    "?5": charsets.COMMON_SYMBOLS,
    "?6": charsets.EMOJI,
}


MASK_ID_RE = re.compile(r"\?[1-6]")


def build_mask_charsets(lang: str | None = None) -> dict[str, str]:
    """Return a mask charset map for the given language."""
    if not lang:
        lang = "English"
    lang_key = lang.replace("-", "_").upper()
    upper = getattr(charsets, f"{lang_key}_UPPER", charsets.ENGLISH_UPPER)
    lower = getattr(charsets, f"{lang_key}_LOWER", charsets.ENGLISH_LOWER)
    mapping = ID_TO_CHARSET.copy()
    mapping["?1"] = upper
    mapping["?2"] = lower
    return mapping


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
    """Return a list of (pci_link_width, hashrate) tuples for each GPU."""
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
            name = os.path.basename(path)
            comp = zlib.compressobj(wbits=31)
            pipe = r.pipeline()
            leftover = b""
            first = True
            queued = 0
            for chunk in wordlist_db.stream_wordlist(name):
                if not chunk:
                    break
                data = comp.compress(chunk)
                if not data:
                    continue
                leftover += data
                encode_len = (len(leftover) // 3) * 3
                if encode_len:
                    encoded = base64.b64encode(leftover[:encode_len]).decode()
                    if first:
                        pipe.set(redis_key, encoded)
                        first = False
                    else:
                        pipe.append(redis_key, encoded)
                    queued += 1
                    if queued % 100 == 0:
                        pipe.execute()
                    leftover = leftover[encode_len:]
            leftover += comp.flush()
            if leftover:
                encoded = base64.b64encode(leftover).decode()
                if first:
                    pipe.set(redis_key, encoded)
                else:
                    pipe.append(redis_key, encoded)
            pipe.execute()
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


def average_benchmark_rate() -> float:
    """Return the average MD5 benchmark rate across GPUs."""
    rates = []
    try:
        for key in r.scan_iter("benchmark:*"):
            if key.startswith("benchmark_total:"):
                continue
            info = r.hgetall(key)
            try:
                rate = float(info.get("MD5") or 0)
            except (TypeError, ValueError):
                rate = 0.0
            if rate > 0:
                rates.append(rate)
    except redis.exceptions.RedisError as e:
        log_error("orchestrator", "system", "SRED", "Redis unavailable", e)
        return 0.0
    if not rates:
        return 0.0
    return sum(rates) / len(rates)


def estimate_keyspace(mask: str, charset_map: dict[str, str]) -> int:
    """Estimate keyspace for a mask given its charsets."""
    tokens = MASK_ID_RE.findall(mask)
    if not tokens:
        return 0
    space = 1
    for t in tokens:
        cs = charset_map.get(t, "")
        space *= max(len(cs), 1)
    return space


def compute_batch_range(gpu_rate: float, keyspace: int) -> tuple[int, int]:
    """Return a start/end range scaled by the GPU benchmark rate."""
    base = 1000
    if keyspace <= 0:
        return 0, base
    if gpu_rate <= 0:
        end = min(base, keyspace)
        return 0, end

    scale = max(gpu_rate / 10.0, 1.0)
    end = int(base * scale)
    if keyspace < end:
        end = keyspace
    return 0, end


def dispatch_batches(lang: str = "English"):
    """Prefetch batches from batch:queue into one of the job streams."""
    try:
        backlog_target = compute_backlog_target()
        pending_high = pending_count(JOB_STREAM, HTTP_GROUP)
        pending_low = pending_count(LOW_BW_JOB_STREAM, LOW_BW_GROUP)
        darkling = any_darkling_workers()

        while (pending_high < backlog_target) or (
            darkling and pending_low < backlog_target
        ):
            batch_id = None
            prio = r.zrevrange("batch:prio", 0, 0)
            if prio:
                batch_id = prio[0]
                r.zrem("batch:prio", batch_id)
                r.lrem("batch:queue", 0, batch_id)
            else:
                batch_id = r.rpop("batch:queue")
            if not batch_id:
                break
            batch = r.hgetall(f"batch:{batch_id}")
            if not batch:
                continue

            try:
                hashes = json.loads(batch.get("hashes", "[]"))
            except Exception:
                hashes = []
            hashes = [h for h in hashes if not is_already_cracked(h)]
            if not hashes:
                continue
            batch["hashes"] = json.dumps(hashes)

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
                "hash_mode": batch.get("hash_mode", "0"),
                "attack_mode": attack,
                "status": "queued",
            }

            use_llm = _LLM is not None

            if use_llm:
                try:
                    stream_choice = _LLM.choose_job_stream(
                        job_data, pending_high, pending_low
                    )
                except Exception:  # pragma: no cover - optional component
                    stream_choice = "high"
                try:
                    start, end = _LLM.suggest_batch_size(job_data)
                    job_data.update({"start": start, "end": end})
                except Exception:
                    pass
            else:
                if attack == "mask" and darkling and pending_low < backlog_target:
                    stream_choice = "low"
                else:
                    stream_choice = "high"

            if stream_choice == "low":
                if attack != "mask":
                    # transform into a basic mask attack for darkling workers
                    mask_length = 8
                    if batch.get("wordlist"):
                        try:
                            lengths = []
                            with open(
                                batch["wordlist"],
                                "r",
                                encoding="utf-8",
                                errors="ignore",
                            ) as f:
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
                    job_data.update(
                        {
                            "mask": mask,
                            "wordlist": "",
                            "wordlist_key": "",
                            "attack_mode": "mask",
                        }
                    )

                cs_map = build_mask_charsets(lang)
                job_data["mask_charsets"] = json.dumps(cs_map)
                keyspace = estimate_keyspace(job_data.get("mask", ""), cs_map)
                rate = average_benchmark_rate()
                if not use_llm:
                    start, end = compute_batch_range(rate, keyspace)
                    job_data.update({"start": start, "end": end})
                task_id = str(uuid.uuid4())
                r.hset(f"job:{task_id}", mapping=job_data)
                r.expire(f"job:{task_id}", 3600)
                if "start" in job_data and "end" in job_data:
                    redis_manager.queue_range(
                        batch_id, int(job_data["start"]), int(job_data["end"])
                    )
                r.xadd(LOW_BW_JOB_STREAM, {"job_id": task_id})
                pending_low += 1
                continue

            task_id = str(uuid.uuid4())
            r.hset(f"job:{task_id}", mapping=job_data)
            r.expire(f"job:{task_id}", 3600)
            if "start" in job_data and "end" in job_data:
                redis_manager.queue_range(
                    batch_id, int(job_data["start"]), int(job_data["end"])
                )
            r.xadd(JOB_STREAM, {"job_id": task_id})
            pending_high += 1

            if (
                not use_llm
                and darkling
                and attack != "mask"
                and pending_low < backlog_target
            ):
                # transform into a basic mask attack for darkling workers
                d_id = str(uuid.uuid4())
                transformed = job_data.copy()

                mask_length = 8
                if batch.get("wordlist"):
                    try:
                        lengths = []
                        with open(
                            batch["wordlist"], "r", encoding="utf-8", errors="ignore"
                        ) as f:
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

                cs_map = build_mask_charsets(lang)
                transformed.update(
                    {
                        "mask": mask,
                        "wordlist": "",
                        "wordlist_key": "",
                        "attack_mode": "mask",
                        "mask_charsets": json.dumps(cs_map),
                    }
                )
                keyspace = estimate_keyspace(mask, cs_map)
                rate = average_benchmark_rate()
                start, end = compute_batch_range(rate, keyspace)
                transformed.update({"start": start, "end": end})
                r.hset(f"job:{d_id}", mapping=transformed)
                r.expire(f"job:{d_id}", 3600)
                redis_manager.queue_range(batch_id, int(start), int(end))
                r.xadd(LOW_BW_JOB_STREAM, {"job_id": d_id})
                pending_low += 1
    except redis.exceptions.RedisError as e:
        log_error("orchestrator", "system", "SRED", "Redis unavailable", e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dispatch batches to workers")
    parser.add_argument(
        "--lang", default="English", help="language for alphabet charsets"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    while True:
        dispatch_batches(args.lang)
        time.sleep(1)
