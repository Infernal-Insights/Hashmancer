import redis
from redis_utils import get_redis
import json
import uuid
import time
import orchestrator_agent

r = get_redis()


def store_batch(
    hashes,
    mask="",
    wordlist="",
    ttl=1800,
    target="any",
    hash_mode="0",
):
    batch_id = str(uuid.uuid4())
    try:
        keyspace = 0
        if mask:
            try:
                cs_map = orchestrator_agent.build_mask_charsets()
                keyspace = orchestrator_agent.estimate_keyspace(mask, cs_map)
            except Exception:
                keyspace = 0
        r.hset(
            f"batch:{batch_id}",
            mapping={
                "hashes": json.dumps(hashes),
                "mask": mask,
                "wordlist": wordlist,
                "created": int(time.time()),
                "target": json.dumps(target),
                "status": "queued",
                "hash_mode": str(hash_mode),
                "keyspace": keyspace,
            },
        )
        r.expire(f"batch:{batch_id}", ttl)
        r.lpush("batch:queue", batch_id)
        for h in hashes:
            r.sadd(f"hash_batches:{h}", batch_id)
            r.expire(f"hash_batches:{h}", ttl)
    except redis.exceptions.RedisError as e:
        print(f"Redis unavailable: {e}")
        return None
    return batch_id


def get_next_batch():
    try:
        batch_id = r.rpop("batch:queue")
        if not batch_id:
            return None
        data = r.hgetall(f"batch:{batch_id}")
        data["batch_id"] = batch_id
        return data
    except redis.exceptions.RedisError as e:
        print(f"Redis unavailable: {e}")
        return None


def requeue_batch(batch_id):
    try:
        r.lpush("batch:queue", batch_id)
    except redis.exceptions.RedisError as e:
        print(f"Redis unavailable: {e}")


def queue_range(batch_id: str, start: int, end: int) -> None:
    """Record a keyspace range as queued for the batch."""
    try:
        r.rpush(f"keyspace:queued:{batch_id}", f"{start}:{end}")
    except redis.exceptions.RedisError:
        pass


def complete_range(batch_id: str, start: int, end: int) -> None:
    """Mark a keyspace range as completed for the batch."""
    try:
        r.rpush(f"keyspace:done:{batch_id}", f"{start}:{end}")
        r.lrem(f"keyspace:queued:{batch_id}", 0, f"{start}:{end}")
    except redis.exceptions.RedisError:
        pass
