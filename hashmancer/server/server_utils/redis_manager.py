import redis
from ..redis_utils import get_redis
import json
import uuid
import time
import logging
from typing import Any, Optional
from .. import orchestrator_agent

# Connection pool for Redis operations
_redis_pool: Optional[redis.ConnectionPool] = None
r = get_redis()


def _get_redis_with_retry() -> redis.Redis:
    """Get Redis connection with retry logic."""
    global _redis_pool, r
    if _redis_pool is None:
        from ..redis_utils import redis_options_from_env
        opts = redis_options_from_env()
        _redis_pool = redis.ConnectionPool(**opts)
    
    try:
        return redis.Redis(connection_pool=_redis_pool)
    except redis.exceptions.ConnectionError:
        # Fallback to module-level Redis instance
        return r


def _redis_retry(func, *args, max_retries: int = 3, **kwargs) -> Any:
    """Execute Redis operation with retry logic."""
    last_error = None
    for attempt in range(max_retries):
        try:
            redis_client = _get_redis_with_retry()
            return func(redis_client, *args, **kwargs)
        except redis.exceptions.RedisError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
            continue
    
    logging.error(f"Redis operation failed after {max_retries} attempts: {last_error}")
    raise last_error


def store_batch(
    hashes,
    mask="",
    wordlist="",
    rule="",
    ttl=1800,
    target="any",
    hash_mode="0",
    priority: int = 0,
):
    batch_id = str(uuid.uuid4())
    
    def _store_batch_ops(redis_client, batch_id, hashes, mask, wordlist, rule, ttl, target, hash_mode, priority, keyspace):
        redis_client.hset(
            f"batch:{batch_id}",
            mapping={
                "hashes": json.dumps(hashes),
                "mask": mask,
                "wordlist": wordlist,
                "rule": rule,
                "created": int(time.time()),
                "target": json.dumps(target),
                "status": "queued",
                "hash_mode": str(hash_mode),
                "keyspace": keyspace,
                "priority": int(priority),
            },
        )
        redis_client.expire(f"batch:{batch_id}", ttl)
        redis_client.lpush("batch:queue", batch_id)
        if int(priority) > 0:
            redis_client.zadd("batch:prio", {batch_id: int(priority)})
        for h in hashes:
            redis_client.sadd(f"hash_batches:{h}", batch_id)
            redis_client.expire(f"hash_batches:{h}", ttl)
    
    try:
        keyspace = 0
        if mask:
            try:
                cs_map = orchestrator_agent.build_mask_charsets()
                keyspace = orchestrator_agent.estimate_keyspace(mask, cs_map)
            except Exception:
                keyspace = 0
                
        _redis_retry(_store_batch_ops, batch_id, hashes, mask, wordlist, rule, ttl, target, hash_mode, priority, keyspace)
        return batch_id
    except redis.exceptions.RedisError as e:
        logging.warning("Redis unavailable: %s", e)
        return None


def get_next_batch():
    def _get_next_batch_ops(redis_client):
        batch_id = redis_client.rpop("batch:queue")
        if not batch_id:
            return None
        data = redis_client.hgetall(f"batch:{batch_id}")
        data["batch_id"] = batch_id
        return data
    
    try:
        return _redis_retry(_get_next_batch_ops)
    except redis.exceptions.RedisError as e:
        logging.warning("Redis unavailable: %s", e)
        return None


def requeue_batch(batch_id):
    def _requeue_batch_ops(redis_client, batch_id):
        redis_client.lpush("batch:queue", batch_id)
        prio = redis_client.hget(f"batch:{batch_id}", "priority")
        if prio and int(prio) > 0:
            redis_client.zadd("batch:prio", {batch_id: int(prio)})
    
    try:
        _redis_retry(_requeue_batch_ops, batch_id)
    except redis.exceptions.RedisError as e:
        logging.warning("Redis unavailable: %s", e)


def queue_range(batch_id: str, start: int, end: int) -> None:
    """Record a keyspace range as queued for the batch."""
    def _queue_range_ops(redis_client, batch_id, start, end):
        redis_client.rpush(f"keyspace:queued:{batch_id}", f"{start}:{end}")
    
    try:
        _redis_retry(_queue_range_ops, batch_id, start, end)
    except redis.exceptions.RedisError:
        pass


def complete_range(batch_id: str, start: int, end: int) -> None:
    """Mark a keyspace range as completed for the batch."""
    def _complete_range_ops(redis_client, batch_id, start, end):
        redis_client.rpush(f"keyspace:done:{batch_id}", f"{start}:{end}")
        redis_client.lrem(f"keyspace:queued:{batch_id}", 0, f"{start}:{end}")
    
    try:
        _redis_retry(_complete_range_ops, batch_id, start, end)
    except redis.exceptions.RedisError:
        pass


def update_status(batch_id: str, status: str) -> None:
    """Set the status field for a batch and refresh TTL."""
    def _update_status_ops(redis_client, batch_id, status):
        redis_client.hset(f"batch:{batch_id}", "status", status)
        if status == "processing":
            redis_client.persist(f"batch:{batch_id}")
    
    try:
        _redis_retry(_update_status_ops, batch_id, status)
    except redis.exceptions.RedisError as e:
        logging.warning("Redis unavailable: %s", e)


def get_hash_batches(hash_value: str) -> list[str]:
    """Return batch IDs associated with the given hash."""
    def _get_hash_batches_ops(redis_client, hash_value):
        return list(redis_client.smembers(f"hash_batches:{hash_value}"))
    
    try:
        return _redis_retry(_get_hash_batches_ops, hash_value)
    except redis.exceptions.RedisError as e:
        logging.warning("Redis unavailable: %s", e)
        return []
