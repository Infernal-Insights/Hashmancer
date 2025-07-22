import redis
from redis_utils import get_redis
import json
import uuid
import time

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
            },
        )
        r.expire(f"batch:{batch_id}", ttl)
        r.lpush("batch:queue", batch_id)
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
