import redis
import json
import uuid
import time

r = redis.Redis(host="localhost", port=6379, decode_responses=True)


def store_batch(hashes, mask="", wordlist="", ttl=1800, target="any"):
    batch_id = str(uuid.uuid4())
    r.hset(
        f"batch:{batch_id}",
        mapping={
            "hashes": json.dumps(hashes),
            "mask": mask,
            "wordlist": wordlist,
            "created": int(time.time()),
            "target": json.dumps(target),
            "status": "queued",
        },
    )
    r.expire(f"batch:{batch_id}", ttl)
    r.lpush("batch:queue", batch_id)
    return batch_id


def get_next_batch():
    batch_id = r.rpop("batch:queue")
    if not batch_id:
        return None
    data = r.hgetall(f"batch:{batch_id}")
    data["batch_id"] = batch_id
    return data


def requeue_batch(batch_id):
    r.lpush("batch:queue", batch_id)
