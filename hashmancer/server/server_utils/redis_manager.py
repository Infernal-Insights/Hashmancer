import redis
from ..unified_redis import get_redis_manager, redis_connection, with_redis_sync
import json
import uuid
import time
import logging
from typing import Any, Optional, Dict, List
from .. import orchestrator_agent

logger = logging.getLogger(__name__)

# Use the unified Redis manager
def _get_redis_with_retry() -> redis.Redis:
    """Get Redis connection using the unified manager."""
    try:
        manager = get_redis_manager()
        return manager.get_legacy_sync_client()
    except Exception as e:
        logger.error(f"Failed to get Redis connection: {e}")
        raise


def _redis_retry(func, *args, max_retries: int = 3, **kwargs) -> Any:
    """Execute Redis operation with retry logic using unified manager."""
    try:
        manager = get_redis_manager()
        return manager.execute_with_retry(func, *args, **kwargs)
    except Exception as e:
        logger.error(f"Redis operation failed: {e}")
        raise


@with_redis_sync
def store_batch(
    redis_client,
    hashes,
    mask="",
    wordlist="",
    rule="",
    ttl=1800,
    target="any",
    hash_mode="0",
    priority: int = 0,
):
    """Store a batch of hashes for processing with improved error handling and atomicity."""
    batch_id = str(uuid.uuid4())
    
    try:
        # Calculate keyspace estimate
        keyspace = 0
        if mask:
            try:
                cs_map = orchestrator_agent.build_mask_charsets()
                keyspace = orchestrator_agent.estimate_keyspace(mask, cs_map)
            except Exception as e:
                logger.warning(f"Failed to estimate keyspace: {e}")
                keyspace = 0
        
        # Use pipeline for atomic batch operations
        with redis_client.pipeline(transaction=True) as pipeline:
            # Store batch metadata
            batch_data = {
                "hashes": json.dumps(hashes),
                "mask": mask,
                "wordlist": wordlist,
                "rule": rule,
                "created": int(time.time()),
                "target": json.dumps(target),
                "status": "queued",
                "hash_mode": str(hash_mode),
                "keyspace": str(keyspace),
                "priority": int(priority),
            }
            
            pipeline.hset(f"batch:{batch_id}", mapping=batch_data)
            pipeline.expire(f"batch:{batch_id}", ttl)
            pipeline.lpush("batch:queue", batch_id)
            
            # Add to priority queue if needed
            if int(priority) > 0:
                pipeline.zadd("batch:prio", {batch_id: int(priority)})
            
            # Associate hashes with batch
            for h in hashes:
                pipeline.sadd(f"hash_batches:{h}", batch_id)
                pipeline.expire(f"hash_batches:{h}", ttl)
            
            # Execute all operations atomically
            pipeline.execute()
            
        logger.info(f"Stored batch {batch_id} with {len(hashes)} hashes, priority {priority}")
        return batch_id
        
    except Exception as e:
        logger.error(f"Failed to store batch: {e}")
        return None


@with_redis_sync
def get_next_batch(redis_client):
    """Get the next batch from the queue with proper error handling."""
    try:
        batch_id = redis_client.rpop("batch:queue")
        if not batch_id:
            return None
            
        data = redis_client.hgetall(f"batch:{batch_id}")
        if not data:
            logger.warning(f"Batch {batch_id} metadata not found")
            return None
            
        data["batch_id"] = batch_id
        logger.debug(f"Retrieved batch {batch_id} from queue")
        return data
        
    except Exception as e:
        logger.error(f"Failed to get next batch: {e}")
        return None


@with_redis_sync
def requeue_batch(redis_client, batch_id: str):
    """Requeue a batch with proper priority handling."""
    try:
        with redis_client.pipeline(transaction=True) as pipeline:
            pipeline.lpush("batch:queue", batch_id)
            
            # Check if batch has priority
            prio = redis_client.hget(f"batch:{batch_id}", "priority")
            if prio and int(prio) > 0:
                pipeline.zadd("batch:prio", {batch_id: int(prio)})
            
            pipeline.execute()
            
        logger.info(f"Requeued batch {batch_id}")
        
    except Exception as e:
        logger.error(f"Failed to requeue batch {batch_id}: {e}")


@with_redis_sync
def queue_range(redis_client, batch_id: str, start: int, end: int) -> None:
    """Record a keyspace range as queued for the batch."""
    try:
        redis_client.rpush(f"keyspace:queued:{batch_id}", f"{start}:{end}")
        logger.debug(f"Queued range {start}:{end} for batch {batch_id}")
    except Exception as e:
        logger.warning(f"Failed to queue range for batch {batch_id}: {e}")


@with_redis_sync
def complete_range(redis_client, batch_id: str, start: int, end: int) -> None:
    """Mark a keyspace range as completed for the batch."""
    try:
        with redis_client.pipeline(transaction=True) as pipeline:
            pipeline.rpush(f"keyspace:done:{batch_id}", f"{start}:{end}")
            pipeline.lrem(f"keyspace:queued:{batch_id}", 0, f"{start}:{end}")
            pipeline.execute()
            
        logger.debug(f"Completed range {start}:{end} for batch {batch_id}")
        
    except Exception as e:
        logger.warning(f"Failed to complete range for batch {batch_id}: {e}")


@with_redis_sync
def update_status(redis_client, batch_id: str, status: str) -> None:
    """Set the status field for a batch and manage TTL."""
    try:
        with redis_client.pipeline(transaction=True) as pipeline:
            pipeline.hset(f"batch:{batch_id}", "status", status)
            pipeline.hset(f"batch:{batch_id}", "updated", int(time.time()))
            
            if status == "processing":
                # Remove TTL for processing batches
                pipeline.persist(f"batch:{batch_id}")
            elif status == "completed" or status == "failed":
                # Set shorter TTL for completed/failed batches
                pipeline.expire(f"batch:{batch_id}", 3600)  # 1 hour
            
            pipeline.execute()
            
        logger.info(f"Updated batch {batch_id} status to {status}")
        
    except Exception as e:
        logger.error(f"Failed to update status for batch {batch_id}: {e}")


@with_redis_sync
def get_hash_batches(redis_client, hash_value: str) -> list[str]:
    """Return batch IDs associated with the given hash."""
    try:
        batch_ids = list(redis_client.smembers(f"hash_batches:{hash_value}"))
        logger.debug(f"Found {len(batch_ids)} batches for hash {hash_value}")
        return batch_ids
        
    except Exception as e:
        logger.error(f"Failed to get batches for hash {hash_value}: {e}")
        return []


@with_redis_sync 
def get_batch_status(redis_client, batch_id: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive status information for a batch."""
    try:
        batch_data = redis_client.hgetall(f"batch:{batch_id}")
        if not batch_data:
            return None
        
        # Get queue information
        queued_ranges = redis_client.lrange(f"keyspace:queued:{batch_id}", 0, -1)
        completed_ranges = redis_client.lrange(f"keyspace:done:{batch_id}", 0, -1)
        
        return {
            "batch_id": batch_id,
            "status": batch_data.get("status", "unknown"),
            "created": batch_data.get("created"),
            "updated": batch_data.get("updated"),
            "priority": int(batch_data.get("priority", 0)),
            "hash_count": len(json.loads(batch_data.get("hashes", "[]"))),
            "keyspace": batch_data.get("keyspace", "0"),
            "queued_ranges": len(queued_ranges),
            "completed_ranges": len(completed_ranges),
            "mask": batch_data.get("mask", ""),
            "wordlist": batch_data.get("wordlist", ""),
            "rule": batch_data.get("rule", ""),
        }
        
    except Exception as e:
        logger.error(f"Failed to get batch status for {batch_id}: {e}")
        return None


@with_redis_sync
def cleanup_expired_batches(redis_client, max_age_hours: int = 24) -> int:
    """Clean up old batch data to prevent Redis memory bloat."""
    try:
        cutoff_time = int(time.time()) - (max_age_hours * 3600)
        cleaned_count = 0
        
        # Find batches to clean up
        for key in redis_client.scan_iter("batch:*"):
            try:
                created = redis_client.hget(key, "created")
                if created and int(created) < cutoff_time:
                    batch_id = key.split(":", 1)[1]
                    
                    with redis_client.pipeline(transaction=True) as pipeline:
                        # Remove batch data
                        pipeline.delete(key)
                        pipeline.delete(f"keyspace:queued:{batch_id}")
                        pipeline.delete(f"keyspace:done:{batch_id}")
                        
                        # Remove from queues
                        pipeline.lrem("batch:queue", 0, batch_id)
                        pipeline.zrem("batch:prio", batch_id)
                        
                        pipeline.execute()
                        
                    cleaned_count += 1
                    
            except Exception as e:
                logger.warning(f"Error cleaning batch {key}: {e}")
                continue
        
        logger.info(f"Cleaned up {cleaned_count} expired batches")
        return cleaned_count
        
    except Exception as e:
        logger.error(f"Failed to cleanup expired batches: {e}")
        return 0
