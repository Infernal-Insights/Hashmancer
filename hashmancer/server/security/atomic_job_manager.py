#!/usr/bin/env python3
"""
Atomic Job Manager for Race-Condition-Free Job Processing
Thread-safe job assignment and batch management using Redis Lua scripts
"""

import json
import time
import uuid
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from ..redis_utils import get_redis

logger = logging.getLogger(__name__)

@dataclass
class JobInfo:
    """Job information structure."""
    job_id: str
    batch_id: str
    status: str  # pending, assigned, processing, completed, failed
    assigned_worker: Optional[str]
    created_at: float
    updated_at: float
    priority: int
    data: Dict[str, Any]

@dataclass
class BatchInfo:
    """Batch information structure."""
    batch_id: str
    status: str  # pending, processing, completed, failed
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    created_at: float
    updated_at: float
    priority: int
    metadata: Dict[str, Any]

class AtomicJobManager:
    """Thread-safe job manager using Redis atomic operations."""
    
    def __init__(self):
        self.redis = get_redis()
        self._init_lua_scripts()
    
    def _init_lua_scripts(self):
        """Initialize Lua scripts for atomic operations."""
        
        # Atomic job assignment script
        self.assign_job_script = self.redis.register_script("""
            local queue_key = KEYS[1]
            local job_prefix = KEYS[2]
            local worker_id = ARGV[1]
            local current_time = ARGV[2]
            
            -- Get next job from priority queue
            local batch_data = redis.call('ZPOPMAX', queue_key)
            if not batch_data or #batch_data == 0 then
                return nil
            end
            
            local batch_id = batch_data[1]
            local batch_key = job_prefix .. ':batch:' .. batch_id
            
            -- Check if batch still exists and is valid
            local batch_info = redis.call('HGETALL', batch_key)
            if #batch_info == 0 then
                return nil
            end
            
            -- Convert batch info to table
            local batch = {}
            for i = 1, #batch_info, 2 do
                batch[batch_info[i]] = batch_info[i + 1]
            end
            
            -- Check if batch is still pending
            if batch['status'] ~= 'pending' then
                return nil
            end
            
            -- Create job ID
            local job_id = 'job_' .. redis.call('INCR', job_prefix .. ':job_counter')
            local job_key = job_prefix .. ':job:' .. job_id
            
            -- Mark batch as processing
            redis.call('HSET', batch_key, 'status', 'processing', 'updated_at', current_time)
            
            -- Create job record
            redis.call('HMSET', job_key,
                'job_id', job_id,
                'batch_id', batch_id,
                'status', 'assigned',
                'assigned_worker', worker_id,
                'created_at', current_time,
                'updated_at', current_time,
                'priority', batch['priority'] or '0'
            )
            
            -- Copy batch data to job
            for key, value in pairs(batch) do
                if key ~= 'status' and key ~= 'updated_at' then
                    redis.call('HSET', job_key, 'batch_' .. key, value)
                end
            end
            
            -- Add to worker's active jobs
            redis.call('SADD', job_prefix .. ':worker:' .. worker_id .. ':jobs', job_id)
            
            -- Set job expiration (24 hours)
            redis.call('EXPIRE', job_key, 86400)
            
            return {job_id, batch_id}
        """)
        
        # Atomic job completion script
        self.complete_job_script = self.redis.register_script("""
            local job_prefix = KEYS[1]
            local job_id = ARGV[1]
            local worker_id = ARGV[2]
            local status = ARGV[3]
            local result_data = ARGV[4]
            local current_time = ARGV[5]
            
            local job_key = job_prefix .. ':job:' .. job_id
            
            -- Get job info
            local job_info = redis.call('HGETALL', job_key)
            if #job_info == 0 then
                return {0, 'Job not found'}
            end
            
            -- Convert to table
            local job = {}
            for i = 1, #job_info, 2 do
                job[job_info[i]] = job_info[i + 1]
            end
            
            -- Verify worker assignment
            if job['assigned_worker'] ~= worker_id then
                return {0, 'Job not assigned to this worker'}
            end
            
            -- Update job status
            redis.call('HMSET', job_key,
                'status', status,
                'completed_at', current_time,
                'updated_at', current_time
            )
            
            -- Store result if provided
            if result_data and result_data ~= '' then
                redis.call('HSET', job_key, 'result', result_data)
            end
            
            -- Remove from worker's active jobs
            redis.call('SREM', job_prefix .. ':worker:' .. worker_id .. ':jobs', job_id)
            
            -- Update batch counters
            local batch_id = job['batch_id']
            local batch_key = job_prefix .. ':batch:' .. batch_id
            
            if status == 'completed' then
                redis.call('HINCRBY', batch_key, 'completed_jobs', 1)
            elseif status == 'failed' then
                redis.call('HINCRBY', batch_key, 'failed_jobs', 1)
            end
            
            redis.call('HSET', batch_key, 'updated_at', current_time)
            
            -- Check if batch is complete
            local batch_info = redis.call('HGETALL', batch_key)
            local batch_data = {}
            for i = 1, #batch_info, 2 do
                batch_data[batch_info[i]] = batch_info[i + 1]
            end
            
            local total_jobs = tonumber(batch_data['total_jobs']) or 1
            local completed = tonumber(batch_data['completed_jobs']) or 0
            local failed = tonumber(batch_data['failed_jobs']) or 0
            
            if (completed + failed) >= total_jobs then
                redis.call('HSET', batch_key, 'status', 'completed')
            end
            
            return {1, 'Success'}
        """)
        
        # Atomic batch creation script
        self.create_batch_script = self.redis.register_script("""
            local queue_key = KEYS[1]
            local batch_prefix = KEYS[2]
            local batch_data = ARGV[1]
            local priority = tonumber(ARGV[2])
            local current_time = ARGV[3]
            
            -- Generate batch ID
            local batch_id = 'batch_' .. redis.call('INCR', batch_prefix .. ':batch_counter')
            local batch_key = batch_prefix .. ':batch:' .. batch_id
            
            -- Parse batch data
            local data = cjson.decode(batch_data)
            
            -- Create batch record
            redis.call('HMSET', batch_key,
                'batch_id', batch_id,
                'status', 'pending',
                'total_jobs', '1',
                'completed_jobs', '0',
                'failed_jobs', '0',
                'created_at', current_time,
                'updated_at', current_time,
                'priority', priority
            )
            
            -- Store batch data
            for key, value in pairs(data) do
                if type(value) == 'table' then
                    redis.call('HSET', batch_key, key, cjson.encode(value))
                else
                    redis.call('HSET', batch_key, key, tostring(value))
                end
            end
            
            -- Add to priority queue
            redis.call('ZADD', queue_key, priority, batch_id)
            
            -- Set expiration (7 days)
            redis.call('EXPIRE', batch_key, 604800)
            
            return batch_id
        """)
    
    async def create_batch(
        self, 
        batch_data: Dict[str, Any], 
        priority: int = 0
    ) -> str:
        """Create a new job batch atomically."""
        try:
            queue_key = "hashmancer:job_queue"
            batch_prefix = "hashmancer:jobs"
            
            batch_id = await self.create_batch_script(
                keys=[queue_key, batch_prefix],
                args=[
                    json.dumps(batch_data),
                    priority,
                    int(time.time())
                ]
            )
            
            logger.info(f"Created batch {batch_id} with priority {priority}")
            return batch_id
            
        except Exception as e:
            logger.error(f"Failed to create batch: {e}")
            raise
    
    async def assign_job_to_worker(self, worker_id: str) -> Optional[Tuple[str, str]]:
        """Assign next job to worker atomically."""
        try:
            queue_key = "hashmancer:job_queue"
            job_prefix = "hashmancer:jobs"
            
            result = await self.assign_job_script(
                keys=[queue_key, job_prefix],
                args=[worker_id, int(time.time())]
            )
            
            if result:
                job_id, batch_id = result
                logger.info(f"Assigned job {job_id} (batch {batch_id}) to worker {worker_id}")
                return job_id, batch_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to assign job to worker {worker_id}: {e}")
            return None
    
    async def complete_job(
        self, 
        job_id: str, 
        worker_id: str, 
        status: str = "completed",
        result_data: Dict[str, Any] = None
    ) -> bool:
        """Mark job as completed atomically."""
        try:
            job_prefix = "hashmancer:jobs"
            
            result_json = json.dumps(result_data) if result_data else ""
            
            result = await self.complete_job_script(
                keys=[job_prefix],
                args=[
                    job_id,
                    worker_id,
                    status,
                    result_json,
                    int(time.time())
                ]
            )
            
            success, message = result
            if success:
                logger.info(f"Job {job_id} completed by worker {worker_id}: {status}")
                return True
            else:
                logger.error(f"Failed to complete job {job_id}: {message}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to complete job {job_id}: {e}")
            return False
    
    async def get_job_info(self, job_id: str) -> Optional[JobInfo]:
        """Get job information."""
        try:
            job_key = f"hashmancer:jobs:job:{job_id}"
            job_data = await self.redis.hgetall(job_key)
            
            if not job_data:
                return None
            
            # Parse job data
            return JobInfo(
                job_id=job_data.get('job_id', job_id),
                batch_id=job_data.get('batch_id', ''),
                status=job_data.get('status', 'unknown'),
                assigned_worker=job_data.get('assigned_worker'),
                created_at=float(job_data.get('created_at', 0)),
                updated_at=float(job_data.get('updated_at', 0)),
                priority=int(job_data.get('priority', 0)),
                data={k[6:]: v for k, v in job_data.items() if k.startswith('batch_')}
            )
            
        except Exception as e:
            logger.error(f"Failed to get job info for {job_id}: {e}")
            return None
    
    async def get_batch_info(self, batch_id: str) -> Optional[BatchInfo]:
        """Get batch information."""
        try:
            batch_key = f"hashmancer:jobs:batch:{batch_id}"
            batch_data = await self.redis.hgetall(batch_key)
            
            if not batch_data:
                return None
            
            return BatchInfo(
                batch_id=batch_data.get('batch_id', batch_id),
                status=batch_data.get('status', 'unknown'),
                total_jobs=int(batch_data.get('total_jobs', 0)),
                completed_jobs=int(batch_data.get('completed_jobs', 0)),
                failed_jobs=int(batch_data.get('failed_jobs', 0)),
                created_at=float(batch_data.get('created_at', 0)),
                updated_at=float(batch_data.get('updated_at', 0)),
                priority=int(batch_data.get('priority', 0)),
                metadata={k: v for k, v in batch_data.items() 
                         if k not in ['batch_id', 'status', 'total_jobs', 'completed_jobs', 
                                     'failed_jobs', 'created_at', 'updated_at', 'priority']}
            )
            
        except Exception as e:
            logger.error(f"Failed to get batch info for {batch_id}: {e}")
            return None
    
    async def get_worker_jobs(self, worker_id: str) -> List[str]:
        """Get list of jobs assigned to worker."""
        try:
            jobs_key = f"hashmancer:jobs:worker:{worker_id}:jobs"
            job_ids = await self.redis.smembers(jobs_key)
            return list(job_ids)
        except Exception as e:
            logger.error(f"Failed to get worker jobs for {worker_id}: {e}")
            return []
    
    async def cancel_job(self, job_id: str, reason: str = "cancelled") -> bool:
        """Cancel a job atomically."""
        try:
            job_key = f"hashmancer:jobs:job:{job_id}"
            
            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()
            pipe.hget(job_key, 'assigned_worker')
            pipe.hget(job_key, 'status')
            results = await pipe.execute()
            
            assigned_worker = results[0]
            current_status = results[1]
            
            if current_status in ['completed', 'failed', 'cancelled']:
                return False  # Already finished
            
            # Update job status
            pipe = self.redis.pipeline()
            pipe.hmset(job_key, {
                'status': 'cancelled',
                'cancelled_reason': reason,
                'updated_at': int(time.time())
            })
            
            # Remove from worker's active jobs if assigned
            if assigned_worker:
                worker_jobs_key = f"hashmancer:jobs:worker:{assigned_worker}:jobs"
                pipe.srem(worker_jobs_key, job_id)
            
            await pipe.execute()
            
            logger.info(f"Cancelled job {job_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def cleanup_expired_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up expired jobs and batches."""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            cleaned_count = 0
            
            # Find expired jobs
            pattern = "hashmancer:jobs:job:*"
            async for key in self.redis.scan_iter(match=pattern):
                created_at = await self.redis.hget(key, 'created_at')
                if created_at and float(created_at) < cutoff_time:
                    status = await self.redis.hget(key, 'status')
                    if status in ['completed', 'failed', 'cancelled']:
                        await self.redis.delete(key)
                        cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired jobs")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired jobs: {e}")
            return 0
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get job queue statistics."""
        try:
            queue_key = "hashmancer:job_queue"
            
            # Get queue length
            queue_length = await self.redis.zcard(queue_key)
            
            # Count jobs by status
            job_pattern = "hashmancer:jobs:job:*"
            status_counts = {"pending": 0, "assigned": 0, "processing": 0, "completed": 0, "failed": 0, "cancelled": 0}
            
            async for key in self.redis.scan_iter(match=job_pattern):
                status = await self.redis.hget(key, 'status')
                if status in status_counts:
                    status_counts[status] += 1
            
            # Count batches by status
            batch_pattern = "hashmancer:jobs:batch:*"
            batch_counts = {"pending": 0, "processing": 0, "completed": 0}
            
            async for key in self.redis.scan_iter(match=batch_pattern):
                status = await self.redis.hget(key, 'status')
                if status in batch_counts:
                    batch_counts[status] += 1
            
            return {
                "queue_length": queue_length,
                "job_counts": status_counts,
                "batch_counts": batch_counts,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}

# Global instance
_atomic_job_manager = None

def get_atomic_job_manager() -> AtomicJobManager:
    """Get global atomic job manager."""
    global _atomic_job_manager
    if _atomic_job_manager is None:
        _atomic_job_manager = AtomicJobManager()
    return _atomic_job_manager