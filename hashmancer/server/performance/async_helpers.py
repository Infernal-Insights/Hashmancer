"""Async performance optimizations for job processing."""

import asyncio
import logging
import time
from typing import List, Callable, Any, Optional, Dict, Awaitable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from queue import Queue
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class JobResult:
    """Result of async job execution."""
    job_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0


class AsyncJobProcessor:
    """High-performance async job processor with batching and rate limiting."""
    
    def __init__(
        self,
        max_workers: int = 10,
        max_concurrent_jobs: int = 100,
        batch_size: int = 50,
        batch_timeout: float = 1.0,
        use_process_pool: bool = False
    ):
        self.max_workers = max_workers
        self.max_concurrent_jobs = max_concurrent_jobs
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.use_process_pool = use_process_pool
        
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._process_pool = ProcessPoolExecutor(max_workers=max_workers) if use_process_pool else None
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._job_queue = Queue()
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._running = False
        self._stats = {
            'jobs_submitted': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'batches_processed': 0,
            'total_duration': 0.0
        }
    
    async def start(self):
        """Start the async job processor."""
        if self._running:
            return
        
        self._running = True
        self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        logger.info(f"Started AsyncJobProcessor with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the async job processor."""
        if not self._running:
            return
        
        self._running = False
        
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass
        
        self._thread_pool.shutdown(wait=True)
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        
        logger.info("Stopped AsyncJobProcessor")
    
    async def submit_job(
        self,
        func: Callable,
        *args,
        job_id: Optional[str] = None,
        use_process_pool: bool = False,
        **kwargs
    ) -> JobResult:
        """Submit a job for async execution."""
        if job_id is None:
            job_id = f"job_{int(time.time() * 1000000)}"
        
        async with self._semaphore:
            self._stats['jobs_submitted'] += 1
            started_at = time.time()
            
            try:
                if use_process_pool and self._process_pool:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self._process_pool, func, *args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self._thread_pool, func, *args, **kwargs)
                
                completed_at = time.time()
                duration = completed_at - started_at
                
                self._stats['jobs_completed'] += 1
                self._stats['total_duration'] += duration
                
                return JobResult(
                    job_id=job_id,
                    success=True,
                    result=result,
                    duration=duration,
                    started_at=started_at,
                    completed_at=completed_at
                )
            
            except Exception as e:
                completed_at = time.time()
                duration = completed_at - started_at
                
                self._stats['jobs_failed'] += 1
                logger.error(f"Job {job_id} failed: {e}")
                
                return JobResult(
                    job_id=job_id,
                    success=False,
                    error=str(e),
                    duration=duration,
                    started_at=started_at,
                    completed_at=completed_at
                )
    
    async def submit_batch(
        self,
        jobs: List[tuple],  # [(func, args, kwargs), ...]
        use_process_pool: bool = False
    ) -> List[JobResult]:
        """Submit multiple jobs as a batch."""
        tasks = []
        
        for i, job_data in enumerate(jobs):
            if len(job_data) == 2:
                func, args = job_data
                kwargs = {}
            elif len(job_data) == 3:
                func, args, kwargs = job_data
            else:
                func = job_data[0]
                args = job_data[1] if len(job_data) > 1 else ()
                kwargs = job_data[2] if len(job_data) > 2 else {}
            
            job_id = f"batch_job_{int(time.time() * 1000000)}_{i}"
            task = asyncio.create_task(
                self.submit_job(func, *args, job_id=job_id, use_process_pool=use_process_pool, **kwargs)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        self._stats['batches_processed'] += 1
        return results
    
    async def _batch_processor_loop(self):
        """Internal batch processor loop."""
        while self._running:
            try:
                await asyncio.sleep(self.batch_timeout)
                # This could be extended to process jobs from a queue in batches
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor loop: {e}")
                await asyncio.sleep(1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        stats = self._stats.copy()
        
        if stats['jobs_completed'] > 0:
            stats['avg_job_duration'] = stats['total_duration'] / stats['jobs_completed']
        else:
            stats['avg_job_duration'] = 0.0
        
        stats['success_rate'] = (
            stats['jobs_completed'] / (stats['jobs_completed'] + stats['jobs_failed'])
            if (stats['jobs_completed'] + stats['jobs_failed']) > 0 else 0.0
        )
        
        return stats


# Global job processor instance
_job_processor: Optional[AsyncJobProcessor] = None


async def get_job_processor() -> AsyncJobProcessor:
    """Get global job processor instance."""
    global _job_processor
    if _job_processor is None:
        _job_processor = AsyncJobProcessor()
        await _job_processor.start()
    return _job_processor


async def async_batch_processor(
    items: List[Any],
    processor_func: Callable[[Any], Awaitable[Any]],
    batch_size: int = 50,
    max_concurrent: int = 10
) -> List[Any]:
    """Process items in batches with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    
    async def process_item(item):
        async with semaphore:
            return await processor_func(item)
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_tasks = [asyncio.create_task(process_item(item)) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        results.extend(batch_results)
    
    return results


class RateLimiter:
    """Token bucket rate limiter for async operations."""
    
    def __init__(self, rate: float, capacity: int = None):
        self.rate = rate  # tokens per second
        self.capacity = capacity or int(rate * 2)  # bucket capacity
        self.tokens = self.capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket."""
        async with self._lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1):
        """Wait until tokens are available."""
        while not await self.acquire(tokens):
            await asyncio.sleep(0.1)


@asynccontextmanager
async def rate_limited(rate_limiter: RateLimiter, tokens: int = 1):
    """Async context manager for rate limiting."""
    await rate_limiter.wait_for_tokens(tokens)
    try:
        yield
    finally:
        pass


class AsyncWorkerPool:
    """Pool of async workers for processing tasks."""
    
    def __init__(self, worker_count: int = 5, max_queue_size: int = 1000):
        self.worker_count = worker_count
        self.max_queue_size = max_queue_size
        self._queue = asyncio.Queue(max_queue_size)
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'workers_active': 0
        }
    
    async def start(self):
        """Start the worker pool."""
        if self._running:
            return
        
        self._running = True
        
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self._workers.append(worker)
        
        logger.info(f"Started AsyncWorkerPool with {self.worker_count} workers")
    
    async def stop(self):
        """Stop the worker pool."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Stopped AsyncWorkerPool")
    
    async def submit_task(self, coro: Awaitable[Any]):
        """Submit a coroutine task to the pool."""
        if not self._running:
            raise RuntimeError("Worker pool is not running")
        
        await self._queue.put(coro)
    
    async def _worker_loop(self, worker_name: str):
        """Worker loop for processing tasks."""
        logger.debug(f"Started worker: {worker_name}")
        self._stats['workers_active'] += 1
        
        try:
            while self._running:
                try:
                    # Get task with timeout
                    coro = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    
                    # Execute the coroutine
                    try:
                        await coro
                        self._stats['tasks_processed'] += 1
                    except Exception as e:
                        self._stats['tasks_failed'] += 1
                        logger.error(f"Task failed in {worker_name}: {e}")
                    finally:
                        self._queue.task_done()
                
                except asyncio.TimeoutError:
                    # No task available, continue loop
                    continue
        
        except asyncio.CancelledError:
            pass
        
        finally:
            self._stats['workers_active'] -= 1
            logger.debug(f"Stopped worker: {worker_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        return {
            **self._stats,
            'queue_size': self._queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'worker_count': self.worker_count
        }