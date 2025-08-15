"""
Enhanced Worker Management System
Advanced worker orchestration, monitoring, and optimization for Hashmancer
"""
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import redis
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class WorkerStatus(Enum):
    """Worker status enumeration."""
    IDLE = "idle"
    WORKING = "working"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    ERROR = "error"
    BENCHMARKING = "benchmarking"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class WorkerInfo:
    """Comprehensive worker information."""
    worker_id: str
    name: str
    status: WorkerStatus
    gpu_info: Dict[str, Any]
    system_info: Dict[str, Any]
    capabilities: List[str]
    current_job: Optional[str] = None
    last_seen: float = 0
    total_jobs_completed: int = 0
    total_hashes_cracked: int = 0
    average_hashrate: float = 0
    error_count: int = 0
    uptime: float = 0
    temperature: float = 0
    power_usage: float = 0
    memory_usage: float = 0
    performance_rating: float = 0

@dataclass
class JobTask:
    """Job task definition."""
    task_id: str
    job_id: str
    worker_id: Optional[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    hash_type: str = ""
    attack_mode: str = ""
    target_hashes: List[str] = None
    wordlist: str = ""
    rules: str = ""
    mask: str = ""
    progress: float = 0
    estimated_time: float = 0
    created_at: float = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"
    results: Dict[str, Any] = None

@dataclass
class WorkerMetrics:
    """Worker performance metrics."""
    worker_id: str
    timestamp: float
    hashrate: float
    gpu_utilization: float
    gpu_temperature: float
    gpu_memory_usage: float
    power_consumption: float
    cpu_usage: float
    ram_usage: float
    efficiency_score: float

class EnhancedWorkerManager:
    """Advanced worker management with intelligent task scheduling."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        self.workers: Dict[str, WorkerInfo] = {}
        self.task_queue: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self.active_tasks: Dict[str, JobTask] = {}
        self.worker_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Configuration
        self.max_concurrent_tasks = 10
        self.worker_timeout = 300  # 5 minutes
        self.metrics_interval = 30  # seconds
        self.auto_balance_enabled = True
        self.smart_scheduling_enabled = True
        
        # Statistics
        self.stats = {
            "total_workers_registered": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            "total_hashes_cracked": 0,
            "average_task_completion_time": 0,
            "system_efficiency": 0
        }
        
        # Background tasks
        self.monitoring_task = None
        self.scheduling_task = None
        self.metrics_collection_task = None
        self.is_running = False
    
    async def initialize(self):
        """Initialize the worker manager."""
        logger.info("Initializing Enhanced Worker Manager...")
        
        # Load existing workers from Redis
        await self._load_workers_from_redis()
        
        # Start background tasks
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._worker_monitoring_loop())
        self.scheduling_task = asyncio.create_task(self._task_scheduling_loop())
        self.metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
        
        logger.info(f"Worker Manager initialized with {len(self.workers)} workers")
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Shutting down Enhanced Worker Manager...")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in [self.monitoring_task, self.scheduling_task, self.metrics_collection_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save worker states to Redis
        await self._save_workers_to_redis()
        
        logger.info("Worker Manager shutdown complete")
    
    async def register_worker(self, worker_data: Dict[str, Any]) -> str:
        """Register a new worker or update existing one."""
        worker_id = worker_data.get("worker_id") or str(uuid.uuid4())
        
        # Parse GPU information
        gpu_info = {
            "name": worker_data.get("gpu_name", "Unknown"),
            "memory": worker_data.get("gpu_memory", 0),
            "compute_capability": worker_data.get("gpu_compute", "Unknown"),
            "driver_version": worker_data.get("gpu_driver", "Unknown")
        }
        
        # Parse system information
        system_info = {
            "os": worker_data.get("os", "Unknown"),
            "cpu": worker_data.get("cpu", "Unknown"),
            "ram": worker_data.get("ram", 0),
            "hostname": worker_data.get("hostname", "Unknown")
        }
        
        # Determine capabilities based on hardware
        capabilities = await self._determine_worker_capabilities(gpu_info, system_info)
        
        # Create or update worker
        worker = WorkerInfo(
            worker_id=worker_id,
            name=worker_data.get("name", f"Worker-{worker_id[:8]}"),
            status=WorkerStatus.IDLE,
            gpu_info=gpu_info,
            system_info=system_info,
            capabilities=capabilities,
            last_seen=time.time()
        )
        
        self.workers[worker_id] = worker
        self.stats["total_workers_registered"] += 1
        
        # Save to Redis
        await self._save_worker_to_redis(worker)
        
        logger.info(f"Worker registered: {worker_id} ({worker.name})")
        
        return worker_id
    
    async def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker."""
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        
        # Reassign any active tasks
        await self._reassign_worker_tasks(worker_id)
        
        # Remove worker
        del self.workers[worker_id]
        
        # Remove from Redis
        await self.redis_client.hdel("workers", worker_id)
        
        logger.info(f"Worker unregistered: {worker_id} ({worker.name})")
        
        return True
    
    async def update_worker_status(self, worker_id: str, status: WorkerStatus, 
                                 additional_data: Optional[Dict] = None) -> bool:
        """Update worker status and additional data."""
        if worker_id not in self.workers:
            logger.warning(f"Attempted to update unknown worker: {worker_id}")
            return False
        
        worker = self.workers[worker_id]
        old_status = worker.status
        worker.status = status
        worker.last_seen = time.time()
        
        # Update additional data if provided
        if additional_data:
            if "temperature" in additional_data:
                worker.temperature = additional_data["temperature"]
            if "power_usage" in additional_data:
                worker.power_usage = additional_data["power_usage"]
            if "memory_usage" in additional_data:
                worker.memory_usage = additional_data["memory_usage"]
        
        # Handle status-specific logic
        if status == WorkerStatus.OFFLINE:
            await self._reassign_worker_tasks(worker_id)
        elif status == WorkerStatus.ERROR:
            worker.error_count += 1
            await self._handle_worker_error(worker_id, additional_data)
        
        await self._save_worker_to_redis(worker)
        
        logger.info(f"Worker {worker_id} status changed: {old_status.value} → {status.value}")
        
        return True
    
    async def submit_task(self, task_data: Dict[str, Any]) -> str:
        """Submit a new task to the queue."""
        task_id = str(uuid.uuid4())
        
        task = JobTask(
            task_id=task_id,
            job_id=task_data.get("job_id", ""),
            priority=TaskPriority(task_data.get("priority", 2)),
            hash_type=task_data.get("hash_type", ""),
            attack_mode=task_data.get("attack_mode", ""),
            target_hashes=task_data.get("target_hashes", []),
            wordlist=task_data.get("wordlist", ""),
            rules=task_data.get("rules", ""),
            mask=task_data.get("mask", ""),
            estimated_time=task_data.get("estimated_time", 0),
            created_at=time.time(),
            status="queued"
        )
        
        # Add to appropriate priority queue
        self.task_queue[task.priority].append(task)
        
        logger.info(f"Task submitted: {task_id} (Priority: {task.priority.name})")
        
        return task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        # Check if task is active
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if task.worker_id:
                # Send cancellation to worker
                await self._send_worker_command(task.worker_id, "cancel_task", {"task_id": task_id})
            
            del self.active_tasks[task_id]
            logger.info(f"Active task cancelled: {task_id}")
            return True
        
        # Check if task is in queue
        for priority_queue in self.task_queue.values():
            for i, task in enumerate(priority_queue):
                if task.task_id == task_id:
                    del priority_queue[i]
                    logger.info(f"Queued task cancelled: {task_id}")
                    return True
        
        return False
    
    async def get_worker_info(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive worker information."""
        if worker_id not in self.workers:
            return None
        
        worker = self.workers[worker_id]
        
        # Get recent metrics
        recent_metrics = list(self.worker_metrics[worker_id])[-10:] if worker_id in self.worker_metrics else []
        
        # Calculate performance statistics
        performance_stats = await self._calculate_worker_performance(worker_id)
        
        return {
            **asdict(worker),
            "recent_metrics": [asdict(m) for m in recent_metrics],
            "performance_stats": performance_stats,
            "current_tasks": [
                task_id for task_id, task in self.active_tasks.items()
                if task.worker_id == worker_id
            ]
        }
    
    async def get_all_workers(self) -> List[Dict[str, Any]]:
        """Get information for all workers."""
        workers_info = []
        
        for worker_id in self.workers:
            worker_info = await self.get_worker_info(worker_id)
            if worker_info:
                workers_info.append(worker_info)
        
        return workers_info
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        current_time = time.time()
        
        # Worker statistics
        worker_stats = {
            "total": len(self.workers),
            "by_status": defaultdict(int),
            "total_capacity": 0,
            "utilized_capacity": 0
        }
        
        for worker in self.workers.values():
            worker_stats["by_status"][worker.status.value] += 1
            worker_stats["total_capacity"] += 1
            if worker.status in [WorkerStatus.WORKING, WorkerStatus.BENCHMARKING]:
                worker_stats["utilized_capacity"] += 1
        
        # Task statistics
        task_stats = {
            "active": len(self.active_tasks),
            "queued": sum(len(queue) for queue in self.task_queue.values()),
            "by_priority": {
                priority.name: len(queue) for priority, queue in self.task_queue.items()
            }
        }
        
        # Performance metrics
        total_hashrate = sum(
            worker.average_hashrate for worker in self.workers.values()
            if worker.status == WorkerStatus.WORKING
        )
        
        utilization_rate = (
            worker_stats["utilized_capacity"] / max(worker_stats["total"], 1)
        ) * 100
        
        return {
            "timestamp": current_time,
            "workers": dict(worker_stats),
            "tasks": task_stats,
            "performance": {
                "total_hashrate": total_hashrate,
                "utilization_rate": utilization_rate,
                "system_efficiency": self.stats["system_efficiency"]
            },
            "statistics": self.stats
        }
    
    async def run_benchmark(self, worker_id: str, benchmark_config: Dict[str, Any]) -> str:
        """Run benchmark on specific worker."""
        if worker_id not in self.workers:
            raise ValueError(f"Worker {worker_id} not found")
        
        worker = self.workers[worker_id]
        if worker.status != WorkerStatus.IDLE:
            raise ValueError(f"Worker {worker_id} is not available for benchmarking")
        
        # Create benchmark task
        benchmark_id = str(uuid.uuid4())
        benchmark_task = {
            "benchmark_id": benchmark_id,
            "applications": benchmark_config.get("applications", ["hashcat"]),
            "hash_types": benchmark_config.get("hash_types", ["0"]),
            "duration": benchmark_config.get("duration", 60),
            "attack_mode": benchmark_config.get("attack_mode", "3")
        }
        
        # Update worker status
        await self.update_worker_status(worker_id, WorkerStatus.BENCHMARKING)
        
        # Send benchmark command to worker
        await self._send_worker_command(worker_id, "run_benchmark", benchmark_task)
        
        logger.info(f"Benchmark started on worker {worker_id}: {benchmark_id}")
        
        return benchmark_id
    
    async def optimize_task_distribution(self) -> Dict[str, Any]:
        """Optimize task distribution across workers."""
        if not self.smart_scheduling_enabled:
            return {"message": "Smart scheduling disabled"}
        
        # Analyze worker performance
        worker_performance = {}
        for worker_id, worker in self.workers.items():
            if worker.status in [WorkerStatus.IDLE, WorkerStatus.WORKING]:
                performance_score = await self._calculate_worker_performance_score(worker_id)
                worker_performance[worker_id] = performance_score
        
        # Reassign tasks based on performance
        reassignments = 0
        for task in list(self.active_tasks.values()):
            if task.worker_id and task.worker_id in worker_performance:
                current_score = worker_performance[task.worker_id]
                
                # Find better worker
                best_worker_id = max(
                    worker_performance.keys(),
                    key=lambda w_id: worker_performance[w_id] if self.workers[w_id].status == WorkerStatus.IDLE else 0
                )
                
                if (self.workers[best_worker_id].status == WorkerStatus.IDLE and 
                    worker_performance[best_worker_id] > current_score * 1.2):
                    
                    # Reassign task
                    await self._reassign_task(task.task_id, best_worker_id)
                    reassignments += 1
        
        return {
            "reassignments": reassignments,
            "worker_performance": worker_performance
        }
    
    # Private methods
    
    async def _load_workers_from_redis(self):
        """Load workers from Redis."""
        try:
            worker_data = await self.redis_client.hgetall("workers")
            for worker_id, data in worker_data.items():
                worker_info = json.loads(data)
                worker = WorkerInfo(**worker_info)
                # Reset status for offline workers
                if time.time() - worker.last_seen > self.worker_timeout:
                    worker.status = WorkerStatus.OFFLINE
                self.workers[worker_id] = worker
        except Exception as e:
            logger.error(f"Error loading workers from Redis: {e}")
    
    async def _save_workers_to_redis(self):
        """Save all workers to Redis."""
        try:
            worker_data = {
                worker_id: json.dumps(asdict(worker))
                for worker_id, worker in self.workers.items()
            }
            if worker_data:
                await self.redis_client.hmset("workers", worker_data)
        except Exception as e:
            logger.error(f"Error saving workers to Redis: {e}")
    
    async def _save_worker_to_redis(self, worker: WorkerInfo):
        """Save single worker to Redis."""
        try:
            await self.redis_client.hset("workers", worker.worker_id, json.dumps(asdict(worker)))
        except Exception as e:
            logger.error(f"Error saving worker {worker.worker_id} to Redis: {e}")
    
    async def _determine_worker_capabilities(self, gpu_info: Dict, system_info: Dict) -> List[str]:
        """Determine worker capabilities based on hardware."""
        capabilities = []
        
        # GPU-based capabilities
        gpu_name = gpu_info.get("name", "").lower()
        if "rtx" in gpu_name or "gtx" in gpu_name:
            capabilities.extend(["cuda", "nvidia"])
        elif "radeon" in gpu_name or "rx" in gpu_name:
            capabilities.extend(["opencl", "amd"])
        
        # Memory-based capabilities
        gpu_memory = gpu_info.get("memory", 0)
        if gpu_memory >= 24000:  # 24GB+
            capabilities.append("large_datasets")
        elif gpu_memory >= 12000:  # 12GB+
            capabilities.append("medium_datasets")
        
        # System capabilities
        ram = system_info.get("ram", 0)
        if ram >= 32:  # 32GB+
            capabilities.append("high_memory")
        
        return capabilities
    
    async def _worker_monitoring_loop(self):
        """Monitor worker health and status."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check for offline workers
                for worker_id, worker in list(self.workers.items()):
                    if current_time - worker.last_seen > self.worker_timeout:
                        if worker.status != WorkerStatus.OFFLINE:
                            await self.update_worker_status(worker_id, WorkerStatus.OFFLINE)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in worker monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _task_scheduling_loop(self):
        """Main task scheduling loop."""
        while self.is_running:
            try:
                # Schedule tasks from queue
                await self._schedule_pending_tasks()
                
                # Balance workload if enabled
                if self.auto_balance_enabled:
                    await self._balance_workload()
                
                await asyncio.sleep(5)  # Schedule every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in task scheduling loop: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_collection_loop(self):
        """Collect worker metrics periodically."""
        while self.is_running:
            try:
                # Request metrics from active workers
                for worker_id, worker in self.workers.items():
                    if worker.status in [WorkerStatus.WORKING, WorkerStatus.IDLE]:
                        await self._request_worker_metrics(worker_id)
                
                await asyncio.sleep(self.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.metrics_interval)
    
    async def _schedule_pending_tasks(self):
        """Schedule pending tasks to available workers."""
        available_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if worker.status == WorkerStatus.IDLE
        ]
        
        if not available_workers or len(self.active_tasks) >= self.max_concurrent_tasks:
            return
        
        # Process tasks by priority
        for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
            queue = self.task_queue[priority]
            
            while queue and available_workers and len(self.active_tasks) < self.max_concurrent_tasks:
                task = queue.popleft()
                
                # Find best worker for this task
                best_worker_id = await self._find_best_worker_for_task(task, available_workers)
                
                if best_worker_id:
                    await self._assign_task_to_worker(task, best_worker_id)
                    available_workers.remove(best_worker_id)
    
    async def _find_best_worker_for_task(self, task: JobTask, available_workers: List[str]) -> Optional[str]:
        """Find the best worker for a specific task."""
        if not available_workers:
            return None
        
        if not self.smart_scheduling_enabled:
            return available_workers[0]
        
        # Score workers based on task requirements
        worker_scores = {}
        
        for worker_id in available_workers:
            worker = self.workers[worker_id]
            score = 0
            
            # Base score from performance rating
            score += worker.performance_rating * 10
            
            # Bonus for task-specific capabilities
            if task.hash_type in ["22000", "22001"] and "high_memory" in worker.capabilities:
                score += 20  # WPA tasks benefit from high memory
            
            if task.attack_mode == "0" and "large_datasets" in worker.capabilities:
                score += 15  # Dictionary attacks benefit from large memory
            
            # Penalty for recent errors
            score -= worker.error_count * 5
            
            # Temperature penalty
            if worker.temperature > 80:
                score -= 10
            
            worker_scores[worker_id] = score
        
        # Return worker with highest score
        return max(worker_scores, key=worker_scores.get)
    
    async def _assign_task_to_worker(self, task: JobTask, worker_id: str):
        """Assign task to specific worker."""
        task.worker_id = worker_id
        task.started_at = time.time()
        task.status = "running"
        
        self.active_tasks[task.task_id] = task
        
        # Update worker status
        await self.update_worker_status(worker_id, WorkerStatus.WORKING)
        self.workers[worker_id].current_job = task.job_id
        
        # Send task to worker
        await self._send_worker_command(worker_id, "start_task", asdict(task))
        
        logger.info(f"Task {task.task_id} assigned to worker {worker_id}")
    
    async def _send_worker_command(self, worker_id: str, command: str, data: Dict[str, Any]):
        """Send command to worker via Redis."""
        try:
            command_data = {
                "command": command,
                "data": data,
                "timestamp": time.time()
            }
            
            await self.redis_client.lpush(f"worker_commands:{worker_id}", json.dumps(command_data))
            logger.debug(f"Command sent to worker {worker_id}: {command}")
            
        except Exception as e:
            logger.error(f"Error sending command to worker {worker_id}: {e}")
    
    async def _reassign_worker_tasks(self, worker_id: str):
        """Reassign all tasks from a worker."""
        tasks_to_reassign = [
            task for task in self.active_tasks.values()
            if task.worker_id == worker_id
        ]
        
        for task in tasks_to_reassign:
            # Remove from active tasks
            del self.active_tasks[task.task_id]
            
            # Reset task state
            task.worker_id = None
            task.started_at = None
            task.status = "queued"
            
            # Add back to queue
            self.task_queue[task.priority].appendleft(task)
            
            logger.info(f"Task {task.task_id} reassigned from worker {worker_id}")
    
    async def _calculate_worker_performance(self, worker_id: str) -> Dict[str, Any]:
        """Calculate detailed worker performance statistics."""
        if worker_id not in self.worker_metrics:
            return {"error": "No metrics available"}
        
        metrics = list(self.worker_metrics[worker_id])
        if not metrics:
            return {"error": "No metrics data"}
        
        # Calculate averages
        avg_hashrate = sum(m.hashrate for m in metrics) / len(metrics)
        avg_gpu_util = sum(m.gpu_utilization for m in metrics) / len(metrics)
        avg_temperature = sum(m.gpu_temperature for m in metrics) / len(metrics)
        avg_efficiency = sum(m.efficiency_score for m in metrics) / len(metrics)
        
        return {
            "average_hashrate": avg_hashrate,
            "average_gpu_utilization": avg_gpu_util,
            "average_temperature": avg_temperature,
            "average_efficiency": avg_efficiency,
            "metric_count": len(metrics)
        }
    
    async def _calculate_worker_performance_score(self, worker_id: str) -> float:
        """Calculate a single performance score for worker ranking."""
        worker = self.workers[worker_id]
        
        # Base score from performance rating
        score = worker.performance_rating
        
        # Adjust based on current status
        if worker.status == WorkerStatus.IDLE:
            score += 10  # Bonus for availability
        elif worker.status == WorkerStatus.ERROR:
            score -= 20  # Penalty for errors
        
        # Temperature penalty
        if worker.temperature > 85:
            score -= 15
        elif worker.temperature > 75:
            score -= 5
        
        # Reliability bonus
        if worker.error_count == 0:
            score += 5
        else:
            score -= worker.error_count * 2
        
        return max(0, score)  # Ensure non-negative score
    
    async def _balance_workload(self):
        """Balance workload across workers."""
        # Simple load balancing - move tasks from overloaded workers
        worker_loads = {
            worker_id: sum(1 for task in self.active_tasks.values() if task.worker_id == worker_id)
            for worker_id in self.workers
        }
        
        # Find overloaded and underutilized workers
        avg_load = sum(worker_loads.values()) / max(len(worker_loads), 1)
        
        overloaded = [w_id for w_id, load in worker_loads.items() if load > avg_load + 1]
        underutilized = [w_id for w_id, load in worker_loads.items() 
                        if load < avg_load and self.workers[w_id].status == WorkerStatus.IDLE]
        
        # Move tasks if beneficial
        for overloaded_worker in overloaded:
            if not underutilized:
                break
                
            # Find a task to move
            worker_tasks = [
                task for task in self.active_tasks.values()
                if task.worker_id == overloaded_worker and task.priority != TaskPriority.CRITICAL
            ]
            
            if worker_tasks:
                task_to_move = min(worker_tasks, key=lambda t: t.priority.value)
                target_worker = underutilized.pop(0)
                
                await self._reassign_task(task_to_move.task_id, target_worker)
    
    async def _reassign_task(self, task_id: str, new_worker_id: str):
        """Reassign a specific task to a new worker."""
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[task_id]
        old_worker_id = task.worker_id
        
        # Cancel task on old worker
        if old_worker_id:
            await self._send_worker_command(old_worker_id, "cancel_task", {"task_id": task_id})
        
        # Assign to new worker
        await self._assign_task_to_worker(task, new_worker_id)
        
        logger.info(f"Task {task_id} reassigned from {old_worker_id} to {new_worker_id}")
        return True
    
    async def _handle_worker_error(self, worker_id: str, error_data: Optional[Dict]):
        """Handle worker error condition."""
        worker = self.workers[worker_id]
        
        logger.error(f"Worker error: {worker_id} - {error_data}")
        
        # Reassign tasks if too many errors
        if worker.error_count >= 3:
            await self._reassign_worker_tasks(worker_id)
            await self.update_worker_status(worker_id, WorkerStatus.MAINTENANCE)
    
    async def _request_worker_metrics(self, worker_id: str):
        """Request metrics from specific worker."""
        await self._send_worker_command(worker_id, "get_metrics", {})
    
    async def health_check(self):
        """Health check for the worker manager."""
        return {
            "status": "healthy" if self.is_running else "stopped",
            "workers_count": len(self.workers),
            "active_tasks": len(self.active_tasks),
            "queued_tasks": sum(len(queue) for queue in self.task_queue.values())
        }