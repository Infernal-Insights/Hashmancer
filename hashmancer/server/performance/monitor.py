"""Performance monitoring and health checks."""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: tuple
    network_io_sent_mb: float = 0.0
    network_io_recv_mb: float = 0.0
    process_count: int = 0
    thread_count: int = 0


@dataclass
class RedisMetrics:
    """Redis performance metrics."""
    timestamp: float
    memory_used_mb: float
    connected_clients: int
    total_commands: int
    keyspace_hits: int
    keyspace_misses: int
    hit_rate: float
    ops_per_second: float = 0.0
    avg_ttl: float = 0.0
    key_count: int = 0


@dataclass
class HashmancerMetrics:
    """Hashmancer-specific performance metrics."""
    timestamp: float
    active_workers: int
    queue_length: int
    processing_jobs: int
    completed_jobs_per_minute: float
    average_job_duration: float
    cache_hit_rate: float
    wordlist_cache_size_mb: float


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, collection_interval: int = 30, history_limit: int = 1000):
        self.collection_interval = collection_interval
        self.history_limit = history_limit
        self._metrics_history: list = []
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._last_network_io = None
        self._last_redis_commands = 0
        self._job_completion_times = []
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started performance monitoring (interval: {self.collection_interval}s)")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self._executor.shutdown(wait=True)
        logger.info("Stopped performance monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Collect metrics asynchronously
                loop = asyncio.get_event_loop()
                
                system_task = loop.run_in_executor(self._executor, self._collect_system_metrics)
                redis_task = loop.run_in_executor(self._executor, self._collect_redis_metrics)
                hashmancer_task = loop.run_in_executor(self._executor, self._collect_hashmancer_metrics)
                
                system_metrics, redis_metrics, hashmancer_metrics = await asyncio.gather(
                    system_task, redis_task, hashmancer_task, return_exceptions=True
                )
                
                # Store metrics
                timestamp = time.time()
                metrics_snapshot = {
                    'timestamp': timestamp,
                    'system': system_metrics if not isinstance(system_metrics, Exception) else None,
                    'redis': redis_metrics if not isinstance(redis_metrics, Exception) else None,
                    'hashmancer': hashmancer_metrics if not isinstance(hashmancer_metrics, Exception) else None
                }
                
                self._metrics_history.append(metrics_snapshot)
                
                # Trim history
                if len(self._metrics_history) > self.history_limit:
                    self._metrics_history = self._metrics_history[-self.history_limit:]
                
                await asyncio.sleep(self.collection_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Load average
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_sent_mb = network_io.bytes_sent / (1024**2)
            network_recv_mb = network_io.bytes_recv / (1024**2)
            
            # Process information
            process_count = len(psutil.pids())
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                load_average=load_avg,
                network_io_sent_mb=network_sent_mb,
                network_io_recv_mb=network_recv_mb,
                process_count=process_count,
                thread_count=thread_count
            )
        
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            raise
    
    def _collect_redis_metrics(self) -> RedisMetrics:
        """Collect Redis performance metrics."""
        try:
            from ..performance.connection_pool import get_optimized_redis
            redis_client = get_optimized_redis('read')
            
            info = redis_client.info()
            memory_info = redis_client.info('memory')
            stats_info = redis_client.info('stats')
            
            # Calculate hit rate
            hits = stats_info.get('keyspace_hits', 0)
            misses = stats_info.get('keyspace_misses', 0)
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
            
            # Calculate ops per second
            total_commands = stats_info.get('total_commands_processed', 0)
            ops_per_second = 0.0
            if hasattr(self, '_last_redis_commands') and self._last_redis_commands > 0:
                command_diff = total_commands - self._last_redis_commands
                ops_per_second = command_diff / self.collection_interval
            self._last_redis_commands = total_commands
            
            # Get key count from all databases
            key_count = 0
            for i in range(16):  # Redis default 16 databases
                db_info = info.get(f'db{i}')
                if db_info:
                    keys = int(db_info.split(',')[0].split('=')[1])
                    key_count += keys
            
            return RedisMetrics(
                timestamp=time.time(),
                memory_used_mb=memory_info.get('used_memory', 0) / (1024**2),
                connected_clients=info.get('connected_clients', 0),
                total_commands=total_commands,
                keyspace_hits=hits,
                keyspace_misses=misses,
                hit_rate=hit_rate,
                ops_per_second=ops_per_second,
                key_count=key_count
            )
        
        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")
            raise
    
    def _collect_hashmancer_metrics(self) -> HashmancerMetrics:
        """Collect Hashmancer-specific performance metrics."""
        try:
            from ..performance.connection_pool import get_optimized_redis
            from ..performance.cache_manager import get_cache_manager
            from ..performance.memory_manager import get_wordlist_cache
            
            redis_client = get_optimized_redis('read')
            
            # Worker and queue metrics
            active_workers = redis_client.scard('workers:active') or 0
            queue_length = redis_client.llen('batch:queue') or 0
            processing_jobs = redis_client.llen('jobs:processing') or 0
            
            # Job completion rate
            completed_jobs_per_minute = 0.0
            if len(self._job_completion_times) > 0:
                recent_completions = [t for t in self._job_completion_times if time.time() - t < 60]
                completed_jobs_per_minute = len(recent_completions)
            
            # Average job duration (from recent completions)
            average_job_duration = 0.0
            if hasattr(self, '_recent_job_durations') and self._recent_job_durations:
                average_job_duration = sum(self._recent_job_durations) / len(self._recent_job_durations)
            
            # Cache metrics
            cache_stats = get_cache_manager().stats()
            cache_hit_rate = 0.0
            if 'memory_cache' in cache_stats:
                memory_stats = cache_stats['memory_cache']
                total_entries = memory_stats.get('total_entries', 0)
                cache_hit_rate = memory_stats.get('avg_access_count', 0) / max(total_entries, 1)
            
            # Wordlist cache size
            wordlist_stats = get_wordlist_cache().get_stats()
            wordlist_cache_size_mb = wordlist_stats.get('current_memory_mb', 0)
            
            return HashmancerMetrics(
                timestamp=time.time(),
                active_workers=active_workers,
                queue_length=queue_length,
                processing_jobs=processing_jobs,
                completed_jobs_per_minute=completed_jobs_per_minute,
                average_job_duration=average_job_duration,
                cache_hit_rate=cache_hit_rate,
                wordlist_cache_size_mb=wordlist_cache_size_mb
            )
        
        except Exception as e:
            logger.error(f"Failed to collect Hashmancer metrics: {e}")
            raise
    
    def record_job_completion(self, duration: float):
        """Record job completion for metrics."""
        current_time = time.time()
        self._job_completion_times.append(current_time)
        
        # Keep only recent completions (last hour)
        self._job_completion_times = [t for t in self._job_completion_times if current_time - t < 3600]
        
        # Track job durations
        if not hasattr(self, '_recent_job_durations'):
            self._recent_job_durations = []
        
        self._recent_job_durations.append(duration)
        
        # Keep only recent durations (last 100)
        if len(self._recent_job_durations) > 100:
            self._recent_job_durations = self._recent_job_durations[-100:]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics."""
        if not self._metrics_history:
            return {'error': 'No metrics available'}
        
        return self._metrics_history[-1]
    
    def get_metrics_history(self, limit: int = 100) -> list:
        """Get recent metrics history."""
        return self._metrics_history[-limit:] if self._metrics_history else []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with key indicators."""
        if not self._metrics_history:
            return {'error': 'No metrics available'}
        
        recent_metrics = self._metrics_history[-10:] if len(self._metrics_history) >= 10 else self._metrics_history
        
        # System averages
        system_metrics = [m['system'] for m in recent_metrics if m.get('system')]
        if system_metrics:
            avg_cpu = sum(m.cpu_percent for m in system_metrics) / len(system_metrics)
            avg_memory = sum(m.memory_percent for m in system_metrics) / len(system_metrics)
            avg_disk = sum(m.disk_usage_percent for m in system_metrics) / len(system_metrics)
        else:
            avg_cpu = avg_memory = avg_disk = 0
        
        # Redis averages
        redis_metrics = [m['redis'] for m in recent_metrics if m.get('redis')]
        if redis_metrics:
            avg_hit_rate = sum(m.hit_rate for m in redis_metrics) / len(redis_metrics)
            avg_ops_per_sec = sum(m.ops_per_second for m in redis_metrics) / len(redis_metrics)
        else:
            avg_hit_rate = avg_ops_per_sec = 0
        
        # Hashmancer averages
        hashmancer_metrics = [m['hashmancer'] for m in recent_metrics if m.get('hashmancer')]
        if hashmancer_metrics:
            avg_workers = sum(m.active_workers for m in hashmancer_metrics) / len(hashmancer_metrics)
            avg_queue_len = sum(m.queue_length for m in hashmancer_metrics) / len(hashmancer_metrics)
        else:
            avg_workers = avg_queue_len = 0
        
        # Health status
        health_status = 'healthy'
        if avg_cpu > 90 or avg_memory > 90 or avg_disk > 90:
            health_status = 'critical'
        elif avg_cpu > 70 or avg_memory > 70 or avg_disk > 80:
            health_status = 'warning'
        
        return {
            'health_status': health_status,
            'system': {
                'avg_cpu_percent': round(avg_cpu, 2),
                'avg_memory_percent': round(avg_memory, 2),
                'avg_disk_usage_percent': round(avg_disk, 2)
            },
            'redis': {
                'avg_hit_rate': round(avg_hit_rate, 3),
                'avg_ops_per_second': round(avg_ops_per_sec, 2)
            },
            'hashmancer': {
                'avg_active_workers': round(avg_workers, 1),
                'avg_queue_length': round(avg_queue_len, 1)
            },
            'collection_interval': self.collection_interval,
            'history_count': len(self._metrics_history)
        }
    
    def get_alerts(self) -> list:
        """Get performance alerts based on thresholds."""
        alerts = []
        
        if not self._metrics_history:
            return alerts
        
        latest = self._metrics_history[-1]
        
        # System alerts
        if latest.get('system'):
            system = latest['system']
            if system.cpu_percent > 90:
                alerts.append({'type': 'critical', 'message': f'High CPU usage: {system.cpu_percent:.1f}%'})
            if system.memory_percent > 90:
                alerts.append({'type': 'critical', 'message': f'High memory usage: {system.memory_percent:.1f}%'})
            if system.disk_usage_percent > 90:
                alerts.append({'type': 'critical', 'message': f'High disk usage: {system.disk_usage_percent:.1f}%'})
        
        # Redis alerts
        if latest.get('redis'):
            redis = latest['redis']
            if redis.hit_rate < 0.5:
                alerts.append({'type': 'warning', 'message': f'Low Redis hit rate: {redis.hit_rate:.1%}'})
            if redis.connected_clients > 1000:
                alerts.append({'type': 'warning', 'message': f'High Redis client count: {redis.connected_clients}'})
        
        # Hashmancer alerts
        if latest.get('hashmancer'):
            hashmancer = latest['hashmancer']
            if hashmancer.queue_length > 1000:
                alerts.append({'type': 'warning', 'message': f'Large queue backlog: {hashmancer.queue_length} batches'})
            if hashmancer.active_workers == 0:
                alerts.append({'type': 'critical', 'message': 'No active workers detected'})
        
        return alerts


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


async def start_performance_monitoring():
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    await monitor.start_monitoring()


async def stop_performance_monitoring():
    """Stop global performance monitoring."""
    monitor = get_performance_monitor()
    await monitor.stop_monitoring()