"""Redis query optimization and batch operations."""

import time
import json
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for query performance monitoring."""
    query_type: str
    execution_time: float
    keys_accessed: int = 0
    keys_modified: int = 0
    pipeline_size: int = 0
    success: bool = True
    error: Optional[str] = None


class QueryOptimizer:
    """Redis query optimization with batching and performance monitoring."""
    
    def __init__(self, batch_size: int = 100, pipeline_buffer_size: int = 1000):
        self.batch_size = batch_size
        self.pipeline_buffer_size = pipeline_buffer_size
        self._metrics: List[QueryMetrics] = []
        self._redis_client = None
        
    def _get_redis(self):
        """Get optimized Redis client."""
        if self._redis_client is None:
            from .connection_pool import get_optimized_redis
            self._redis_client = get_optimized_redis()
        return self._redis_client
    
    def _record_metrics(self, metrics: QueryMetrics):
        """Record query metrics."""
        self._metrics.append(metrics)
        
        # Keep only recent metrics to prevent memory growth
        if len(self._metrics) > 1000:
            self._metrics = self._metrics[-500:]
    
    @contextmanager
    def _timed_operation(self, query_type: str):
        """Context manager for timing operations."""
        start_time = time.time()
        metrics = QueryMetrics(query_type=query_type, execution_time=0.0)
        
        try:
            yield metrics
            metrics.success = True
        except Exception as e:
            metrics.success = False
            metrics.error = str(e)
            raise
        finally:
            metrics.execution_time = time.time() - start_time
            self._record_metrics(metrics)
    
    def batch_get(self, keys: List[str], key_prefix: str = "") -> Dict[str, Any]:
        """Efficiently get multiple keys using pipeline."""
        if not keys:
            return {}
        
        with self._timed_operation("batch_get") as metrics:
            redis_keys = [f"{key_prefix}{key}" if key_prefix else key for key in keys]
            metrics.keys_accessed = len(redis_keys)
            
            redis_client = self._get_redis()
            
            # Use pipeline for batch operations
            with redis_client.pipeline() as pipeline:
                for key in redis_keys:
                    pipeline.get(key)
                
                values = pipeline.execute()
                metrics.pipeline_size = len(redis_keys)
            
            # Build result dictionary
            result = {}
            for i, (original_key, value) in enumerate(zip(keys, values)):
                if value is not None:
                    try:
                        # Try to deserialize JSON
                        result[original_key] = json.loads(value.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                        # Return raw value if not JSON
                        result[original_key] = value.decode('utf-8') if isinstance(value, bytes) else value
            
            return result
    
    def batch_set(self, data: Dict[str, Any], key_prefix: str = "", ttl: Optional[int] = None) -> int:
        """Efficiently set multiple keys using pipeline."""
        if not data:
            return 0
        
        with self._timed_operation("batch_set") as metrics:
            metrics.keys_modified = len(data)
            
            redis_client = self._get_redis()
            
            with redis_client.pipeline() as pipeline:
                for key, value in data.items():
                    redis_key = f"{key_prefix}{key}" if key_prefix else key
                    
                    # Serialize value if necessary
                    if isinstance(value, (dict, list)):
                        serialized_value = json.dumps(value)
                    else:
                        serialized_value = str(value)
                    
                    if ttl:
                        pipeline.setex(redis_key, ttl, serialized_value)
                    else:
                        pipeline.set(redis_key, serialized_value)
                
                results = pipeline.execute()
                metrics.pipeline_size = len(data)
            
            return sum(1 for result in results if result)
    
    def batch_delete(self, keys: List[str], key_prefix: str = "") -> int:
        """Efficiently delete multiple keys."""
        if not keys:
            return 0
        
        with self._timed_operation("batch_delete") as metrics:
            redis_keys = [f"{key_prefix}{key}" if key_prefix else key for key in keys]
            metrics.keys_modified = len(redis_keys)
            
            redis_client = self._get_redis()
            
            # Use single delete command for efficiency
            deleted_count = redis_client.delete(*redis_keys)
            return deleted_count
    
    def batch_hash_operations(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Execute batch hash operations efficiently."""
        if not operations:
            return []
        
        with self._timed_operation("batch_hash_ops") as metrics:
            redis_client = self._get_redis()
            metrics.keys_accessed = len(operations)
            
            with redis_client.pipeline() as pipeline:
                for op in operations:
                    op_type = op.get('type')
                    key = op.get('key')
                    
                    if op_type == 'hget':
                        pipeline.hget(key, op.get('field'))
                    elif op_type == 'hset':
                        pipeline.hset(key, op.get('field'), op.get('value'))
                    elif op_type == 'hgetall':
                        pipeline.hgetall(key)
                    elif op_type == 'hmset':
                        pipeline.hmset(key, op.get('mapping', {}))
                    elif op_type == 'hdel':
                        pipeline.hdel(key, *op.get('fields', []))
                    else:
                        logger.warning(f"Unknown hash operation: {op_type}")
                
                results = pipeline.execute()
                metrics.pipeline_size = len(operations)
            
            return results
    
    def batch_list_operations(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Execute batch list operations efficiently."""
        if not operations:
            return []
        
        with self._timed_operation("batch_list_ops") as metrics:
            redis_client = self._get_redis()
            metrics.keys_accessed = len(operations)
            
            with redis_client.pipeline() as pipeline:
                for op in operations:
                    op_type = op.get('type')
                    key = op.get('key')
                    
                    if op_type == 'lpush':
                        pipeline.lpush(key, *op.get('values', []))
                    elif op_type == 'rpush':
                        pipeline.rpush(key, *op.get('values', []))
                    elif op_type == 'lpop':
                        pipeline.lpop(key)
                    elif op_type == 'rpop':
                        pipeline.rpop(key)
                    elif op_type == 'llen':
                        pipeline.llen(key)
                    elif op_type == 'lrange':
                        start = op.get('start', 0)
                        end = op.get('end', -1)
                        pipeline.lrange(key, start, end)
                    elif op_type == 'ltrim':
                        start = op.get('start', 0)
                        end = op.get('end', -1)
                        pipeline.ltrim(key, start, end)
                    else:
                        logger.warning(f"Unknown list operation: {op_type}")
                
                results = pipeline.execute()
                metrics.pipeline_size = len(operations)
            
            return results
    
    def optimized_scan(self, pattern: str = "*", count: int = 1000, match_type: str = "keys") -> List[str]:
        """Optimized key scanning with pattern matching."""
        with self._timed_operation(f"scan_{match_type}") as metrics:
            redis_client = self._get_redis()
            results = []
            cursor = 0
            
            while True:
                if match_type == "keys":
                    cursor, keys = redis_client.scan(cursor, match=pattern, count=count)
                elif match_type == "hash_keys":
                    cursor, keys = redis_client.hscan(pattern, cursor, count=count)
                else:
                    raise ValueError(f"Unsupported scan type: {match_type}")
                
                if isinstance(keys, dict):
                    # Hash scan returns dict
                    results.extend(keys.keys())
                else:
                    # Key scan returns list
                    results.extend(keys)
                
                if cursor == 0:
                    break
            
            metrics.keys_accessed = len(results)
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in results]
    
    def batch_expire(self, key_ttl_pairs: List[tuple], key_prefix: str = "") -> int:
        """Set expiration on multiple keys efficiently."""
        if not key_ttl_pairs:
            return 0
        
        with self._timed_operation("batch_expire") as metrics:
            redis_client = self._get_redis()
            metrics.keys_modified = len(key_ttl_pairs)
            
            with redis_client.pipeline() as pipeline:
                for key, ttl in key_ttl_pairs:
                    redis_key = f"{key_prefix}{key}" if key_prefix else key
                    pipeline.expire(redis_key, ttl)
                
                results = pipeline.execute()
                metrics.pipeline_size = len(key_ttl_pairs)
            
            return sum(1 for result in results if result)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self._metrics:
            return {'total_queries': 0}
        
        # Group metrics by query type
        by_type: Dict[str, List[QueryMetrics]] = {}
        for metric in self._metrics:
            if metric.query_type not in by_type:
                by_type[metric.query_type] = []
            by_type[metric.query_type].append(metric)
        
        # Calculate statistics
        stats = {
            'total_queries': len(self._metrics),
            'by_type': {}
        }
        
        for query_type, type_metrics in by_type.items():
            execution_times = [m.execution_time for m in type_metrics]
            successful = [m for m in type_metrics if m.success]
            
            type_stats = {
                'count': len(type_metrics),
                'success_count': len(successful),
                'success_rate': len(successful) / len(type_metrics),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'total_keys_accessed': sum(m.keys_accessed for m in type_metrics),
                'total_keys_modified': sum(m.keys_modified for m in type_metrics),
                'avg_pipeline_size': sum(m.pipeline_size for m in type_metrics) / len(type_metrics) if type_metrics else 0
            }
            
            # Add recent errors
            recent_errors = [m.error for m in type_metrics[-10:] if m.error]
            if recent_errors:
                type_stats['recent_errors'] = recent_errors
            
            stats['by_type'][query_type] = type_stats
        
        return stats
    
    def clear_metrics(self):
        """Clear performance metrics."""
        self._metrics.clear()


# Global query optimizer instance
_query_optimizer: Optional[QueryOptimizer] = None


def get_query_optimizer() -> QueryOptimizer:
    """Get global query optimizer instance."""
    global _query_optimizer
    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer()
    return _query_optimizer


def batch_redis_operations(operations: List[Dict[str, Any]]) -> List[Any]:
    """Execute batch Redis operations with optimal performance."""
    optimizer = get_query_optimizer()
    
    # Group operations by type for efficiency
    hash_ops = []
    list_ops = []
    key_ops = []
    
    for op in operations:
        op_type = op.get('type', '')
        if op_type.startswith('h'):  # Hash operations
            hash_ops.append(op)
        elif op_type.startswith('l'):  # List operations
            list_ops.append(op)
        else:  # Key operations
            key_ops.append(op)
    
    results = []
    
    # Execute grouped operations
    if hash_ops:
        results.extend(optimizer.batch_hash_operations(hash_ops))
    
    if list_ops:
        results.extend(optimizer.batch_list_operations(list_ops))
    
    # Handle key operations (get, set, delete)
    if key_ops:
        get_keys = [op.get('key') for op in key_ops if op.get('type') == 'get']
        if get_keys:
            get_results = optimizer.batch_get(get_keys)
            results.extend(get_results.values())
        
        set_data = {op.get('key'): op.get('value') for op in key_ops if op.get('type') == 'set'}
        if set_data:
            set_result = optimizer.batch_set(set_data)
            results.append(set_result)
        
        delete_keys = [op.get('key') for op in key_ops if op.get('type') == 'delete']
        if delete_keys:
            delete_result = optimizer.batch_delete(delete_keys)
            results.append(delete_result)
    
    return results


async def async_batch_redis_operations(operations: List[Dict[str, Any]]) -> List[Any]:
    """Async version of batch Redis operations."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, batch_redis_operations, operations)


class RedisTransactionManager:
    """Manage Redis transactions with automatic retry and rollback."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._redis_client = None
    
    def _get_redis(self):
        """Get Redis client."""
        if self._redis_client is None:
            from .connection_pool import get_optimized_redis
            self._redis_client = get_optimized_redis()
        return self._redis_client
    
    @contextmanager
    def transaction(self, watch_keys: Optional[List[str]] = None):
        """Context manager for Redis transactions with WATCH/MULTI/EXEC."""
        redis_client = self._get_redis()
        
        for attempt in range(self.max_retries):
            try:
                if watch_keys:
                    redis_client.watch(*watch_keys)
                
                pipeline = redis_client.multi()
                
                try:
                    yield pipeline
                    
                    # Execute transaction
                    results = pipeline.execute()
                    return results
                
                except Exception as e:
                    # Discard transaction on error
                    pipeline.discard()
                    raise e
                
                finally:
                    # Always unwatch
                    if watch_keys:
                        redis_client.unwatch()
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Transaction failed (attempt {attempt + 1}), retrying: {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Transaction failed after {self.max_retries} attempts: {e}")
                    raise


def get_redis_stats() -> Dict[str, Any]:
    """Get comprehensive Redis statistics."""
    try:
        from .connection_pool import get_optimized_redis, get_connection_stats
        redis_client = get_optimized_redis()
        
        # Get Redis info
        info = redis_client.info()
        memory_info = redis_client.info('memory')
        stats_info = redis_client.info('stats')
        
        # Get connection pool stats
        pool_stats = get_connection_stats()
        
        # Get query optimizer stats
        optimizer_stats = get_query_optimizer().get_performance_stats()
        
        return {
            'server': {
                'version': info.get('redis_version'),
                'uptime_seconds': info.get('uptime_in_seconds'),
                'connected_clients': info.get('connected_clients'),
                'used_memory_human': memory_info.get('used_memory_human'),
                'keyspace_hits': stats_info.get('keyspace_hits', 0),
                'keyspace_misses': stats_info.get('keyspace_misses', 0),
                'total_commands_processed': stats_info.get('total_commands_processed', 0)
            },
            'connection_pool': pool_stats,
            'query_optimizer': optimizer_stats
        }
    
    except Exception as e:
        return {'error': str(e)}