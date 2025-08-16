"""
Improved Redis Connection Pooling
Fixes connection leaks, implements proper health checks, and provides connection monitoring
"""

import redis
import redis.asyncio as aioredis
import asyncio
import threading
import time
import logging
import weakref
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class RedisPoolConfig:
    """Redis connection pool configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    
    # Pool settings
    max_connections: int = 50
    min_connections: int = 5
    max_idle_time: int = 300  # seconds
    connection_timeout: int = 5
    socket_timeout: int = 5
    
    # Health check settings
    health_check_interval: int = 30
    max_health_check_failures: int = 3
    
    # Retry settings
    retry_on_timeout: bool = True
    max_retries: int = 3
    retry_backoff: float = 0.1
    
    @classmethod
    def from_env(cls) -> 'RedisPoolConfig':
        """Create config from environment variables."""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            db=int(os.getenv("REDIS_DB", "0")),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50")),
            min_connections=int(os.getenv("REDIS_MIN_CONNECTIONS", "5")),
            max_idle_time=int(os.getenv("REDIS_MAX_IDLE_TIME", "300")),
            connection_timeout=int(os.getenv("REDIS_CONNECTION_TIMEOUT", "5")),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
        )


@dataclass
class ConnectionStats:
    """Connection pool statistics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_commands: int = 0
    failed_commands: int = 0
    avg_response_time: float = 0.0
    pool_hits: int = 0
    pool_misses: int = 0
    last_health_check: float = 0
    health_check_failures: int = 0


class ImprovedRedisPool:
    """Enhanced Redis connection pool with proper lifecycle management."""
    
    def __init__(self, config: RedisPoolConfig):
        self.config = config
        self.stats = ConnectionStats()
        
        # Connection pools
        self._sync_pool: Optional[redis.ConnectionPool] = None
        self._async_pool: Optional[aioredis.ConnectionPool] = None
        
        # Pool management
        self._pool_lock = threading.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Connection tracking
        self._active_connections: weakref.WeakSet = weakref.WeakSet()
        self._connection_times: Dict[str, float] = {}
        self._response_times: List[float] = []
        
        # Initialize pools
        self._initialize_pools()
        
        # Start health checking
        self._start_health_check()
    
    def _initialize_pools(self):
        """Initialize Redis connection pools."""
        try:
            # Sync pool
            self._sync_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_connect_timeout=self.config.connection_timeout,
                socket_timeout=self.config.socket_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
                decode_responses=True
            )
            
            # Async pool
            self._async_pool = aioredis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_connect_timeout=self.config.connection_timeout,
                socket_timeout=self.config.socket_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
                decode_responses=True
            )
            
            logger.info("Redis connection pools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis pools: {e}")
            raise
    
    def _start_health_check(self):
        """Start background health check task."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                self.stats.health_check_failures += 1
    
    async def _perform_health_check(self):
        """Perform health check on pools."""
        try:
            # Test async pool
            async with self.get_async_connection() as redis_client:
                await redis_client.ping()
            
            # Test sync pool
            with self.get_sync_connection() as redis_client:
                redis_client.ping()
            
            self.stats.last_health_check = time.time()
            self.stats.health_check_failures = 0
            logger.debug("Redis health check passed")
            
        except Exception as e:
            self.stats.health_check_failures += 1
            logger.warning(f"Redis health check failed: {e}")
            
            if self.stats.health_check_failures >= self.config.max_health_check_failures:
                logger.error("Redis health check failures exceeded threshold")
                # Could trigger pool recreation here
    
    @contextmanager
    def get_sync_connection(self):
        """Get synchronous Redis connection with proper cleanup."""
        if not self._sync_pool:
            raise Exception("Redis sync pool not initialized")
        
        connection = None
        start_time = time.time()
        
        try:
            connection = redis.Redis(connection_pool=self._sync_pool)
            self._active_connections.add(connection)
            self.stats.pool_hits += 1
            yield connection
            
        except redis.ConnectionError as e:
            self.stats.failed_connections += 1
            self.stats.pool_misses += 1
            logger.error(f"Redis connection error: {e}")
            raise
        except Exception as e:
            self.stats.failed_commands += 1
            logger.error(f"Redis operation error: {e}")
            raise
        finally:
            if connection:
                try:
                    # Record response time
                    response_time = time.time() - start_time
                    self._response_times.append(response_time)
                    
                    # Keep only last 100 response times
                    if len(self._response_times) > 100:
                        self._response_times = self._response_times[-100:]
                    
                    # Update average response time
                    self.stats.avg_response_time = sum(self._response_times) / len(self._response_times)
                    
                    # Connection is automatically returned to pool when Redis object is deleted
                    if connection in self._active_connections:
                        self._active_connections.discard(connection)
                
                except Exception as e:
                    logger.debug(f"Error during connection cleanup: {e}")
            
            self.stats.total_commands += 1
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get asynchronous Redis connection with proper cleanup."""
        if not self._async_pool:
            raise Exception("Redis async pool not initialized")
        
        connection = None
        start_time = time.time()
        
        try:
            connection = aioredis.Redis(connection_pool=self._async_pool)
            self._active_connections.add(connection)
            self.stats.pool_hits += 1
            yield connection
            
        except aioredis.ConnectionError as e:
            self.stats.failed_connections += 1
            self.stats.pool_misses += 1
            logger.error(f"Async Redis connection error: {e}")
            raise
        except Exception as e:
            self.stats.failed_commands += 1
            logger.error(f"Async Redis operation error: {e}")
            raise
        finally:
            if connection:
                try:
                    # Record response time
                    response_time = time.time() - start_time
                    self._response_times.append(response_time)
                    
                    # Keep only last 100 response times
                    if len(self._response_times) > 100:
                        self._response_times = self._response_times[-100:]
                    
                    # Update average response time
                    if self._response_times:
                        self.stats.avg_response_time = sum(self._response_times) / len(self._response_times)
                    
                    # Close async connection
                    await connection.close()
                    
                    if connection in self._active_connections:
                        self._active_connections.discard(connection)
                
                except Exception as e:
                    logger.debug(f"Error during async connection cleanup: {e}")
            
            self.stats.total_commands += 1
    
    def get_legacy_sync_client(self) -> redis.Redis:
        """Get legacy sync client for backward compatibility."""
        if not self._sync_pool:
            raise Exception("Redis sync pool not initialized")
        return redis.Redis(connection_pool=self._sync_pool)
    
    async def get_legacy_async_client(self) -> aioredis.Redis:
        """Get legacy async client for backward compatibility."""
        if not self._async_pool:
            raise Exception("Redis async pool not initialized")
        return aioredis.Redis(connection_pool=self._async_pool)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get detailed pool statistics."""
        sync_pool_stats = {}
        async_pool_stats = {}
        
        if self._sync_pool:
            sync_pool_stats = {
                "created_connections": getattr(self._sync_pool, 'created_connections', 0),
                "available_connections": len(getattr(self._sync_pool, '_available_connections', [])),
                "in_use_connections": len(getattr(self._sync_pool, '_in_use_connections', [])),
                "max_connections": self._sync_pool.max_connections,
            }
        
        if self._async_pool:
            async_pool_stats = {
                "created_connections": getattr(self._async_pool, '_created_connections', 0),
                "available_connections": len(getattr(self._async_pool, '_pool', [])),
                "max_connections": self._async_pool.max_connections,
            }
        
        return {
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "max_connections": self.config.max_connections,
                "health_check_interval": self.config.health_check_interval,
            },
            "stats": {
                "total_commands": self.stats.total_commands,
                "failed_commands": self.stats.failed_commands,
                "failed_connections": self.stats.failed_connections,
                "pool_hits": self.stats.pool_hits,
                "pool_misses": self.stats.pool_misses,
                "avg_response_time_ms": round(self.stats.avg_response_time * 1000, 2),
                "health_check_failures": self.stats.health_check_failures,
                "last_health_check": self.stats.last_health_check,
                "active_connections": len(self._active_connections),
            },
            "sync_pool": sync_pool_stats,
            "async_pool": async_pool_stats,
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test both sync and async connections."""
        results = {
            "sync": {"success": False, "latency_ms": None, "error": None},
            "async": {"success": False, "latency_ms": None, "error": None}
        }
        
        # Test sync connection
        try:
            start_time = time.time()
            with self.get_sync_connection() as redis_client:
                redis_client.ping()
            
            results["sync"]["success"] = True
            results["sync"]["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            
        except Exception as e:
            results["sync"]["error"] = str(e)
        
        # Test async connection
        try:
            start_time = time.time()
            async with self.get_async_connection() as redis_client:
                await redis_client.ping()
            
            results["async"]["success"] = True
            results["async"]["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            
        except Exception as e:
            results["async"]["error"] = str(e)
        
        return results
    
    async def shutdown(self):
        """Gracefully shutdown the Redis pool."""
        logger.info("Shutting down Redis connection pool...")
        
        # Stop health check
        self._shutdown_event.set()
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all active connections
        for connection in list(self._active_connections):
            try:
                if hasattr(connection, 'close'):
                    if asyncio.iscoroutinefunction(connection.close):
                        await connection.close()
                    else:
                        connection.close()
            except Exception as e:
                logger.debug(f"Error closing connection: {e}")
        
        # Disconnect pools
        if self._sync_pool:
            try:
                self._sync_pool.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting sync pool: {e}")
        
        if self._async_pool:
            try:
                await self._async_pool.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting async pool: {e}")
        
        logger.info("Redis connection pool shutdown complete")


# Global pool instance
_redis_pool: Optional[ImprovedRedisPool] = None
_pool_lock = threading.Lock()


def get_redis_pool() -> ImprovedRedisPool:
    """Get the global Redis pool instance."""
    global _redis_pool
    
    if _redis_pool is None:
        with _pool_lock:
            if _redis_pool is None:
                config = RedisPoolConfig.from_env()
                _redis_pool = ImprovedRedisPool(config)
    
    return _redis_pool


def get_redis_sync() -> redis.Redis:
    """Get synchronous Redis client (legacy compatibility)."""
    pool = get_redis_pool()
    return pool.get_legacy_sync_client()


async def get_redis_async() -> aioredis.Redis:
    """Get asynchronous Redis client (legacy compatibility)."""
    pool = get_redis_pool()
    return await pool.get_legacy_async_client()


# Context managers for proper connection handling
@contextmanager
def redis_sync_connection():
    """Context manager for sync Redis connection."""
    pool = get_redis_pool()
    with pool.get_sync_connection() as connection:
        yield connection


@asynccontextmanager
async def redis_async_connection():
    """Context manager for async Redis connection."""
    pool = get_redis_pool()
    async with pool.get_async_connection() as connection:
        yield connection


# Decorator for automatic Redis connection management
def with_redis(sync: bool = True):
    """Decorator to inject Redis connection into function."""
    def decorator(func):
        if sync:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with redis_sync_connection() as redis_client:
                    return func(redis_client, *args, **kwargs)
            return sync_wrapper
        else:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with redis_async_connection() as redis_client:
                    return await func(redis_client, *args, **kwargs)
            return async_wrapper
    
    return decorator


async def shutdown_redis_pool():
    """Shutdown the global Redis pool."""
    global _redis_pool
    if _redis_pool:
        await _redis_pool.shutdown()
        _redis_pool = None