"""
Unified Redis Connection Manager for Hashmancer
This module provides a single, reliable Redis connection system that consolidates
all Redis functionality and ensures proper connection management.
"""

import os
import json
import time
import redis
import redis.asyncio as aioredis
import asyncio
import threading
import logging
import weakref
from typing import Optional, Dict, Any, List, Union, AsyncGenerator, Generator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Unified Redis configuration."""
    # Connection settings
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    
    # SSL settings
    ssl: bool = False
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_ca_cert: Optional[str] = None
    
    # Pool settings
    max_connections: int = 50
    min_connections: int = 5
    max_idle_time: int = 300
    connection_timeout: int = 10
    socket_timeout: int = 10
    socket_keepalive: bool = True
    
    # Health check settings
    health_check_interval: int = 30
    max_health_check_failures: int = 3
    
    # Retry settings
    retry_on_timeout: bool = True
    max_retries: int = 3
    retry_backoff: float = 0.2
    
    # Performance settings
    decode_responses: bool = True
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Create config from environment variables and config files."""
        config = cls()
        
        # Basic connection
        config.host = os.getenv("REDIS_HOST", "localhost")
        config.port = int(os.getenv("REDIS_PORT", "6379"))
        config.db = int(os.getenv("REDIS_DB", "0"))
        
        # Password handling
        config.password = cls._read_secret("REDIS_PASSWORD")
        
        # SSL settings
        ssl_env = os.getenv("REDIS_SSL", "0")
        config.ssl = ssl_env.lower() in ("1", "true", "yes", "on")
        
        if config.ssl:
            config.ssl_cert = cls._resolve_ssl_file("REDIS_SSL_CERT")
            config.ssl_key = cls._resolve_ssl_file("REDIS_SSL_KEY")
            config.ssl_ca_cert = cls._resolve_ssl_file("REDIS_SSL_CA_CERT")
        
        # Pool settings
        config.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        config.min_connections = int(os.getenv("REDIS_MIN_CONNECTIONS", "5"))
        config.max_idle_time = int(os.getenv("REDIS_MAX_IDLE_TIME", "300"))
        config.connection_timeout = int(os.getenv("REDIS_CONNECTION_TIMEOUT", "10"))
        config.socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "10"))
        
        # Health check settings
        config.health_check_interval = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))
        config.max_health_check_failures = int(os.getenv("REDIS_MAX_HEALTH_CHECK_FAILURES", "3"))
        
        # Retry settings
        config.max_retries = int(os.getenv("REDIS_MAX_RETRIES", "3"))
        config.retry_backoff = float(os.getenv("REDIS_RETRY_BACKOFF", "0.2"))
        
        # Load from config file if it exists
        try:
            config_file = Path("config.json")
            if config_file.exists():
                with open(config_file) as f:
                    cfg = json.load(f)
                
                config.host = cfg.get("redis_host", config.host)
                config.port = cfg.get("redis_port", config.port)
                config.password = cfg.get("redis_password", config.password)
                config.ssl = cfg.get("redis_ssl", config.ssl)
                config.ssl_cert = cfg.get("redis_ssl_cert", config.ssl_cert)
                config.ssl_key = cfg.get("redis_ssl_key", config.ssl_key)
                config.ssl_ca_cert = cfg.get("redis_ssl_ca_cert", config.ssl_ca_cert)
        except Exception as e:
            logger.debug(f"Could not load config file: {e}")
        
        return config
    
    @staticmethod
    def _read_secret(var: str) -> Optional[str]:
        """Read secret from environment variable or file."""
        file_var = f"{var}_FILE"
        if file_path := os.getenv(file_var):
            try:
                return Path(file_path).read_text().strip()
            except OSError:
                return None
        return os.getenv(var)
    
    @staticmethod
    def _resolve_ssl_file(var: str) -> Optional[str]:
        """Resolve SSL file path or create temporary file from content."""
        file_var = f"{var}_FILE"
        if file_path := os.getenv(file_var):
            return file_path
        
        value = os.getenv(var)
        if not value:
            return None
        
        # Check if it's a file path
        path = Path(value)
        if path.exists():
            return str(path)
        
        # Assume it's certificate/key content - create temporary file
        fd, tmp_path = tempfile.mkstemp(suffix='.pem', prefix=f'{var.lower()}_')
        with os.fdopen(fd, "w") as tmp:
            tmp.write(value)
        return tmp_path


@dataclass
class ConnectionStats:
    """Redis connection statistics."""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    total_operations: int = 0
    failed_operations: int = 0
    avg_response_time_ms: float = 0.0
    last_health_check: float = 0.0
    health_check_failures: int = 0
    pool_hits: int = 0
    pool_misses: int = 0


class UnifiedRedisManager:
    """Unified Redis connection manager for the entire application."""
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig.from_env()
        self.stats = ConnectionStats()
        
        # Connection pools
        self._sync_pool: Optional[redis.ConnectionPool] = None
        self._async_pool: Optional[aioredis.ConnectionPool] = None
        
        # Thread safety
        self._pool_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Connection tracking
        self._active_connections: weakref.WeakSet = weakref.WeakSet()
        self._response_times: List[float] = []
        
        # Initialize
        self._initialize_pools()
        self._start_health_check()
    
    def _initialize_pools(self):
        """Initialize Redis connection pools."""
        with self._pool_lock:
            try:
                # Common connection arguments
                connection_kwargs = {
                    "host": self.config.host,
                    "port": self.config.port,
                    "password": self.config.password,
                    "db": self.config.db,
                    "socket_connect_timeout": self.config.connection_timeout,
                    "socket_timeout": self.config.socket_timeout,
                    "socket_keepalive": self.config.socket_keepalive,
                    "retry_on_timeout": self.config.retry_on_timeout,
                    "health_check_interval": self.config.health_check_interval,
                    "decode_responses": self.config.decode_responses,
                }
                
                # Add SSL settings if enabled
                if self.config.ssl:
                    connection_kwargs.update({
                        "ssl": True,
                        "ssl_ca_certs": self.config.ssl_ca_cert,
                        "ssl_certfile": self.config.ssl_cert,
                        "ssl_keyfile": self.config.ssl_key,
                    })
                
                # Synchronous pool
                self._sync_pool = redis.ConnectionPool(
                    max_connections=self.config.max_connections,
                    **connection_kwargs
                )
                
                # Asynchronous pool
                self._async_pool = aioredis.ConnectionPool(
                    max_connections=self.config.max_connections,
                    **connection_kwargs
                )
                
                logger.info(f"Redis pools initialized - Host: {self.config.host}:{self.config.port}, SSL: {self.config.ssl}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Redis pools: {e}")
                raise
    
    def _start_health_check(self):
        """Start background health check task."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._health_check_task = loop.create_task(self._health_check_loop())
        except RuntimeError:
            # No event loop running, health checks will be done synchronously
            pass
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health check error: {e}")
                with self._stats_lock:
                    self.stats.health_check_failures += 1
    
    async def _perform_health_check(self):
        """Perform health check on both pools."""
        try:
            # Test async pool
            async with self.get_async_connection() as conn:
                await conn.ping()
            
            # Test sync pool
            with self.get_sync_connection() as conn:
                conn.ping()
            
            with self._stats_lock:
                self.stats.last_health_check = time.time()
                self.stats.health_check_failures = 0
                
            logger.debug("Redis health check passed")
            
        except Exception as e:
            with self._stats_lock:
                self.stats.health_check_failures += 1
                
            logger.warning(f"Redis health check failed: {e}")
            
            if self.stats.health_check_failures >= self.config.max_health_check_failures:
                logger.error("Redis health check failures exceeded threshold, reinitializing pools")
                self._initialize_pools()
    
    @contextmanager
    def get_sync_connection(self) -> Generator[redis.Redis, None, None]:
        """Get synchronous Redis connection with automatic cleanup."""
        if not self._sync_pool:
            raise ConnectionError("Redis sync pool not initialized")
        
        connection = None
        start_time = time.time()
        
        try:
            connection = redis.Redis(connection_pool=self._sync_pool)
            self._active_connections.add(connection)
            
            with self._stats_lock:
                self.stats.active_connections += 1
                self.stats.pool_hits += 1
            
            yield connection
            
        except redis.ConnectionError as e:
            with self._stats_lock:
                self.stats.failed_connections += 1
                self.stats.pool_misses += 1
            logger.error(f"Redis sync connection error: {e}")
            raise
            
        except Exception as e:
            with self._stats_lock:
                self.stats.failed_operations += 1
            logger.error(f"Redis sync operation error: {e}")
            raise
            
        finally:
            if connection:
                try:
                    # Update statistics
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    self._response_times.append(response_time)
                    
                    # Keep only last 100 response times
                    if len(self._response_times) > 100:
                        self._response_times = self._response_times[-100:]
                    
                    with self._stats_lock:
                        self.stats.total_operations += 1
                        self.stats.active_connections = max(0, self.stats.active_connections - 1)
                        if self._response_times:
                            self.stats.avg_response_time_ms = sum(self._response_times) / len(self._response_times)
                    
                    # Remove from active connections
                    self._active_connections.discard(connection)
                    
                except Exception as e:
                    logger.debug(f"Error during sync connection cleanup: {e}")
    
    @asynccontextmanager
    async def get_async_connection(self) -> AsyncGenerator[aioredis.Redis, None]:
        """Get asynchronous Redis connection with automatic cleanup."""
        if not self._async_pool:
            raise ConnectionError("Redis async pool not initialized")
        
        connection = None
        start_time = time.time()
        
        try:
            connection = aioredis.Redis(connection_pool=self._async_pool)
            self._active_connections.add(connection)
            
            with self._stats_lock:
                self.stats.active_connections += 1
                self.stats.pool_hits += 1
            
            yield connection
            
        except aioredis.ConnectionError as e:
            with self._stats_lock:
                self.stats.failed_connections += 1
                self.stats.pool_misses += 1
            logger.error(f"Redis async connection error: {e}")
            raise
            
        except Exception as e:
            with self._stats_lock:
                self.stats.failed_operations += 1
            logger.error(f"Redis async operation error: {e}")
            raise
            
        finally:
            if connection:
                try:
                    # Update statistics
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    self._response_times.append(response_time)
                    
                    # Keep only last 100 response times
                    if len(self._response_times) > 100:
                        self._response_times = self._response_times[-100:]
                    
                    with self._stats_lock:
                        self.stats.total_operations += 1
                        self.stats.active_connections = max(0, self.stats.active_connections - 1)
                        if self._response_times:
                            self.stats.avg_response_time_ms = sum(self._response_times) / len(self._response_times)
                    
                    # Close async connection
                    await connection.close()
                    
                    # Remove from active connections
                    self._active_connections.discard(connection)
                    
                except Exception as e:
                    logger.debug(f"Error during async connection cleanup: {e}")
    
    def execute_with_retry(self, operation, *args, **kwargs):
        """Execute Redis operation with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                with self.get_sync_connection() as conn:
                    return operation(conn, *args, **kwargs)
                    
            except redis.RedisError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    sleep_time = self.config.retry_backoff * (2 ** attempt)
                    time.sleep(sleep_time)
                    logger.debug(f"Redis operation retry {attempt + 1}/{self.config.max_retries} after {sleep_time}s")
                continue
        
        logger.error(f"Redis operation failed after {self.config.max_retries} attempts: {last_error}")
        raise last_error
    
    async def execute_async_with_retry(self, operation, *args, **kwargs):
        """Execute async Redis operation with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.get_async_connection() as conn:
                    return await operation(conn, *args, **kwargs)
                    
            except aioredis.RedisError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    sleep_time = self.config.retry_backoff * (2 ** attempt)
                    await asyncio.sleep(sleep_time)
                    logger.debug(f"Async Redis operation retry {attempt + 1}/{self.config.max_retries} after {sleep_time}s")
                continue
        
        logger.error(f"Async Redis operation failed after {self.config.max_retries} attempts: {last_error}")
        raise last_error
    
    def get_legacy_sync_client(self) -> redis.Redis:
        """Get legacy sync client for backward compatibility."""
        if not self._sync_pool:
            raise ConnectionError("Redis sync pool not initialized")
        return redis.Redis(connection_pool=self._sync_pool)
    
    async def get_legacy_async_client(self) -> aioredis.Redis:
        """Get legacy async client for backward compatibility."""
        if not self._async_pool:
            raise ConnectionError("Redis async pool not initialized")
        return aioredis.Redis(connection_pool=self._async_pool)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Redis statistics."""
        with self._stats_lock:
            stats = {
                "config": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "ssl": self.config.ssl,
                    "max_connections": self.config.max_connections,
                    "health_check_interval": self.config.health_check_interval,
                },
                "connections": {
                    "total": self.stats.total_connections,
                    "active": self.stats.active_connections,
                    "failed": self.stats.failed_connections,
                },
                "operations": {
                    "total": self.stats.total_operations,
                    "failed": self.stats.failed_operations,
                    "success_rate": (
                        (self.stats.total_operations - self.stats.failed_operations) / 
                        max(1, self.stats.total_operations) * 100
                    ),
                },
                "performance": {
                    "avg_response_time_ms": round(self.stats.avg_response_time_ms, 2),
                    "pool_hits": self.stats.pool_hits,
                    "pool_misses": self.stats.pool_misses,
                    "hit_ratio": (
                        self.stats.pool_hits / max(1, self.stats.pool_hits + self.stats.pool_misses) * 100
                    ),
                },
                "health": {
                    "last_check": self.stats.last_health_check,
                    "check_failures": self.stats.health_check_failures,
                    "status": "healthy" if self.stats.health_check_failures < self.config.max_health_check_failures else "unhealthy",
                },
            }
        
        # Add pool-specific stats
        if self._sync_pool:
            try:
                stats["sync_pool"] = {
                    "created_connections": getattr(self._sync_pool, 'created_connections', 0),
                    "available_connections": len(getattr(self._sync_pool, '_available_connections', [])),
                    "in_use_connections": len(getattr(self._sync_pool, '_in_use_connections', [])),
                }
            except Exception:
                pass
        
        if self._async_pool:
            try:
                stats["async_pool"] = {
                    "created_connections": getattr(self._async_pool, '_created_connections', 0),
                    "available_connections": len(getattr(self._async_pool, '_pool', [])),
                }
            except Exception:
                pass
        
        return stats
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test both sync and async connections."""
        results = {
            "sync": {"success": False, "latency_ms": None, "error": None},
            "async": {"success": False, "latency_ms": None, "error": None}
        }
        
        # Test sync connection
        try:
            start_time = time.time()
            with self.get_sync_connection() as conn:
                conn.ping()
            
            results["sync"]["success"] = True
            results["sync"]["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            
        except Exception as e:
            results["sync"]["error"] = str(e)
        
        # Test async connection
        try:
            start_time = time.time()
            async with self.get_async_connection() as conn:
                await conn.ping()
            
            results["async"]["success"] = True
            results["async"]["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            
        except Exception as e:
            results["async"]["error"] = str(e)
        
        return results
    
    async def shutdown(self):
        """Gracefully shutdown the Redis manager."""
        logger.info("Shutting down Redis manager...")
        
        # Stop health check task
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
        with self._pool_lock:
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
        
        logger.info("Redis manager shutdown complete")


# Global manager instance
_redis_manager: Optional[UnifiedRedisManager] = None
_manager_lock = threading.RLock()


def get_redis_manager() -> UnifiedRedisManager:
    """Get the global Redis manager instance."""
    global _redis_manager
    
    if _redis_manager is None:
        with _manager_lock:
            if _redis_manager is None:
                _redis_manager = UnifiedRedisManager()
    
    return _redis_manager


def get_redis(**kwargs) -> redis.Redis:
    """Get synchronous Redis client for backward compatibility."""
    manager = get_redis_manager()
    return manager.get_legacy_sync_client()


async def get_redis_async(**kwargs) -> aioredis.Redis:
    """Get asynchronous Redis client for backward compatibility."""
    manager = get_redis_manager()
    return await manager.get_legacy_async_client()


# Preferred context managers for proper connection handling
@contextmanager
def redis_connection():
    """Context manager for synchronous Redis connection."""
    manager = get_redis_manager()
    with manager.get_sync_connection() as conn:
        yield conn


@asynccontextmanager
async def redis_async_connection():
    """Context manager for asynchronous Redis connection."""
    manager = get_redis_manager()
    async with manager.get_async_connection() as conn:
        yield conn


# Decorators for automatic connection management
def with_redis_sync(func):
    """Decorator to inject sync Redis connection as first argument."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with redis_connection() as conn:
            return func(conn, *args, **kwargs)
    return wrapper


def with_redis_async(func):
    """Decorator to inject async Redis connection as first argument."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        async with redis_async_connection() as conn:
            return await func(conn, *args, **kwargs)
    return wrapper


def redis_retry(max_retries: int = 3):
    """Decorator to add retry logic to Redis operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_redis_manager()
            return manager.execute_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator


def redis_async_retry(max_retries: int = 3):
    """Decorator to add retry logic to async Redis operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_redis_manager()
            return await manager.execute_async_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator


async def shutdown_redis():
    """Shutdown the global Redis manager."""
    global _redis_manager
    if _redis_manager:
        await _redis_manager.shutdown()
        _redis_manager = None


# Health check functions
def redis_health_check() -> Dict[str, Any]:
    """Perform synchronous Redis health check."""
    try:
        manager = get_redis_manager()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(manager.test_connection())
        finally:
            loop.close()
        return result
    except Exception as e:
        return {
            "sync": {"success": False, "error": str(e)},
            "async": {"success": False, "error": str(e)}
        }


async def redis_async_health_check() -> Dict[str, Any]:
    """Perform asynchronous Redis health check."""
    try:
        manager = get_redis_manager()
        return await manager.test_connection()
    except Exception as e:
        return {
            "sync": {"success": False, "error": str(e)},
            "async": {"success": False, "error": str(e)}
        }


def get_redis_stats() -> Dict[str, Any]:
    """Get Redis statistics."""
    try:
        manager = get_redis_manager()
        return manager.get_stats()
    except Exception as e:
        return {"error": str(e)}