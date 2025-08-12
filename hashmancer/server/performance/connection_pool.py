"""Optimized Redis connection pooling with clustering support."""

import os
import time
import redis
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from threading import Lock
from redis.sentinel import Sentinel
from rediscluster import RedisCluster

logger = logging.getLogger(__name__)


class RedisConnectionPool:
    """High-performance Redis connection pool with automatic failover."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_config()
        self._pools: Dict[str, redis.ConnectionPool] = {}
        self._cluster_client: Optional[RedisCluster] = None
        self._sentinel_client: Optional[Sentinel] = None
        self._lock = Lock()
        self._health_check_interval = 30
        self._last_health_check = 0
        
        self._initialize_connections()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load Redis configuration from environment."""
        return {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'password': os.getenv('REDIS_PASSWORD'),
            'db': int(os.getenv('REDIS_DB', '0')),
            'ssl': os.getenv('REDIS_SSL', '0') == '1',
            'ssl_ca_cert': os.getenv('REDIS_SSL_CA_CERT'),
            'ssl_cert': os.getenv('REDIS_SSL_CERT'),
            'ssl_key': os.getenv('REDIS_SSL_KEY'),
            
            # Performance settings
            'max_connections': int(os.getenv('REDIS_MAX_CONNECTIONS', '100')),
            'max_connections_per_pool': int(os.getenv('REDIS_MAX_CONNECTIONS_PER_POOL', '50')),
            'connection_timeout': int(os.getenv('REDIS_CONNECTION_TIMEOUT', '5')),
            'socket_timeout': int(os.getenv('REDIS_SOCKET_TIMEOUT', '5')),
            'socket_keepalive': os.getenv('REDIS_SOCKET_KEEPALIVE', '1') == '1',
            'socket_keepalive_options': {
                'TCP_KEEPIDLE': 1,
                'TCP_KEEPINTVL': 3,
                'TCP_KEEPCNT': 5,
            },
            
            # Clustering
            'cluster_enabled': os.getenv('REDIS_CLUSTER_ENABLED', '0') == '1',
            'cluster_nodes': os.getenv('REDIS_CLUSTER_NODES', '').split(','),
            
            # Sentinel
            'sentinel_enabled': os.getenv('REDIS_SENTINEL_ENABLED', '0') == '1',
            'sentinel_hosts': os.getenv('REDIS_SENTINEL_HOSTS', '').split(','),
            'sentinel_service': os.getenv('REDIS_SENTINEL_SERVICE', 'mymaster'),
        }
    
    def _initialize_connections(self):
        """Initialize connection pools based on configuration."""
        try:
            if self.config['cluster_enabled'] and self.config['cluster_nodes']:
                self._initialize_cluster()
            elif self.config['sentinel_enabled'] and self.config['sentinel_hosts']:
                self._initialize_sentinel()
            else:
                self._initialize_standalone()
        except Exception as e:
            logger.error(f"Failed to initialize Redis connections: {e}")
            raise
    
    def _initialize_cluster(self):
        """Initialize Redis cluster connection."""
        startup_nodes = []
        for node in self.config['cluster_nodes']:
            if ':' in node:
                host, port = node.split(':', 1)
                startup_nodes.append({'host': host.strip(), 'port': int(port)})
        
        if not startup_nodes:
            raise ValueError("No valid cluster nodes provided")
        
        self._cluster_client = RedisCluster(
            startup_nodes=startup_nodes,
            password=self.config['password'],
            ssl=self.config['ssl'],
            ssl_ca_certs=self.config['ssl_ca_cert'],
            ssl_certfile=self.config['ssl_cert'],
            ssl_keyfile=self.config['ssl_key'],
            socket_timeout=self.config['socket_timeout'],
            socket_connect_timeout=self.config['connection_timeout'],
            socket_keepalive=self.config['socket_keepalive'],
            socket_keepalive_options=self.config['socket_keepalive_options'],
            max_connections=self.config['max_connections'],
            skip_full_coverage_check=True,
            decode_responses=False
        )
        
        logger.info(f"Initialized Redis cluster with {len(startup_nodes)} nodes")
    
    def _initialize_sentinel(self):
        """Initialize Redis Sentinel connection."""
        sentinel_hosts = []
        for host in self.config['sentinel_hosts']:
            if ':' in host:
                host, port = host.split(':', 1)
                sentinel_hosts.append((host.strip(), int(port)))
            else:
                sentinel_hosts.append((host.strip(), 26379))
        
        self._sentinel_client = Sentinel(
            sentinel_hosts,
            socket_timeout=self.config['socket_timeout'],
            socket_connect_timeout=self.config['connection_timeout'],
            socket_keepalive=self.config['socket_keepalive'],
            socket_keepalive_options=self.config['socket_keepalive_options']
        )
        
        logger.info(f"Initialized Redis Sentinel with {len(sentinel_hosts)} sentinels")
    
    def _initialize_standalone(self):
        """Initialize standalone Redis connection pools."""
        # Main connection pool
        self._pools['main'] = redis.ConnectionPool(
            host=self.config['host'],
            port=self.config['port'],
            password=self.config['password'],
            db=self.config['db'],
            ssl=self.config['ssl'],
            ssl_ca_certs=self.config['ssl_ca_cert'],
            ssl_certfile=self.config['ssl_cert'],
            ssl_keyfile=self.config['ssl_key'],
            max_connections=self.config['max_connections_per_pool'],
            socket_timeout=self.config['socket_timeout'],
            socket_connect_timeout=self.config['connection_timeout'],
            socket_keepalive=self.config['socket_keepalive'],
            socket_keepalive_options=self.config['socket_keepalive_options'],
            decode_responses=False
        )
        
        # Separate pools for different operations
        for pool_name in ['read', 'write', 'pubsub']:
            self._pools[pool_name] = redis.ConnectionPool(
                host=self.config['host'],
                port=self.config['port'],
                password=self.config['password'],
                db=self.config['db'],
                ssl=self.config['ssl'],
                ssl_ca_certs=self.config['ssl_ca_cert'],
                ssl_certfile=self.config['ssl_cert'],
                ssl_keyfile=self.config['ssl_key'],
                max_connections=min(20, self.config['max_connections_per_pool']),
                socket_timeout=self.config['socket_timeout'],
                socket_connect_timeout=self.config['connection_timeout'],
                socket_keepalive=self.config['socket_keepalive'],
                socket_keepalive_options=self.config['socket_keepalive_options'],
                decode_responses=False
            )
        
        logger.info(f"Initialized Redis standalone pools: {list(self._pools.keys())}")
    
    def get_connection(self, operation_type: str = 'main') -> redis.Redis:
        """Get Redis connection optimized for specific operation type."""
        self._health_check()
        
        if self._cluster_client:
            return self._cluster_client
        
        if self._sentinel_client:
            if operation_type in ['read', 'get', 'hget', 'smembers']:
                return self._sentinel_client.slave_for(
                    self.config['sentinel_service'],
                    password=self.config['password']
                )
            else:
                return self._sentinel_client.master_for(
                    self.config['sentinel_service'],
                    password=self.config['password']
                )
        
        pool_name = operation_type if operation_type in self._pools else 'main'
        return redis.Redis(connection_pool=self._pools[pool_name])
    
    @contextmanager
    def get_pipeline(self, operation_type: str = 'main', transaction: bool = True):
        """Get Redis pipeline for batched operations."""
        conn = self.get_connection(operation_type)
        pipeline = conn.pipeline(transaction=transaction)
        try:
            yield pipeline
        finally:
            pipeline.reset()
    
    def _health_check(self):
        """Perform periodic health checks on connections."""
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return
        
        with self._lock:
            if current_time - self._last_health_check < self._health_check_interval:
                return
            
            try:
                conn = self.get_connection('read')
                conn.ping()
                self._last_health_check = current_time
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
                # Reinitialize connections on failure
                try:
                    self._initialize_connections()
                    self._last_health_check = current_time
                except Exception as init_e:
                    logger.error(f"Failed to reinitialize Redis connections: {init_e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        stats = {
            'config': {
                'cluster_enabled': self.config['cluster_enabled'],
                'sentinel_enabled': self.config['sentinel_enabled'],
                'max_connections': self.config['max_connections']
            },
            'pools': {}
        }
        
        for name, pool in self._pools.items():
            stats['pools'][name] = {
                'created_connections': pool.created_connections,
                'available_connections': len(pool._available_connections),
                'in_use_connections': len(pool._in_use_connections)
            }
        
        return stats
    
    def close(self):
        """Close all connection pools."""
        for pool in self._pools.values():
            pool.disconnect()
        
        if self._cluster_client:
            self._cluster_client.connection_pool.disconnect()
        
        logger.info("Closed all Redis connection pools")


# Global connection pool instance
_connection_pool: Optional[RedisConnectionPool] = None


def get_optimized_redis(operation_type: str = 'main') -> redis.Redis:
    """Get optimized Redis connection from global pool."""
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = RedisConnectionPool()
    
    return _connection_pool.get_connection(operation_type)


def get_redis_pipeline(operation_type: str = 'main', transaction: bool = True):
    """Get Redis pipeline from global pool."""
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = RedisConnectionPool()
    
    return _connection_pool.get_pipeline(operation_type, transaction)


def get_connection_stats() -> Dict[str, Any]:
    """Get connection pool statistics."""
    global _connection_pool
    
    if _connection_pool is None:
        return {'error': 'Connection pool not initialized'}
    
    return _connection_pool.get_stats()


def close_connection_pool():
    """Close the global connection pool."""
    global _connection_pool
    
    if _connection_pool is not None:
        _connection_pool.close()
        _connection_pool = None