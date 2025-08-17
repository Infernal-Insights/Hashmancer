"""Advanced caching layer with multiple backends and intelligent invalidation."""

import time
import json
import hashlib
import asyncio
import logging
from typing import Any, Dict, Optional, Callable, Union, List
from functools import wraps
from threading import RLock
from collections import OrderedDict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    last_access: Optional[float] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.last_access is None:
            self.last_access = self.created_at
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None):
        """Set value in cache."""
        current_time = time.time()
        expires_at = current_time + ttl if ttl else None
        
        with self._lock:
            # Remove oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(
                value=value,
                created_at=current_time,
                expires_at=expires_at,
                tags=tags or []
            )
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            return self._cache.pop(key, None) is not None
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with given tag."""
        count = 0
        with self._lock:
            to_remove = []
            for key, entry in self._cache.items():
                if tag in entry.tags:
                    to_remove.append(key)
            
            for key in to_remove:
                del self._cache[key]
                count += 1
        
        return count
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        count = 0
        with self._lock:
            to_remove = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    to_remove.append(key)
            
            for key in to_remove:
                del self._cache[key]
                count += 1
        
        return count
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(1 for entry in self._cache.values() if entry.is_expired())
            
            if total_entries > 0:
                avg_access_count = sum(entry.access_count for entry in self._cache.values()) / total_entries
                oldest_entry = min(self._cache.values(), key=lambda e: e.created_at)
                newest_entry = max(self._cache.values(), key=lambda e: e.created_at)
                age_span = newest_entry.created_at - oldest_entry.created_at
            else:
                avg_access_count = 0
                age_span = 0
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'max_size': self.max_size,
                'utilization': total_entries / self.max_size if self.max_size > 0 else 0,
                'avg_access_count': avg_access_count,
                'age_span_seconds': age_span
            }


class CacheManager:
    """Multi-tier caching system with Redis backing."""
    
    def __init__(self, memory_cache_size: int = 1000, default_ttl: int = 300):
        self.memory_cache = LRUCache(memory_cache_size)
        self.default_ttl = default_ttl
        self._redis_client = None
        self._last_cleanup = time.time()
        self._cleanup_interval = 60  # seconds
    
    def _get_redis(self):
        """Get Redis client lazily."""
        if self._redis_client is None:
            from .connection_pool import get_optimized_redis
            self._redis_client = get_optimized_redis('read')
        return self._redis_client
    
    def _make_key(self, key: str, prefix: str = "cache") -> str:
        """Make Redis key with prefix."""
        return f"{prefix}:{key}"
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        return json.dumps(value, default=str)
    
    def _deserialize(self, data: str) -> Any:
        """Deserialize value from storage."""
        return json.loads(data)
    
    def _periodic_cleanup(self):
        """Perform periodic cleanup of expired entries."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            expired = self.memory_cache.cleanup_expired()
            if expired > 0:
                logger.debug(f"Cleaned up {expired} expired cache entries")
            self._last_cleanup = current_time
    
    async def aget(self, key: str, default: Any = None) -> Any:
        """Async get value from cache."""
        return self.get(key, default)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from multi-tier cache."""
        self._periodic_cleanup()
        
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        try:
            redis_key = self._make_key(key)
            redis_value = self._get_redis().get(redis_key)
            if redis_value:
                deserialized = self._deserialize(redis_value)
                # Populate memory cache
                self.memory_cache.set(key, deserialized, ttl=self.default_ttl)
                return deserialized
        except Exception as e:
            logger.warning(f"Redis cache get failed: {e}")
        
        return default
    
    async def aset(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None):
        """Async set value in cache."""
        return self.set(key, value, ttl, tags)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None):
        """Set value in multi-tier cache."""
        ttl = ttl or self.default_ttl
        
        # Set in memory cache
        self.memory_cache.set(key, value, ttl, tags)
        
        # Set in Redis cache
        try:
            redis_key = self._make_key(key)
            serialized = self._serialize(value)
            self._get_redis().setex(redis_key, ttl, serialized)
            
            # Store tags separately for invalidation
            if tags:
                for tag in tags:
                    tag_key = self._make_key(f"tag:{tag}", "cache_tags")
                    self._get_redis().sadd(tag_key, key)
                    self._get_redis().expire(tag_key, ttl + 60)  # Keep tags slightly longer
        except Exception as e:
            logger.warning(f"Redis cache set failed: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from all cache tiers."""
        # Delete from memory cache
        memory_deleted = self.memory_cache.delete(key)
        
        # Delete from Redis cache
        redis_deleted = False
        try:
            redis_key = self._make_key(key)
            redis_deleted = bool(self._get_redis().delete(redis_key))
        except Exception as e:
            logger.warning(f"Redis cache delete failed: {e}")
        
        return memory_deleted or redis_deleted
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate entries by tag across all tiers."""
        total_deleted = 0
        
        # Invalidate in memory cache
        total_deleted += self.memory_cache.invalidate_by_tag(tag)
        
        # Invalidate in Redis cache
        try:
            redis_client = self._get_redis()
            tag_key = self._make_key(f"tag:{tag}", "cache_tags")
            keys = redis_client.smembers(tag_key)
            
            if keys:
                # Delete cache entries
                redis_keys = [self._make_key(key.decode() if isinstance(key, bytes) else key) for key in keys]
                total_deleted += redis_client.delete(*redis_keys)
                
                # Delete tag index
                redis_client.delete(tag_key)
        except Exception as e:
            logger.warning(f"Redis tag invalidation failed: {e}")
        
        return total_deleted
    
    def clear(self):
        """Clear all cache tiers."""
        self.memory_cache.clear()
        
        try:
            redis_client = self._get_redis()
            keys = redis_client.keys(self._make_key("*"))
            if keys:
                redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis cache clear failed: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.stats()
        
        redis_stats = {'error': 'Redis unavailable'}
        try:
            redis_client = self._get_redis()
            info = redis_client.info('memory')
            redis_stats = {
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'connected_clients': redis_client.info().get('connected_clients', 0)
            }
        except Exception as e:
            redis_stats['error'] = str(e)
        
        return {
            'memory_cache': memory_stats,
            'redis_cache': redis_stats,
            'default_ttl': self.default_ttl
        }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cache_with_ttl(ttl: int = 300, tags: Optional[List[str]] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results with TTL."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Generate key from function name and arguments
                args_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__module__}.{func.__name__}:{hashlib.md5(args_str.encode()).hexdigest()}"
            
            cache_manager = get_cache_manager()
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl, tags)
            return result
        
        # Add cache management methods to the function
        wrapper.cache_invalidate = lambda *args, **kwargs: get_cache_manager().delete(
            key_func(*args, **kwargs) if key_func else f"{func.__module__}.{func.__name__}:{hashlib.md5((str(args) + str(sorted(kwargs.items()))).encode()).hexdigest()}"
        )
        wrapper.cache_invalidate_by_tag = lambda tag: get_cache_manager().invalidate_by_tag(tag)
        
        return wrapper
    return decorator


def invalidate_cache_by_tag(tag: str) -> int:
    """Invalidate all cached entries with given tag."""
    return get_cache_manager().invalidate_by_tag(tag)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return get_cache_manager().stats()