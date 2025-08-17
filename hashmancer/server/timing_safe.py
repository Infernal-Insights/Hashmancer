"""
Timing Attack Protection
Provides timing-safe comparison functions and other protections against timing attacks
"""

import hmac
import time
import secrets
import hashlib
from typing import Union, Optional, Any, Dict
import asyncio
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def constant_time_compare(a: Union[str, bytes], b: Union[str, bytes]) -> bool:
    """
    Perform constant-time comparison of two strings or byte sequences.
    
    This prevents timing attacks by ensuring the comparison always takes
    the same amount of time regardless of where the difference occurs.
    """
    # Convert to bytes if necessary
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')
    
    # Use hmac.compare_digest for constant-time comparison
    return hmac.compare_digest(a, b)


def timing_safe_hash_compare(hash1: str, hash2: str) -> bool:
    """
    Safely compare two hash strings in constant time.
    
    Args:
        hash1: First hash string
        hash2: Second hash string
    
    Returns:
        True if hashes match, False otherwise
    """
    # Normalize hash strings (remove whitespace, convert to lowercase)
    normalized_hash1 = hash1.strip().lower()
    normalized_hash2 = hash2.strip().lower()
    
    return constant_time_compare(normalized_hash1, normalized_hash2)


def timing_safe_token_compare(token1: str, token2: str) -> bool:
    """
    Safely compare two authentication tokens in constant time.
    
    Args:
        token1: First token
        token2: Second token
    
    Returns:
        True if tokens match, False otherwise
    """
    return constant_time_compare(token1, token2)


class TimingSafeValidator:
    """Provides timing-safe validation methods."""
    
    def __init__(self, min_operation_time: float = 0.01):
        """
        Initialize validator with minimum operation time.
        
        Args:
            min_operation_time: Minimum time each operation should take (seconds)
        """
        self.min_operation_time = min_operation_time
    
    def _ensure_minimum_time(self, start_time: float):
        """Ensure operation takes at least minimum time."""
        elapsed = time.time() - start_time
        if elapsed < self.min_operation_time:
            time.sleep(self.min_operation_time - elapsed)
    
    async def _ensure_minimum_time_async(self, start_time: float):
        """Async version of ensure minimum time."""
        elapsed = time.time() - start_time
        if elapsed < self.min_operation_time:
            await asyncio.sleep(self.min_operation_time - elapsed)
    
    def validate_credentials(self, provided_username: str, provided_password: str, 
                           stored_username: str, stored_password_hash: str) -> bool:
        """
        Validate user credentials with timing attack protection.
        
        Args:
            provided_username: Username provided by user
            provided_password: Password provided by user
            stored_username: Stored username
            stored_password_hash: Stored password hash
        
        Returns:
            True if credentials are valid, False otherwise
        """
        start_time = time.time()
        
        try:
            # Compare username (timing-safe)
            username_match = constant_time_compare(provided_username, stored_username)
            
            # Always hash the provided password, even if username doesn't match
            provided_hash = hashlib.sha256(provided_password.encode()).hexdigest()
            
            # Compare password hash (timing-safe)
            password_match = constant_time_compare(provided_hash, stored_password_hash)
            
            # Both must match
            result = username_match and password_match
            
        except Exception as e:
            logger.warning(f"Credential validation error: {e}")
            result = False
        
        finally:
            # Ensure constant time regardless of success/failure
            self._ensure_minimum_time(start_time)
        
        return result
    
    async def validate_credentials_async(self, provided_username: str, provided_password: str,
                                       stored_username: str, stored_password_hash: str) -> bool:
        """Async version of validate_credentials."""
        start_time = time.time()
        
        try:
            # Compare username (timing-safe)
            username_match = constant_time_compare(provided_username, stored_username)
            
            # Always hash the provided password, even if username doesn't match
            provided_hash = hashlib.sha256(provided_password.encode()).hexdigest()
            
            # Compare password hash (timing-safe)
            password_match = constant_time_compare(provided_hash, stored_password_hash)
            
            # Both must match
            result = username_match and password_match
            
        except Exception as e:
            logger.warning(f"Async credential validation error: {e}")
            result = False
        
        finally:
            # Ensure constant time regardless of success/failure
            await self._ensure_minimum_time_async(start_time)
        
        return result
    
    def validate_api_key(self, provided_key: str, stored_key: str) -> bool:
        """
        Validate API key with timing attack protection.
        
        Args:
            provided_key: API key provided by client
            stored_key: Stored API key
        
        Returns:
            True if API key is valid, False otherwise
        """
        start_time = time.time()
        
        try:
            result = constant_time_compare(provided_key, stored_key)
        except Exception as e:
            logger.warning(f"API key validation error: {e}")
            result = False
        finally:
            self._ensure_minimum_time(start_time)
        
        return result
    
    async def validate_api_key_async(self, provided_key: str, stored_key: str) -> bool:
        """Async version of validate_api_key."""
        start_time = time.time()
        
        try:
            result = constant_time_compare(provided_key, stored_key)
        except Exception as e:
            logger.warning(f"Async API key validation error: {e}")
            result = False
        finally:
            await self._ensure_minimum_time_async(start_time)
        
        return result
    
    def validate_session_token(self, provided_token: str, stored_token: str) -> bool:
        """
        Validate session token with timing attack protection.
        
        Args:
            provided_token: Token provided by client
            stored_token: Stored session token
        
        Returns:
            True if token is valid, False otherwise
        """
        start_time = time.time()
        
        try:
            result = constant_time_compare(provided_token, stored_token)
        except Exception as e:
            logger.warning(f"Session token validation error: {e}")
            result = False
        finally:
            self._ensure_minimum_time(start_time)
        
        return result


class TimingSafeCache:
    """Cache with timing attack protection for lookups."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}
    
    def _generate_dummy_work(self):
        """Perform dummy computation to normalize timing."""
        # Perform some dummy computation
        dummy = 0
        for i in range(100):
            dummy += hashlib.sha256(str(i).encode()).hexdigest().__hash__()
        return dummy
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with timing protection."""
        start_time = time.time()
        
        # Always perform the same operations regardless of hit/miss
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        if key in self.cache:
            result = self.cache[key]
            self.access_count[key] = self.access_count.get(key, 0) + 1
        else:
            result = default
            # Perform dummy work to match the timing of a cache hit
            self._generate_dummy_work()
        
        # Ensure minimum time
        elapsed = time.time() - start_time
        min_time = 0.001  # 1ms minimum
        if elapsed < min_time:
            time.sleep(min_time - elapsed)
        
        return result
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            if self.access_count:
                lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
                del self.cache[lru_key]
                del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 1


def timing_safe_decorator(min_time: float = 0.01):
    """
    Decorator to ensure functions take a minimum amount of time.
    
    Args:
        min_time: Minimum execution time in seconds
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                if elapsed < min_time:
                    time.sleep(min_time - elapsed)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                if elapsed < min_time:
                    await asyncio.sleep(min_time - elapsed)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure token.
    
    Args:
        length: Token length in bytes
    
    Returns:
        Secure token string
    """
    return secrets.token_urlsafe(length)


def secure_random_delay(min_delay: float = 0.001, max_delay: float = 0.01):
    """
    Add a secure random delay to operations.
    
    Args:
        min_delay: Minimum delay in seconds
        max_delay: Maximum delay in seconds
    """
    delay = secrets.SystemRandom().uniform(min_delay, max_delay)
    time.sleep(delay)


async def secure_random_delay_async(min_delay: float = 0.001, max_delay: float = 0.01):
    """
    Add a secure random delay to async operations.
    
    Args:
        min_delay: Minimum delay in seconds
        max_delay: Maximum delay in seconds
    """
    delay = secrets.SystemRandom().uniform(min_delay, max_delay)
    await asyncio.sleep(delay)


class RateLimitedTimingSafeValidator(TimingSafeValidator):
    """Timing-safe validator with built-in rate limiting."""
    
    def __init__(self, min_operation_time: float = 0.01, max_attempts_per_minute: int = 5):
        super().__init__(min_operation_time)
        self.max_attempts = max_attempts_per_minute
        self.attempts: Dict[str, list] = {}
    
    def _check_rate_limit(self, identifier: str) -> bool:
        """Check if identifier is within rate limits."""
        current_time = time.time()
        
        if identifier not in self.attempts:
            self.attempts[identifier] = []
        
        # Remove attempts older than 1 minute
        self.attempts[identifier] = [
            attempt_time for attempt_time in self.attempts[identifier]
            if current_time - attempt_time < 60
        ]
        
        # Check if within limit
        if len(self.attempts[identifier]) >= self.max_attempts:
            return False
        
        # Record this attempt
        self.attempts[identifier].append(current_time)
        return True
    
    def validate_credentials_with_rate_limit(self, identifier: str, provided_username: str, 
                                           provided_password: str, stored_username: str, 
                                           stored_password_hash: str) -> tuple[bool, bool]:
        """
        Validate credentials with rate limiting.
        
        Args:
            identifier: Unique identifier for rate limiting (e.g., IP address)
            provided_username: Username provided by user
            provided_password: Password provided by user
            stored_username: Stored username
            stored_password_hash: Stored password hash
        
        Returns:
            Tuple of (is_valid, rate_limit_exceeded)
        """
        start_time = time.time()
        
        try:
            # Check rate limit first
            if not self._check_rate_limit(identifier):
                # Still perform dummy validation to maintain timing
                self._dummy_validation()
                return False, True
            
            # Perform actual validation
            result = self.validate_credentials(
                provided_username, provided_password, stored_username, stored_password_hash
            )
            
            return result, False
            
        finally:
            self._ensure_minimum_time(start_time)
    
    def _dummy_validation(self):
        """Perform dummy validation work to maintain consistent timing."""
        # Simulate the work done in actual validation
        dummy_username = "dummy"
        dummy_password = "dummy"
        dummy_hash = hashlib.sha256(dummy_password.encode()).hexdigest()
        
        constant_time_compare(dummy_username, dummy_username)
        constant_time_compare(dummy_hash, dummy_hash)


# Global validator instance
_timing_safe_validator = TimingSafeValidator()
_rate_limited_validator = RateLimitedTimingSafeValidator()

# Export commonly used functions
validate_credentials = _timing_safe_validator.validate_credentials
validate_credentials_async = _timing_safe_validator.validate_credentials_async
validate_api_key = _timing_safe_validator.validate_api_key
validate_api_key_async = _timing_safe_validator.validate_api_key_async
validate_session_token = _timing_safe_validator.validate_session_token

# Rate-limited versions
validate_credentials_with_rate_limit = _rate_limited_validator.validate_credentials_with_rate_limit


# Utility functions for common use cases
def safe_string_equals(str1: str, str2: str) -> bool:
    """Safely compare two strings for equality."""
    return constant_time_compare(str1, str2)


def safe_hash_verification(data: str, expected_hash: str, algorithm: str = 'sha256') -> bool:
    """
    Safely verify data against expected hash.
    
    Args:
        data: Data to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm to use
    
    Returns:
        True if hash matches, False otherwise
    """
    if algorithm.lower() == 'sha256':
        computed_hash = hashlib.sha256(data.encode()).hexdigest()
    elif algorithm.lower() == 'sha1':
        computed_hash = hashlib.sha1(data.encode()).hexdigest()
    elif algorithm.lower() == 'md5':
        computed_hash = hashlib.md5(data.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    return constant_time_compare(computed_hash, expected_hash)


def create_timing_safe_lookup_table(data: Dict[str, Any]) -> TimingSafeCache:
    """
    Create a timing-safe lookup table from a dictionary.
    
    Args:
        data: Dictionary to convert to timing-safe cache
    
    Returns:
        TimingSafeCache instance
    """
    cache = TimingSafeCache(max_size=len(data) * 2)  # Allow some growth
    for key, value in data.items():
        cache.set(key, value)
    return cache