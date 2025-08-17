"""Advanced rate limiting with DDoS protection and IP-based rules."""

import time
import hashlib
import logging
import asyncio
import json
import secrets
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from threading import Lock
from functools import wraps
from fastapi import Request, HTTPException
import ipaddress
import re

from ..redis_utils import get_redis

logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    requests: int  # Number of requests
    window: int    # Time window in seconds
    burst: int     # Burst allowance
    description: str = ""


@dataclass
class RateLimitStatus:
    """Current rate limit status for a client."""
    requests_made: int
    requests_remaining: int
    window_reset: float
    blocked: bool
    burst_used: int
    last_request: float


@dataclass
class SuspiciousActivity:
    """Suspicious activity detection record."""
    ip: str
    activity_type: str
    timestamp: float
    details: Dict[str, Any]
    severity: str  # low, medium, high, critical


class RateLimiter:
    """Advanced rate limiter with DDoS protection and adaptive rules."""
    
    def __init__(self):
        self._clients: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self._blocked_ips: Dict[str, float] = {}
        self._suspicious_activity: List[SuspiciousActivity] = []
        self._lock = Lock()
        self.redis = get_redis()
        
        # Default rate limiting rules
        self._rules = {
            'global': RateLimitRule(100, 60, 20, "Global rate limit"),
            'login': RateLimitRule(5, 300, 2, "Login attempts"),
            'register': RateLimitRule(3, 3600, 1, "Worker registration"),
            'api': RateLimitRule(1000, 60, 50, "API requests"),
            'upload': RateLimitRule(10, 60, 5, "File uploads"),
            'download': RateLimitRule(50, 60, 10, "Downloads"),
        }
        
        # Suspicious patterns
        self._suspicious_patterns = [
            (re.compile(r'(?i)(union|select|insert|delete|drop|exec|script)', re.I), 'sql_injection'),
            (re.compile(r'(?i)(<script|javascript:|vbscript:|onload=)', re.I), 'xss_attempt'),
            (re.compile(r'(?i)(\.\.\/|\.\.\\|\/etc\/|\/proc\/)', re.I), 'path_traversal'),
            (re.compile(r'(?i)(cmd|powershell|bash|sh)[\s=]', re.I), 'command_injection'),
        ]
        
        # Trusted IP ranges (configurable)
        self._trusted_ranges = [
            ipaddress.ip_network('127.0.0.0/8'),  # Localhost
            ipaddress.ip_network('10.0.0.0/8'),   # Private
            ipaddress.ip_network('172.16.0.0/12'), # Private
            ipaddress.ip_network('192.168.0.0/16') # Private
        ]
        
        # Geoblocking (example countries to block)
        self._blocked_countries = set()  # Can be populated with country codes
        
    def _get_client_key(self, request: Request) -> str:
        """Generate unique client key from request."""
        # Try to get real IP from headers (proxy-aware)
        client_ip = (
            request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or
            request.headers.get('X-Real-IP', '').strip() or
            request.headers.get('CF-Connecting-IP', '').strip() or
            getattr(request.client, 'host', 'unknown')
        )
        
        # Include user agent for more granular tracking
        user_agent = request.headers.get('User-Agent', 'unknown')
        ua_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
        
        return f"{client_ip}:{ua_hash}"
    
    def _is_trusted_ip(self, ip_str: str) -> bool:
        """Check if IP is in trusted ranges."""
        try:
            ip = ipaddress.ip_address(ip_str)
            return any(ip in network for network in self._trusted_ranges)
        except ValueError:
            return False
    
    def _detect_suspicious_activity(self, request: Request, client_key: str):
        """Detect suspicious activity patterns."""
        current_time = time.time()
        client_ip = client_key.split(':')[0]
        
        # Check request patterns
        url = str(request.url)
        headers = dict(request.headers)
        
        suspicious_items = []
        
        # Check for malicious patterns in URL and headers
        for pattern, activity_type in self._suspicious_patterns:
            if pattern.search(url) or any(pattern.search(str(v)) for v in headers.values()):
                suspicious_items.append(activity_type)
        
        # Check for rapid-fire requests (potential bot)
        with self._lock:
            client_history = self._clients[client_key]['global']
            if len(client_history) >= 10:
                recent_requests = [req for req in client_history if current_time - req <= 10]
                if len(recent_requests) >= 10:
                    suspicious_items.append('rapid_fire_requests')
        
        # Check for unusual user agents
        user_agent = headers.get('user-agent', '').lower()
        if any(bot in user_agent for bot in ['bot', 'crawler', 'spider', 'scraper']) and not any(
            legit in user_agent for legit in ['googlebot', 'bingbot', 'slackbot']
        ):
            suspicious_items.append('suspicious_user_agent')
        
        # Log suspicious activity
        for activity_type in suspicious_items:
            severity = 'high' if activity_type in ['sql_injection', 'command_injection'] else 'medium'
            
            activity = SuspiciousActivity(
                ip=client_ip,
                activity_type=activity_type,
                timestamp=current_time,
                details={
                    'url': url,
                    'user_agent': headers.get('user-agent', ''),
                    'method': request.method
                },
                severity=severity
            )
            
            self._suspicious_activity.append(activity)
            
            # Keep only recent activity (last 24 hours)
            cutoff = current_time - 86400
            self._suspicious_activity = [
                a for a in self._suspicious_activity if a.timestamp > cutoff
            ]
            
            logger.warning(f"Suspicious activity detected: {activity_type} from {client_ip}")
            
            # Auto-block for critical activities
            if severity == 'high':
                self._blocked_ips[client_ip] = current_time + 3600  # Block for 1 hour
                logger.error(f"Auto-blocked IP {client_ip} for {activity_type}")
    
    async def allow_request(
        self, 
        key: str, 
        max_requests: int, 
        window_seconds: int,
        burst_allowance: int = 0
    ) -> bool:
        """Redis-based distributed rate limiting."""
        try:
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Use Redis sorted set for sliding window
            rate_key = f"rate_limit:{key}"
            
            # Remove old entries
            self.redis.zremrangebyscore(rate_key, 0, window_start)
            
            # Count current requests
            current_requests = self.redis.zcard(rate_key)
            
            # Check if within limit
            if current_requests >= max_requests + burst_allowance:
                # Set expiration for cleanup
                self.redis.expire(rate_key, window_seconds)
                return False
            
            # Add current request
            request_id = f"{current_time}:{secrets.token_hex(8)}"
            self.redis.zadd(rate_key, {request_id: current_time})
            
            # Set expiration
            self.redis.expire(rate_key, window_seconds)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis rate limiting failed for {key}: {e}")
            # Fallback to allow request on Redis failure
            return True
    
    async def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked in Redis."""
        try:
            block_key = f"blocked_ip:{ip}"
            blocked_until = self.redis.get(block_key)
            if blocked_until:
                if int(blocked_until) > time.time():
                    return True
                else:
                    self.redis.delete(block_key)
            return False
        except Exception as e:
            logger.error(f"Failed to check IP block status: {e}")
            return False
    
    async def block_ip_redis(self, ip: str, duration: int, reason: str):
        """Block IP in Redis for distributed blocking."""
        try:
            block_key = f"blocked_ip:{ip}"
            blocked_until = int(time.time()) + duration
            self.redis.setex(block_key, duration, blocked_until)
            
            # Log the block
            block_log_key = f"block_log:{ip}"
            block_info = {
                "timestamp": int(time.time()),
                "duration": duration,
                "reason": reason,
                "blocked_until": blocked_until
            }
            self.redis.lpush(block_log_key, json.dumps(block_info))
            self.redis.ltrim(block_log_key, 0, 99)  # Keep last 100 blocks
            self.redis.expire(block_log_key, 86400 * 30)  # Expire after 30 days
            
            logger.warning(f"Blocked IP {ip} for {duration} seconds: {reason}")
        except Exception as e:
            logger.error(f"Failed to block IP {ip}: {e}")
    
    def check_rate_limit(
        self, 
        request: Request, 
        rule_name: str = 'global',
        custom_key: Optional[str] = None
    ) -> RateLimitStatus:
        """Check if request is within rate limits."""
        current_time = time.time()
        client_key = custom_key or self._get_client_key(request)
        client_ip = client_key.split(':')[0]
        
        # Check if IP is blocked
        if client_ip in self._blocked_ips:
            if current_time > self._blocked_ips[client_ip]:
                del self._blocked_ips[client_ip]
            else:
                return RateLimitStatus(
                    requests_made=0,
                    requests_remaining=0,
                    window_reset=self._blocked_ips[client_ip],
                    blocked=True,
                    burst_used=0,
                    last_request=current_time
                )
        
        # Skip rate limiting for trusted IPs (configurable)
        if self._is_trusted_ip(client_ip):
            return RateLimitStatus(
                requests_made=0,
                requests_remaining=999999,
                window_reset=current_time + 60,
                blocked=False,
                burst_used=0,
                last_request=current_time
            )
        
        # Get rate limiting rule
        rule = self._rules.get(rule_name, self._rules['global'])
        
        with self._lock:
            # Clean old entries
            client_history = self._clients[client_key][rule_name]
            cutoff = current_time - rule.window
            while client_history and client_history[0] <= cutoff:
                client_history.popleft()
            
            # Check current usage
            requests_made = len(client_history)
            requests_remaining = max(0, rule.requests - requests_made)
            
            # Calculate burst usage
            recent_cutoff = current_time - min(60, rule.window // 4)  # Last quarter of window or 1 min
            recent_requests = sum(1 for req in client_history if req > recent_cutoff)
            burst_used = min(recent_requests, rule.burst)
            
            # Check if over limit
            blocked = requests_made >= rule.requests
            
            # Apply burst protection
            if recent_requests > rule.burst:
                blocked = True
                logger.warning(f"Burst limit exceeded for {client_ip} on {rule_name}")
            
            # Record this request if not blocked
            if not blocked:
                client_history.append(current_time)
                self._detect_suspicious_activity(request, client_key)
            
            window_reset = cutoff + rule.window + rule.window
            
            return RateLimitStatus(
                requests_made=requests_made + (1 if not blocked else 0),
                requests_remaining=max(0, requests_remaining - (1 if not blocked else 0)),
                window_reset=window_reset,
                blocked=blocked,
                burst_used=burst_used + (1 if not blocked and recent_requests < rule.burst else 0),
                last_request=current_time
            )
    
    def add_rule(self, name: str, rule: RateLimitRule):
        """Add or update a rate limiting rule."""
        self._rules[name] = rule
        logger.info(f"Rate limit rule added: {name} - {rule.description}")
    
    def block_ip(self, ip: str, duration: int = 3600, reason: str = "Manual block"):
        """Manually block an IP address."""
        block_until = time.time() + duration
        with self._lock:
            self._blocked_ips[ip] = block_until
        logger.info(f"IP {ip} blocked until {block_until} - Reason: {reason}")
    
    def unblock_ip(self, ip: str):
        """Manually unblock an IP address."""
        with self._lock:
            if ip in self._blocked_ips:
                del self._blocked_ips[ip]
                logger.info(f"IP {ip} unblocked")
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting statistics."""
        current_time = time.time()
        
        with self._lock:
            # Active clients per rule
            active_clients = {}
            for rule_name in self._rules:
                count = 0
                for client_key in self._clients:
                    client_history = self._clients[client_key][rule_name]
                    if client_history and current_time - client_history[-1] <= 300:  # Active in last 5 min
                        count += 1
                active_clients[rule_name] = count
            
            # Blocked IPs
            blocked_ips = []
            for ip, unblock_time in self._blocked_ips.items():
                if current_time < unblock_time:
                    blocked_ips.append({
                        'ip': ip,
                        'unblock_time': unblock_time,
                        'remaining_seconds': int(unblock_time - current_time)
                    })
            
            # Recent suspicious activity
            recent_suspicious = [
                asdict(activity) for activity in self._suspicious_activity[-50:]
            ]
            
            # Top IPs by request volume
            ip_requests = defaultdict(int)
            for client_key in self._clients:
                ip = client_key.split(':')[0]
                for rule_name in self._clients[client_key]:
                    ip_requests[ip] += len(self._clients[client_key][rule_name])
            
            top_ips = sorted(ip_requests.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'rules': {name: asdict(rule) for name, rule in self._rules.items()},
                'active_clients': active_clients,
                'blocked_ips': blocked_ips,
                'recent_suspicious_activity': recent_suspicious,
                'suspicious_activity_count': len(self._suspicious_activity),
                'top_ips_by_requests': top_ips,
                'total_tracked_clients': len(self._clients)
            }
    
    def cleanup_old_data(self, max_age: int = 86400):
        """Clean up old tracking data."""
        current_time = time.time()
        cutoff = current_time - max_age
        
        with self._lock:
            # Clean client histories
            clients_to_remove = []
            for client_key in self._clients:
                rules_to_remove = []
                for rule_name in self._clients[client_key]:
                    history = self._clients[client_key][rule_name]
                    # Remove old entries
                    while history and history[0] <= cutoff:
                        history.popleft()
                    
                    if not history:
                        rules_to_remove.append(rule_name)
                
                # Remove empty rule histories
                for rule_name in rules_to_remove:
                    del self._clients[client_key][rule_name]
                
                # Mark client for removal if no rules left
                if not self._clients[client_key]:
                    clients_to_remove.append(client_key)
            
            # Remove empty clients
            for client_key in clients_to_remove:
                del self._clients[client_key]
            
            # Clean expired IP blocks
            expired_blocks = [ip for ip, unblock_time in self._blocked_ips.items() 
                            if current_time > unblock_time]
            for ip in expired_blocks:
                del self._blocked_ips[ip]
            
            # Clean old suspicious activity
            self._suspicious_activity = [
                activity for activity in self._suspicious_activity 
                if activity.timestamp > cutoff
            ]
        
        logger.debug(f"Cleaned up rate limiter data older than {max_age} seconds")


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def rate_limit(rule_name: str = 'global', custom_key: Optional[str] = None):
    """Decorator for rate limiting endpoints."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request object in arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # If no request found, skip rate limiting (shouldn't happen in normal usage)
                return await func(*args, **kwargs)
            
            limiter = get_rate_limiter()
            status = limiter.check_rate_limit(request, rule_name, custom_key)
            
            if status.blocked:
                logger.warning(f"Rate limit exceeded: {request.client.host} on {rule_name}")
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "rule": rule_name,
                        "retry_after": int(status.window_reset - time.time()),
                        "requests_remaining": status.requests_remaining
                    },
                    headers={
                        "Retry-After": str(int(status.window_reset - time.time())),
                        "X-RateLimit-Limit": str(limiter._rules[rule_name].requests),
                        "X-RateLimit-Remaining": str(status.requests_remaining),
                        "X-RateLimit-Reset": str(int(status.window_reset))
                    }
                )
            
            # Add rate limit headers to response
            response = await func(*args, **kwargs)
            
            if hasattr(response, 'headers'):
                response.headers["X-RateLimit-Limit"] = str(limiter._rules[rule_name].requests)
                response.headers["X-RateLimit-Remaining"] = str(status.requests_remaining)
                response.headers["X-RateLimit-Reset"] = str(int(status.window_reset))
            
            return response
        
        return wrapper
    return decorator


def get_rate_limiter_stats() -> Dict[str, Any]:
    """Get rate limiter statistics."""
    return get_rate_limiter().get_stats()


def block_ip_address(ip: str, duration: int = 3600, reason: str = "Manual block") -> bool:
    """Block an IP address manually."""
    try:
        get_rate_limiter().block_ip(ip, duration, reason)
        return True
    except Exception as e:
        logger.error(f"Failed to block IP {ip}: {e}")
        return False


def unblock_ip_address(ip: str) -> bool:
    """Unblock an IP address manually."""
    try:
        return get_rate_limiter().unblock_ip(ip)
    except Exception as e:
        logger.error(f"Failed to unblock IP {ip}: {e}")
        return False