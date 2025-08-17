
# TODO: Security Review Required
# Session token generation needs comprehensive security audit
# Consider implementing:
# 1. secrets.token_urlsafe(32) for session tokens
# 2. Proper CSRF protection
# 3. Secure session storage with expiration
# 4. Rate limiting for authentication attempts
"""Authentication middleware and utilities."""

import secrets
import time
import hmac
import hashlib
from typing import Optional
from hashmancer.server.redis_utils import get_redis


def sign_session(session_id: str, expiry: int, passkey: str) -> str:
    """Return a signed session token using the provided passkey."""
    key = (passkey or "").encode()
    payload = f"{session_id}|{expiry}".encode()
    sig = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return f"{session_id}|{expiry}|{sig}"


def verify_session_token(token: str, passkey: str, session_ttl: int = 3600) -> bool:
    """Validate a session token and confirm it's stored in Redis."""
    if not token or not passkey:
        return False
        
    try:
        session_id, exp_s, sig = token.split("|")
        expiry = int(exp_s)
    except ValueError:
        return False
        
    if expiry < int(time.time()):
        return False
        
    expected = hmac.new(
        passkey.encode(), f"{session_id}|{expiry}".encode(), hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(expected, sig):
        return False
        
    try:
        r = get_redis()
        if not r.exists(f"session:{session_id}"):
            return False
            
        ttl = r.ttl(f"session:{session_id}")
        if ttl <= 0:
            return False
            
        return True
    except Exception:
        return False