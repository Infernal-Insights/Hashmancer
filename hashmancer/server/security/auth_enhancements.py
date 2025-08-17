"""Enhanced authentication with 2FA, session management, and password policies."""

import time
import secrets
import hashlib
import hmac
import base64
import qrcode
import io
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import re
import json
from datetime import datetime, timedelta

try:
    import pyotp
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False
    logging.warning("pyotp not available - TOTP 2FA will be disabled")

logger = logging.getLogger(__name__)


class TwoFactorMethod(Enum):
    """Two-factor authentication methods."""
    TOTP = "totp"  # Time-based One-Time Password (Google Authenticator, etc.)
    SMS = "sms"    # SMS verification
    EMAIL = "email"  # Email verification
    BACKUP_CODES = "backup_codes"  # Backup recovery codes


class SessionStatus(Enum):
    """Session status types."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPICIOUS = "suspicious"


@dataclass
class TwoFactorConfig:
    """Two-factor authentication configuration."""
    enabled: bool
    method: TwoFactorMethod
    secret: Optional[str] = None
    backup_codes: Optional[List[str]] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    created_at: float = 0.0
    last_used: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class SessionInfo:
    """User session information."""
    session_id: str
    user_id: str
    ip_address: str
    user_agent: str
    created_at: float
    last_activity: float
    expires_at: float
    status: SessionStatus
    two_factor_verified: bool = False
    login_method: str = "password"
    device_fingerprint: Optional[str] = None
    location: Optional[Dict[str, str]] = None
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at or self.status != SessionStatus.ACTIVE
    
    def is_inactive(self, timeout: int = 1800) -> bool:  # 30 minutes default
        return time.time() - self.last_activity > timeout


@dataclass
class PasswordPolicy:
    """Password policy configuration."""
    min_length: int = 8
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special: bool = True
    min_unique_chars: int = 4
    prevent_reuse: int = 5  # Number of previous passwords to check
    max_age_days: int = 90  # Password expiration
    lockout_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    strength_check: bool = True
    
    def validate_password(self, password: str, username: str = "") -> Tuple[bool, List[str]]:
        """Validate password against policy."""
        errors = []
        
        # Length checks
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
        if len(password) > self.max_length:
            errors.append(f"Password must be at most {self.max_length} characters long")
        
        # Character requirements
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        if self.require_numbers and not re.search(r'[0-9]', password):
            errors.append("Password must contain at least one number")
        if self.require_special and not re.search(r'[^a-zA-Z0-9]', password):
            errors.append("Password must contain at least one special character")
        
        # Unique characters
        if len(set(password)) < self.min_unique_chars:
            errors.append(f"Password must contain at least {self.min_unique_chars} unique characters")
        
        # Username similarity
        if username and username.lower() in password.lower():
            errors.append("Password cannot contain username")
        
        # Common weak passwords
        weak_patterns = [
            r'123+', r'abc+', r'qwerty', r'password', r'admin', r'login',
            r'(.)\1{3,}',  # Repeated characters
        ]
        for pattern in weak_patterns:
            if re.search(pattern, password, re.I):
                errors.append("Password contains common weak patterns")
                break
        
        return len(errors) == 0, errors
    
    def calculate_strength(self, password: str) -> int:
        """Calculate password strength score (0-100)."""
        score = 0
        
        # Length bonus
        score += min(len(password) * 2, 25)
        
        # Character variety
        if re.search(r'[a-z]', password):
            score += 10
        if re.search(r'[A-Z]', password):
            score += 10
        if re.search(r'[0-9]', password):
            score += 10
        if re.search(r'[^a-zA-Z0-9]', password):
            score += 15
        
        # Unique characters
        unique_ratio = len(set(password)) / len(password) if password else 0
        score += int(unique_ratio * 20)
        
        # Penalize common patterns
        if re.search(r'(.)\1{2,}', password):  # Repeated chars
            score -= 10
        if re.search(r'(012|123|234|345|456|567|678|789|890)', password):  # Sequential numbers
            score -= 10
        if re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', password, re.I):  # Sequential letters
            score -= 10
        
        return max(0, min(score, 100))


class TwoFactorAuth:
    """Two-factor authentication manager."""
    
    def __init__(self, app_name: str = "Hashmancer"):
        self.app_name = app_name
        self._user_configs: Dict[str, TwoFactorConfig] = {}
        self._pending_verifications: Dict[str, Dict[str, Any]] = {}
        self._verification_attempts: Dict[str, List[float]] = {}
        
        if not TOTP_AVAILABLE:
            logger.warning("TOTP 2FA disabled - pyotp not available")
    
    def setup_totp(self, user_id: str, user_email: str) -> Dict[str, str]:
        """Set up TOTP 2FA for a user."""
        if not TOTP_AVAILABLE:
            raise RuntimeError("TOTP not available - install pyotp")
        
        # Generate secret
        secret = pyotp.random_base32()
        
        # Create TOTP instance
        totp = pyotp.TOTP(secret)
        
        # Generate provisioning URI for QR code
        provisioning_uri = totp.provisioning_uri(
            user_email,
            issuer_name=self.app_name
        )
        
        # Generate backup codes
        backup_codes = [secrets.token_hex(4) for _ in range(10)]
        
        # Store configuration (not activated until verified)
        temp_config = TwoFactorConfig(
            enabled=False,  # Will be enabled after verification
            method=TwoFactorMethod.TOTP,
            secret=secret,
            backup_codes=backup_codes,
            email=user_email
        )
        
        # Store temporarily until verified
        verification_token = secrets.token_urlsafe(32)
        self._pending_verifications[verification_token] = {
            'user_id': user_id,
            'config': temp_config,
            'expires': time.time() + 600,  # 10 minutes to verify
        }
        
        return {
            'secret': secret,
            'qr_code_uri': provisioning_uri,
            'backup_codes': backup_codes,
            'verification_token': verification_token
        }
    
    def verify_totp_setup(self, verification_token: str, totp_code: str) -> bool:
        """Verify TOTP setup with code."""
        if not TOTP_AVAILABLE:
            return False
        
        pending = self._pending_verifications.get(verification_token)
        if not pending or time.time() > pending['expires']:
            return False
        
        config = pending['config']
        totp = pyotp.TOTP(config.secret)
        
        if totp.verify(totp_code, valid_window=1):  # Allow 1 window tolerance
            # Activate 2FA
            config.enabled = True
            config.last_used = time.time()
            self._user_configs[pending['user_id']] = config
            
            # Clean up pending verification
            del self._pending_verifications[verification_token]
            
            logger.info(f"TOTP 2FA enabled for user {pending['user_id']}")
            return True
        
        return False
    
    def generate_backup_codes(self, user_id: str) -> List[str]:
        """Generate new backup codes for user."""
        config = self._user_configs.get(user_id)
        if not config:
            raise ValueError("2FA not configured for user")
        
        backup_codes = [secrets.token_hex(4) for _ in range(10)]
        config.backup_codes = backup_codes
        
        logger.info(f"Generated new backup codes for user {user_id}")
        return backup_codes
    
    def verify_code(self, user_id: str, code: str, method: TwoFactorMethod = TwoFactorMethod.TOTP) -> bool:
        """Verify 2FA code."""
        config = self._user_configs.get(user_id)
        if not config or not config.enabled:
            return False
        
        # Rate limiting
        now = time.time()
        attempts = self._verification_attempts.setdefault(user_id, [])
        attempts = [t for t in attempts if now - t < 300]  # Keep attempts from last 5 minutes
        
        if len(attempts) >= 5:  # Max 5 attempts per 5 minutes
            logger.warning(f"2FA rate limit exceeded for user {user_id}")
            return False
        
        attempts.append(now)
        self._verification_attempts[user_id] = attempts
        
        verified = False
        
        if method == TwoFactorMethod.TOTP and TOTP_AVAILABLE:
            totp = pyotp.TOTP(config.secret)
            verified = totp.verify(code, valid_window=1)
        
        elif method == TwoFactorMethod.BACKUP_CODES:
            if config.backup_codes and code in config.backup_codes:
                # Remove used backup code
                config.backup_codes.remove(code)
                verified = True
                logger.info(f"Backup code used for user {user_id}")
        
        if verified:
            config.last_used = time.time()
            # Clear rate limiting on successful verification
            self._verification_attempts.pop(user_id, None)
        
        return verified
    
    def disable_2fa(self, user_id: str) -> bool:
        """Disable 2FA for a user."""
        if user_id in self._user_configs:
            del self._user_configs[user_id]
            logger.info(f"2FA disabled for user {user_id}")
            return True
        return False
    
    def is_enabled(self, user_id: str) -> bool:
        """Check if 2FA is enabled for user."""
        config = self._user_configs.get(user_id)
        return config is not None and config.enabled
    
    def get_config(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get 2FA configuration for user (without secrets)."""
        config = self._user_configs.get(user_id)
        if not config:
            return None
        
        return {
            'enabled': config.enabled,
            'method': config.method.value,
            'has_backup_codes': bool(config.backup_codes),
            'backup_codes_remaining': len(config.backup_codes) if config.backup_codes else 0,
            'created_at': config.created_at,
            'last_used': config.last_used
        }
    
    def generate_qr_code(self, provisioning_uri: str) -> bytes:
        """Generate QR code image for TOTP setup."""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    def cleanup_expired(self):
        """Clean up expired pending verifications."""
        now = time.time()
        expired = [token for token, data in self._pending_verifications.items() 
                  if now > data['expires']]
        
        for token in expired:
            del self._pending_verifications[token]


class SessionManager:
    """Advanced session management with security features."""
    
    def __init__(self, default_timeout: int = 3600, max_sessions_per_user: int = 5):
        self.default_timeout = default_timeout
        self.max_sessions_per_user = max_sessions_per_user
        self._sessions: Dict[str, SessionInfo] = {}
        self._user_sessions: Dict[str, List[str]] = {}
        self._login_attempts: Dict[str, List[float]] = {}
    
    def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        timeout: Optional[int] = None,
        two_factor_verified: bool = False,
        login_method: str = "password"
    ) -> str:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(32)
        now = time.time()
        
        session = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=now,
            last_activity=now,
            expires_at=now + (timeout or self.default_timeout),
            status=SessionStatus.ACTIVE,
            two_factor_verified=two_factor_verified,
            login_method=login_method,
            device_fingerprint=self._generate_device_fingerprint(ip_address, user_agent)
        )
        
        self._sessions[session_id] = session
        
        # Track user sessions
        user_sessions = self._user_sessions.setdefault(user_id, [])
        user_sessions.append(session_id)
        
        # Enforce max sessions per user
        if len(user_sessions) > self.max_sessions_per_user:
            # Remove oldest sessions
            oldest_sessions = sorted(
                user_sessions[:-self.max_sessions_per_user],
                key=lambda sid: self._sessions[sid].created_at if sid in self._sessions else 0
            )
            for old_session_id in oldest_sessions:
                self.revoke_session(old_session_id, "Max sessions exceeded")
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        # Check if session is expired
        if session.is_expired():
            self.revoke_session(session_id, "Expired")
            return None
        
        return session
    
    def update_activity(self, session_id: str) -> bool:
        """Update session last activity."""
        session = self._sessions.get(session_id)
        if not session or session.is_expired():
            return False
        
        session.last_activity = time.time()
        return True
    
    def revoke_session(self, session_id: str, reason: str = "Manual revocation") -> bool:
        """Revoke a session."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        session.status = SessionStatus.REVOKED
        
        # Remove from user sessions
        user_sessions = self._user_sessions.get(session.user_id, [])
        if session_id in user_sessions:
            user_sessions.remove(session_id)
        
        logger.info(f"Revoked session {session_id} for user {session.user_id}: {reason}")
        return True
    
    def revoke_all_user_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """Revoke all sessions for a user."""
        user_sessions = self._user_sessions.get(user_id, []).copy()
        revoked_count = 0
        
        for session_id in user_sessions:
            if session_id != except_session:
                if self.revoke_session(session_id, "All sessions revoked"):
                    revoked_count += 1
        
        return revoked_count
    
    def get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get all active sessions for a user."""
        session_ids = self._user_sessions.get(user_id, [])
        sessions = []
        
        for session_id in session_ids.copy():  # Copy to avoid modification during iteration
            session = self.get_session(session_id)  # This handles expiry
            if session:
                sessions.append(session)
            else:
                # Clean up dead session reference
                session_ids.remove(session_id)
        
        return sessions
    
    def detect_suspicious_activity(self, session_id: str) -> bool:
        """Detect suspicious session activity."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        suspicious = False
        
        # Check for IP address changes (could indicate session hijacking)
        # This would require storing historical IP data
        
        # Check for unusual user agent changes
        # This would require storing historical user agent data
        
        # Check for concurrent sessions from different locations
        user_sessions = self.get_user_sessions(session.user_id)
        unique_ips = set(s.ip_address for s in user_sessions)
        if len(unique_ips) > 3:  # More than 3 different IPs
            suspicious = True
        
        if suspicious:
            session.status = SessionStatus.SUSPICIOUS
            logger.warning(f"Suspicious activity detected for session {session_id}")
        
        return suspicious
    
    def _generate_device_fingerprint(self, ip_address: str, user_agent: str) -> str:
        """Generate device fingerprint."""
        data = f"{ip_address}:{user_agent}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = time.time()
        expired_sessions = []
        
        for session_id, session in self._sessions.items():
            if session.is_expired() or session.is_inactive():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.revoke_session(session_id, "Cleanup - expired")
        
        return len(expired_sessions)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        active_sessions = [s for s in self._sessions.values() if s.status == SessionStatus.ACTIVE]
        
        return {
            'total_sessions': len(self._sessions),
            'active_sessions': len(active_sessions),
            'users_with_sessions': len(self._user_sessions),
            'avg_sessions_per_user': len(active_sessions) / max(len(self._user_sessions), 1),
            'suspicious_sessions': len([s for s in active_sessions if s.status == SessionStatus.SUSPICIOUS]),
            '2fa_verified_sessions': len([s for s in active_sessions if s.two_factor_verified])
        }


# Global instances
_two_factor_auth: Optional[TwoFactorAuth] = None
_session_manager: Optional[SessionManager] = None
_password_policy: Optional[PasswordPolicy] = None


def get_two_factor_auth() -> TwoFactorAuth:
    """Get global 2FA instance."""
    global _two_factor_auth
    if _two_factor_auth is None:
        _two_factor_auth = TwoFactorAuth()
    return _two_factor_auth


def get_session_manager() -> SessionManager:
    """Get global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_password_policy() -> PasswordPolicy:
    """Get global password policy instance."""
    global _password_policy
    if _password_policy is None:
        _password_policy = PasswordPolicy()
    return _password_policy