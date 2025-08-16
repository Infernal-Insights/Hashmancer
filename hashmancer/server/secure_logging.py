"""
Secure Logging System
Provides secure logging with sensitive data protection, structured logging, and audit trails
"""

import logging
import json
import re
import hashlib
import time
import threading
from typing import Dict, Any, Optional, List, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import sys
from pathlib import Path
import gzip
import shutil
from datetime import datetime, timedelta
import asyncio

# Custom log levels
AUDIT_LEVEL = 25  # Between INFO (20) and WARNING (30)
SECURITY_LEVEL = 35  # Between WARNING (30) and ERROR (40)

logging.addLevelName(AUDIT_LEVEL, "AUDIT")
logging.addLevelName(SECURITY_LEVEL, "SECURITY")


class LogLevel(Enum):
    """Custom log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    AUDIT = AUDIT_LEVEL
    WARNING = logging.WARNING
    SECURITY = SECURITY_LEVEL
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class SensitiveDataType(Enum):
    """Types of sensitive data to protect."""
    PASSWORD = "password"
    TOKEN = "token"
    SESSION_ID = "session_id"
    API_KEY = "api_key"
    HASH = "hash"
    EMAIL = "email"
    IP_ADDRESS = "ip_address"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    PRIVATE_KEY = "private_key"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: float
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None
    exception_info: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class SensitiveDataFilter:
    """Filter to remove or mask sensitive data from logs."""
    
    def __init__(self):
        # Regex patterns for sensitive data
        self.patterns = {
            SensitiveDataType.PASSWORD: [
                r'password["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)',
                r'passwd["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)',
                r'pwd["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)',
            ],
            SensitiveDataType.TOKEN: [
                r'token["\']?\s*[:=]\s*["\']?([A-Za-z0-9+/=]{20,})',
                r'bearer\s+([A-Za-z0-9+/=]{20,})',
                r'jwt["\']?\s*[:=]\s*["\']?([A-Za-z0-9._-]{20,})',
            ],
            SensitiveDataType.API_KEY: [
                r'api_key["\']?\s*[:=]\s*["\']?([A-Za-z0-9]{20,})',
                r'apikey["\']?\s*[:=]\s*["\']?([A-Za-z0-9]{20,})',
                r'key["\']?\s*[:=]\s*["\']?([A-Za-z0-9]{32,})',
            ],
            SensitiveDataType.SESSION_ID: [
                r'session_id["\']?\s*[:=]\s*["\']?([A-Za-z0-9+/=_-]{20,})',
                r'sessionid["\']?\s*[:=]\s*["\']?([A-Za-z0-9+/=_-]{20,})',
            ],
            SensitiveDataType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            SensitiveDataType.IP_ADDRESS: [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
            ],
            SensitiveDataType.HASH: [
                r'hash["\']?\s*[:=]\s*["\']?([a-fA-F0-9]{32,})',
                r'\b[a-fA-F0-9]{64}\b',  # SHA-256
                r'\b[a-fA-F0-9]{40}\b',  # SHA-1
                r'\b[a-fA-F0-9]{32}\b',  # MD5
            ],
            SensitiveDataType.CREDIT_CARD: [
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            ],
            SensitiveDataType.PRIVATE_KEY: [
                r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----.*?-----END\s+(RSA\s+)?PRIVATE\s+KEY-----',
            ],
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {}
        for data_type, pattern_list in self.patterns.items():
            self.compiled_patterns[data_type] = [
                re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                for pattern in pattern_list
            ]
    
    def filter_sensitive_data(self, text: str, mask_char: str = "*") -> str:
        """
        Filter sensitive data from text.
        
        Args:
            text: Text to filter
            mask_char: Character to use for masking
        
        Returns:
            Filtered text with sensitive data masked
        """
        if not text:
            return text
        
        filtered_text = text
        
        for data_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if data_type in [SensitiveDataType.EMAIL, SensitiveDataType.IP_ADDRESS]:
                    # For emails and IPs, replace with type indicator
                    filtered_text = pattern.sub(f'[{data_type.value.upper()}_REDACTED]', filtered_text)
                else:
                    # For other sensitive data, mask with asterisks
                    def mask_match(match):
                        if len(match.groups()) > 0:
                            # Replace the captured group (sensitive part)
                            sensitive_part = match.group(1)
                            mask_length = min(len(sensitive_part), 8)  # Limit mask length
                            replacement = mask_char * mask_length
                            return match.group(0).replace(sensitive_part, replacement)
                        else:
                            # Replace entire match
                            mask_length = min(len(match.group(0)), 8)
                            return mask_char * mask_length
                    
                    filtered_text = pattern.sub(mask_match, filtered_text)
        
        return filtered_text
    
    def filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data from dictionary."""
        if not isinstance(data, dict):
            return data
        
        filtered_data = {}
        sensitive_keys = {
            'password', 'passwd', 'pwd', 'token', 'api_key', 'apikey',
            'session_id', 'sessionid', 'private_key', 'secret', 'hash'
        }
        
        for key, value in data.items():
            key_lower = key.lower()
            
            if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
                # Mask sensitive values
                if isinstance(value, str) and len(value) > 0:
                    filtered_data[key] = "***REDACTED***"
                else:
                    filtered_data[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered_data[key] = self.filter_dict(value)
            elif isinstance(value, str):
                filtered_data[key] = self.filter_sensitive_data(value)
            else:
                filtered_data[key] = value
        
        return filtered_data


class SecureLogFormatter(logging.Formatter):
    """Custom formatter that filters sensitive data and structures logs."""
    
    def __init__(self, include_sensitive: bool = False):
        super().__init__()
        self.include_sensitive = include_sensitive
        self.sensitive_filter = SensitiveDataFilter()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with sensitive data filtering."""
        # Create structured log entry
        log_entry = LogEntry(
            timestamp=record.created,
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process,
        )
        
        # Add additional context if available
        if hasattr(record, 'user_id'):
            log_entry.user_id = record.user_id
        if hasattr(record, 'session_id'):
            log_entry.session_id = record.session_id
        if hasattr(record, 'request_id'):
            log_entry.request_id = record.request_id
        if hasattr(record, 'ip_address'):
            log_entry.ip_address = record.ip_address
        if hasattr(record, 'user_agent'):
            log_entry.user_agent = record.user_agent
        if hasattr(record, 'additional_data'):
            log_entry.additional_data = record.additional_data
        
        # Add exception info if present
        if record.exc_info:
            log_entry.exception_info = self.formatException(record.exc_info)
        
        # Convert to dictionary
        log_dict = log_entry.to_dict()
        
        # Filter sensitive data unless specifically included
        if not self.include_sensitive:
            log_dict = self.sensitive_filter.filter_dict(log_dict)
            if log_entry.message:
                log_dict['message'] = self.sensitive_filter.filter_sensitive_data(log_entry.message)
            if log_entry.exception_info:
                log_dict['exception_info'] = self.sensitive_filter.filter_sensitive_data(log_entry.exception_info)
        
        # Return as JSON
        return json.dumps(log_dict, separators=(',', ':'))


class AuditLogger:
    """Specialized logger for audit events."""
    
    def __init__(self, name: str = "audit"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(AUDIT_LEVEL)
    
    def log_auth_attempt(self, user_id: Optional[str], ip_address: str, 
                        success: bool, method: str = "password"):
        """Log authentication attempt."""
        self.logger.log(AUDIT_LEVEL, f"Auth attempt: user={user_id}, ip={ip_address}, "
                                    f"success={success}, method={method}",
                       extra={
                           'event_type': 'auth_attempt',
                           'user_id': user_id,
                           'ip_address': ip_address,
                           'success': success,
                           'auth_method': method
                       })
    
    def log_permission_check(self, user_id: str, resource: str, action: str, granted: bool):
        """Log permission check."""
        self.logger.log(AUDIT_LEVEL, f"Permission check: user={user_id}, "
                                    f"resource={resource}, action={action}, granted={granted}",
                       extra={
                           'event_type': 'permission_check',
                           'user_id': user_id,
                           'resource': resource,
                           'action': action,
                           'granted': granted
                       })
    
    def log_data_access(self, user_id: str, resource_type: str, resource_id: str, action: str):
        """Log data access."""
        self.logger.log(AUDIT_LEVEL, f"Data access: user={user_id}, "
                                    f"type={resource_type}, id={resource_id}, action={action}",
                       extra={
                           'event_type': 'data_access',
                           'user_id': user_id,
                           'resource_type': resource_type,
                           'resource_id': resource_id,
                           'action': action
                       })
    
    def log_admin_action(self, admin_user_id: str, action: str, target: str, details: Dict[str, Any]):
        """Log administrative action."""
        self.logger.log(AUDIT_LEVEL, f"Admin action: admin={admin_user_id}, "
                                    f"action={action}, target={target}",
                       extra={
                           'event_type': 'admin_action',
                           'admin_user_id': admin_user_id,
                           'action': action,
                           'target': target,
                           'additional_data': details
                       })


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self, name: str = "security"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(SECURITY_LEVEL)
    
    def log_security_violation(self, violation_type: str, details: str, 
                             ip_address: Optional[str] = None, user_id: Optional[str] = None):
        """Log security violation."""
        self.logger.log(SECURITY_LEVEL, f"Security violation: {violation_type} - {details}",
                       extra={
                           'event_type': 'security_violation',
                           'violation_type': violation_type,
                           'ip_address': ip_address,
                           'user_id': user_id,
                           'additional_data': {'details': details}
                       })
    
    def log_rate_limit_exceeded(self, identifier: str, limit_type: str, 
                               ip_address: Optional[str] = None):
        """Log rate limit exceeded."""
        self.logger.log(SECURITY_LEVEL, f"Rate limit exceeded: {limit_type} for {identifier}",
                       extra={
                           'event_type': 'rate_limit_exceeded',
                           'identifier': identifier,
                           'limit_type': limit_type,
                           'ip_address': ip_address
                       })
    
    def log_suspicious_activity(self, activity_type: str, description: str,
                               ip_address: Optional[str] = None, user_id: Optional[str] = None):
        """Log suspicious activity."""
        self.logger.log(SECURITY_LEVEL, f"Suspicious activity: {activity_type} - {description}",
                       extra={
                           'event_type': 'suspicious_activity',
                           'activity_type': activity_type,
                           'ip_address': ip_address,
                           'user_id': user_id,
                           'additional_data': {'description': description}
                       })


class LogRotationManager:
    """Manages log file rotation and compression."""
    
    def __init__(self, log_dir: Path, max_size_mb: int = 100, max_files: int = 10):
        self.log_dir = Path(log_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_files = max_files
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def should_rotate(self, log_file: Path) -> bool:
        """Check if log file should be rotated."""
        if not log_file.exists():
            return False
        return log_file.stat().st_size >= self.max_size_bytes
    
    def rotate_log(self, log_file: Path):
        """Rotate and compress log file."""
        if not self.should_rotate(log_file):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"{log_file.stem}_{timestamp}.log.gz"
        rotated_path = self.log_dir / rotated_name
        
        # Compress the log file
        with open(log_file, 'rb') as f_in:
            with gzip.open(rotated_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Clear the original log file
        log_file.write_text("")
        
        # Clean up old log files
        self._cleanup_old_logs(log_file.stem)
    
    def _cleanup_old_logs(self, log_stem: str):
        """Clean up old rotated log files."""
        pattern = f"{log_stem}_*.log.gz"
        old_logs = sorted(self.log_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        
        if len(old_logs) > self.max_files:
            for old_log in old_logs[:-self.max_files]:
                old_log.unlink()


class SecureLoggerManager:
    """Central manager for secure logging configuration."""
    
    def __init__(self, log_dir: str = "/var/log/hashmancer", 
                 include_sensitive: bool = False):
        self.log_dir = Path(log_dir)
        self.include_sensitive = include_sensitive
        self.rotation_manager = LogRotationManager(self.log_dir)
        
        # Create specialized loggers
        self.audit_logger = AuditLogger()
        self.security_logger = SecurityLogger()
        
        # Setup logging configuration
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup secure logging configuration."""
        # Create formatters
        secure_formatter = SecureLogFormatter(include_sensitive=self.include_sensitive)
        
        # Setup file handlers with rotation
        self._setup_file_handler("application.log", logging.INFO, secure_formatter)
        self._setup_file_handler("audit.log", AUDIT_LEVEL, secure_formatter, "audit")
        self._setup_file_handler("security.log", SECURITY_LEVEL, secure_formatter, "security")
        self._setup_file_handler("error.log", logging.ERROR, secure_formatter)
        
        # Setup console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Add console handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.DEBUG)
    
    def _setup_file_handler(self, filename: str, level: int, formatter: logging.Formatter,
                           logger_name: Optional[str] = None):
        """Setup file handler with rotation."""
        log_file = self.log_dir / filename
        
        handler = logging.FileHandler(log_file)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger()
        
        logger.addHandler(handler)
    
    def rotate_logs(self):
        """Manually trigger log rotation."""
        for log_file in self.log_dir.glob("*.log"):
            self.rotation_manager.rotate_log(log_file)
    
    def get_audit_logger(self) -> AuditLogger:
        """Get audit logger instance."""
        return self.audit_logger
    
    def get_security_logger(self) -> SecurityLogger:
        """Get security logger instance."""
        return self.security_logger


# Global logger manager
_logger_manager: Optional[SecureLoggerManager] = None
_manager_lock = threading.Lock()


def get_logger_manager() -> SecureLoggerManager:
    """Get the global logger manager instance."""
    global _logger_manager
    
    if _logger_manager is None:
        with _manager_lock:
            if _logger_manager is None:
                _logger_manager = SecureLoggerManager()
    
    return _logger_manager


def get_audit_logger() -> AuditLogger:
    """Get audit logger instance."""
    return get_logger_manager().get_audit_logger()


def get_security_logger() -> SecurityLogger:
    """Get security logger instance."""
    return get_logger_manager().get_security_logger()


def secure_log(level: LogLevel, message: str, **kwargs):
    """Log message with secure formatting."""
    logger = logging.getLogger()
    logger.log(level.value, message, extra=kwargs)


# Convenience functions
def log_audit(message: str, **kwargs):
    """Log audit message."""
    secure_log(LogLevel.AUDIT, message, **kwargs)


def log_security(message: str, **kwargs):
    """Log security message."""
    secure_log(LogLevel.SECURITY, message, **kwargs)


def log_with_context(level: LogLevel, message: str, user_id: Optional[str] = None,
                    session_id: Optional[str] = None, ip_address: Optional[str] = None,
                    **kwargs):
    """Log message with user context."""
    extra = {
        'user_id': user_id,
        'session_id': session_id,
        'ip_address': ip_address,
        **kwargs
    }
    secure_log(level, message, **extra)