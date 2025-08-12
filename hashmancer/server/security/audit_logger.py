"""Comprehensive audit logging system for security compliance."""

import time
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import asyncio
from collections import deque
import hashlib
from fastapi import Request

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    USER_CREATED = "user_created"
    USER_MODIFIED = "user_modified"
    USER_DELETED = "user_deleted"
    PASSWORD_CHANGED = "password_changed"
    PERMISSION_CHANGED = "permission_changed"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    FILE_UPLOAD = "file_upload"
    FILE_DOWNLOAD = "file_download"
    CONFIG_CHANGED = "config_changed"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SECURITY_VIOLATION = "security_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ADMIN_ACTION = "admin_action"
    API_ACCESS = "api_access"
    WORKER_REGISTRATION = "worker_registration"
    BATCH_SUBMITTED = "batch_submitted"
    HASH_CRACKED = "hash_cracked"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event record."""
    timestamp: float
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: str
    user_agent: Optional[str]
    resource: Optional[str]
    action: str
    details: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    event_id: Optional[str] = None
    
    def __post_init__(self):
        if self.event_id is None:
            # Generate unique event ID
            data = f"{self.timestamp}{self.event_type.value}{self.ip_address}{self.action}"
            self.event_id = hashlib.sha256(data.encode()).hexdigest()[:16]


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, log_file: Optional[Path] = None, max_memory_events: int = 10000):
        self.log_file = log_file or Path("audit.log")
        self.max_memory_events = max_memory_events
        self._memory_events: deque = deque(maxlen=max_memory_events)
        self._event_counts = {event_type: 0 for event_type in AuditEventType}
        self._severity_counts = {severity: 0 for severity in AuditSeverity}
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file logger
        self._file_logger = logging.getLogger('audit')
        self._file_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self._file_logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self._file_logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self._file_logger.propagate = False
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        success: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: str = "unknown",
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        error_message: Optional[str] = None,
        request: Optional[Request] = None
    ) -> str:
        """Log an audit event."""
        
        # Extract details from request if provided
        if request:
            ip_address = self._extract_ip(request)
            user_agent = request.headers.get('User-Agent')
            if user_id is None:
                # Try to extract user from request state or headers
                user_id = getattr(request.state, 'user_id', None)
            if session_id is None:
                session_id = getattr(request.state, 'session_id', None)
        
        # Create audit event
        event = AuditEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            details=details or {},
            success=success,
            error_message=error_message
        )
        
        # Add to memory storage
        self._memory_events.append(event)
        
        # Update counters
        self._event_counts[event_type] += 1
        self._severity_counts[severity] += 1
        
        # Log to file
        self._log_to_file(event)
        
        # Log critical events to main logger
        if severity == AuditSeverity.CRITICAL:
            logger.critical(f"AUDIT: {event_type.value} - {action} - {details}")
        elif severity == AuditSeverity.HIGH:
            logger.error(f"AUDIT: {event_type.value} - {action} - {details}")
        elif severity == AuditSeverity.MEDIUM:
            logger.warning(f"AUDIT: {event_type.value} - {action} - {details}")
        
        return event.event_id
    
    def _extract_ip(self, request: Request) -> str:
        """Extract real IP address from request."""
        # Check for proxy headers
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # Take the first IP (original client)
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip.strip()
        
        cf_ip = request.headers.get('CF-Connecting-IP')
        if cf_ip:
            return cf_ip.strip()
        
        # Fallback to client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _log_to_file(self, event: AuditEvent):
        """Log event to file."""
        try:
            log_entry = {
                'timestamp': event.timestamp,
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'severity': event.severity.value,
                'user_id': event.user_id,
                'session_id': event.session_id,
                'ip_address': event.ip_address,
                'user_agent': event.user_agent,
                'resource': event.resource,
                'action': event.action,
                'success': event.success,
                'details': event.details,
                'error_message': event.error_message
            }
            
            # Log as JSON for easy parsing
            self._file_logger.info(json.dumps(log_entry, default=str))
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_events(
        self,
        limit: int = 100,
        event_type: Optional[AuditEventType] = None,
        severity: Optional[AuditSeverity] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        success_only: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Get audit events with filtering."""
        events = list(self._memory_events)
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if ip_address:
            events = [e for e in events if e.ip_address == ip_address]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        if success_only is not None:
            events = [e for e in events if e.success == success_only]
        
        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        events = events[:limit]
        
        # Convert to dictionaries
        return [asdict(event) for event in events]
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit statistics for the specified time period."""
        cutoff = time.time() - (hours * 3600)
        recent_events = [e for e in self._memory_events if e.timestamp >= cutoff]
        
        # Event type distribution
        event_type_counts = {}
        severity_counts = {}
        user_activity = {}
        ip_activity = {}
        hourly_activity = [0] * hours
        
        for event in recent_events:
            # Event types
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # Severities
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # User activity
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
            
            # IP activity
            ip_activity[event.ip_address] = ip_activity.get(event.ip_address, 0) + 1
            
            # Hourly distribution
            hours_ago = int((time.time() - event.timestamp) / 3600)
            if 0 <= hours_ago < hours:
                hourly_activity[hours - 1 - hours_ago] += 1
        
        # Top activities
        top_users = sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        top_ips = sorted(ip_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Failed events
        failed_events = [e for e in recent_events if not e.success]
        failed_count = len(failed_events)
        
        # Security events
        security_events = [e for e in recent_events if e.event_type in [
            AuditEventType.SECURITY_VIOLATION,
            AuditEventType.SUSPICIOUS_ACTIVITY,
            AuditEventType.RATE_LIMIT_EXCEEDED,
            AuditEventType.LOGIN_FAILED
        ]]
        
        return {
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'failed_events': failed_count,
            'success_rate': (len(recent_events) - failed_count) / max(len(recent_events), 1),
            'event_type_distribution': event_type_counts,
            'severity_distribution': severity_counts,
            'top_users': top_users,
            'top_ip_addresses': top_ips,
            'hourly_activity': hourly_activity,
            'security_events': len(security_events),
            'recent_security_events': [asdict(e) for e in security_events[-5:]]
        }
    
    def search_events(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search audit events by query string."""
        query_lower = query.lower()
        matching_events = []
        
        for event in self._memory_events:
            # Search in multiple fields
            searchable_text = " ".join([
                event.action.lower(),
                str(event.details).lower(),
                event.user_id or "",
                event.ip_address,
                event.resource or "",
                event.error_message or ""
            ])
            
            if query_lower in searchable_text:
                matching_events.append(event)
        
        # Sort by timestamp (newest first) and limit
        matching_events.sort(key=lambda e: e.timestamp, reverse=True)
        matching_events = matching_events[:limit]
        
        return [asdict(event) for event in matching_events]
    
    def export_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        format: str = 'json'
    ) -> str:
        """Export audit events to JSON or CSV format."""
        events = list(self._memory_events)
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        events.sort(key=lambda e: e.timestamp)
        
        if format.lower() == 'json':
            return json.dumps([asdict(event) for event in events], indent=2, default=str)
        
        elif format.lower() == 'csv':
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write header
            if events:
                writer.writerow([
                    'timestamp', 'event_id', 'event_type', 'severity', 'user_id',
                    'session_id', 'ip_address', 'user_agent', 'resource', 'action',
                    'success', 'details', 'error_message'
                ])
                
                # Write data
                for event in events:
                    writer.writerow([
                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp)),
                        event.event_id,
                        event.event_type.value,
                        event.severity.value,
                        event.user_id or '',
                        event.session_id or '',
                        event.ip_address,
                        event.user_agent or '',
                        event.resource or '',
                        event.action,
                        event.success,
                        json.dumps(event.details),
                        event.error_message or ''
                    ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_log(
    event_type: AuditEventType,
    action: str,
    success: bool = True,
    **kwargs
) -> str:
    """Convenience function for logging audit events."""
    return get_audit_logger().log_event(event_type, action, success, **kwargs)


def get_audit_logs(**kwargs) -> List[Dict[str, Any]]:
    """Get audit logs with filtering."""
    return get_audit_logger().get_events(**kwargs)


def get_audit_statistics(hours: int = 24) -> Dict[str, Any]:
    """Get audit statistics."""
    return get_audit_logger().get_statistics(hours)


def search_audit_logs(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search audit logs."""
    return get_audit_logger().search_events(query, limit)