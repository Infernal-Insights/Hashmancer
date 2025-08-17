"""
Centralized Error Handling System
Provides consistent error handling, logging, and response formatting
"""

import logging
import traceback
import time
import json
import uuid
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for classification."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RATE_LIMIT = "rate_limit"
    SYSTEM_ERROR = "system_error"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for errors."""
    user_id: Optional[str] = None
    client_ip: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: float = 0
    additional_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class HashmancerError:
    """Standardized error structure."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    code: str
    context: ErrorContext
    internal_message: Optional[str] = None
    suggestions: Optional[List[str]] = None
    retry_after: Optional[int] = None
    
    def to_dict(self, include_internal: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result = {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "code": self.code,
            "timestamp": self.context.timestamp,
        }
        
        if self.suggestions:
            result["suggestions"] = self.suggestions
            
        if self.retry_after:
            result["retry_after"] = self.retry_after
            
        if include_internal and self.internal_message:
            result["internal_message"] = self.internal_message
            
        return result


class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_history = 1000
        
        # Error mapping configurations
        self.status_code_map = {
            ErrorCategory.AUTHENTICATION: 401,
            ErrorCategory.AUTHORIZATION: 403,
            ErrorCategory.VALIDATION: 400,
            ErrorCategory.RESOURCE_NOT_FOUND: 404,
            ErrorCategory.RATE_LIMIT: 429,
            ErrorCategory.SYSTEM_ERROR: 500,
            ErrorCategory.EXTERNAL_SERVICE: 502,
            ErrorCategory.DATABASE: 503,
            ErrorCategory.NETWORK: 503,
            ErrorCategory.SECURITY: 403,
        }
        
        # Suggestions mapping
        self.suggestion_map = {
            ErrorCategory.AUTHENTICATION: [
                "Check your authentication credentials",
                "Ensure your token is valid and not expired"
            ],
            ErrorCategory.AUTHORIZATION: [
                "Contact an administrator for required permissions",
                "Verify your access level for this resource"
            ],
            ErrorCategory.VALIDATION: [
                "Check the format and content of your request",
                "Refer to the API documentation for valid parameters"
            ],
            ErrorCategory.RATE_LIMIT: [
                "Reduce your request frequency",
                "Implement exponential backoff in your client"
            ],
            ErrorCategory.SYSTEM_ERROR: [
                "Try again in a few moments",
                "Contact support if the problem persists"
            ]
        }
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        return f"err_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    def _extract_context_from_request(self, request: Optional[Request]) -> ErrorContext:
        """Extract context information from FastAPI request."""
        if not request:
            return ErrorContext()
        
        return ErrorContext(
            client_ip=request.client.host if request.client else None,
            endpoint=str(request.url.path) if request.url else None,
            method=request.method,
            user_agent=request.headers.get("user-agent"),
            request_id=request.headers.get("x-request-id"),
            additional_data={
                "query_params": dict(request.query_params) if hasattr(request, 'query_params') else None
            }
        )
    
    def create_error(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        message: str,
        code: str,
        context: Optional[ErrorContext] = None,
        internal_message: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        retry_after: Optional[int] = None
    ) -> HashmancerError:
        """Create a standardized error."""
        error_id = self._generate_error_id()
        
        if context is None:
            context = ErrorContext()
        
        if suggestions is None:
            suggestions = self.suggestion_map.get(category, [])
        
        error = HashmancerError(
            error_id=error_id,
            category=category,
            severity=severity,
            message=message,
            code=code,
            context=context,
            internal_message=internal_message,
            suggestions=suggestions,
            retry_after=retry_after
        )
        
        # Track error statistics
        self._track_error(error)
        
        return error
    
    def _track_error(self, error: HashmancerError):
        """Track error for statistics and monitoring."""
        category_key = error.category.value
        self.error_counts[category_key] = self.error_counts.get(category_key, 0) + 1
        
        # Add to history (with rotation)
        self.error_history.append({
            "error_id": error.error_id,
            "category": error.category.value,
            "severity": error.severity.value,
            "timestamp": error.context.timestamp,
            "code": error.code
        })
        
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
    
    def log_error(self, error: HashmancerError, exception: Optional[Exception] = None):
        """Log error with appropriate level."""
        log_data = {
            "error_id": error.error_id,
            "category": error.category.value,
            "code": error.code,
            "message": error.message,
            "context": asdict(error.context)
        }
        
        if error.internal_message:
            log_data["internal_message"] = error.internal_message
        
        log_message = f"[{error.error_id}] {error.category.value.upper()}: {error.message}"
        
        if exception:
            log_data["exception"] = str(exception)
            log_data["traceback"] = traceback.format_exc()
        
        # Log based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra=log_data)
        else:
            logger.info(log_message, extra=log_data)
    
    def create_http_response(self, error: HashmancerError, include_internal: bool = False) -> JSONResponse:
        """Create HTTP response from error."""
        status_code = self.status_code_map.get(error.category, 500)
        
        headers = {}
        if error.retry_after:
            headers["Retry-After"] = str(error.retry_after)
        
        return JSONResponse(
            status_code=status_code,
            content=error.to_dict(include_internal=include_internal),
            headers=headers
        )
    
    def handle_exception(
        self,
        exception: Exception,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        context: Optional[ErrorContext] = None,
        user_message: Optional[str] = None,
        code: Optional[str] = None
    ) -> HashmancerError:
        """Handle exception and convert to standardized error."""
        
        # Auto-detect category and severity if not provided
        if category is None:
            category = self._categorize_exception(exception)
        
        if severity is None:
            severity = self._assess_severity(exception, category)
        
        if code is None:
            code = f"{category.value.upper()}_{type(exception).__name__}"
        
        if user_message is None:
            user_message = self._generate_user_message(exception, category)
        
        error = self.create_error(
            category=category,
            severity=severity,
            message=user_message,
            code=code,
            context=context,
            internal_message=str(exception)
        )
        
        self.log_error(error, exception)
        return error
    
    def _categorize_exception(self, exception: Exception) -> ErrorCategory:
        """Auto-categorize exception by type."""
        exception_type = type(exception).__name__
        
        if isinstance(exception, HTTPException):
            if exception.status_code == 401:
                return ErrorCategory.AUTHENTICATION
            elif exception.status_code == 403:
                return ErrorCategory.AUTHORIZATION
            elif exception.status_code == 404:
                return ErrorCategory.RESOURCE_NOT_FOUND
            elif exception.status_code == 429:
                return ErrorCategory.RATE_LIMIT
            elif 400 <= exception.status_code < 500:
                return ErrorCategory.VALIDATION
            else:
                return ErrorCategory.SYSTEM_ERROR
        
        # Database errors
        if any(db_term in exception_type.lower() for db_term in ['redis', 'connection', 'timeout']):
            return ErrorCategory.DATABASE
        
        # Network errors
        if any(net_term in exception_type.lower() for net_term in ['network', 'socket', 'connect']):
            return ErrorCategory.NETWORK
        
        # Security errors
        if any(sec_term in exception_type.lower() for sec_term in ['security', 'auth', 'permission']):
            return ErrorCategory.SECURITY
        
        return ErrorCategory.SYSTEM_ERROR
    
    def _assess_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess severity based on exception type and category."""
        if category in [ErrorCategory.SECURITY, ErrorCategory.SYSTEM_ERROR]:
            return ErrorSeverity.HIGH
        elif category in [ErrorCategory.DATABASE, ErrorCategory.EXTERNAL_SERVICE]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _generate_user_message(self, exception: Exception, category: ErrorCategory) -> str:
        """Generate user-friendly message from exception."""
        if isinstance(exception, HTTPException):
            return exception.detail
        
        category_messages = {
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please check your credentials.",
            ErrorCategory.AUTHORIZATION: "You don't have permission to access this resource.",
            ErrorCategory.VALIDATION: "Invalid request data. Please check your input.",
            ErrorCategory.RESOURCE_NOT_FOUND: "The requested resource was not found.",
            ErrorCategory.RATE_LIMIT: "Too many requests. Please slow down.",
            ErrorCategory.SYSTEM_ERROR: "An internal error occurred. Please try again later.",
            ErrorCategory.EXTERNAL_SERVICE: "External service is unavailable. Please try again later.",
            ErrorCategory.DATABASE: "Database operation failed. Please try again.",
            ErrorCategory.NETWORK: "Network error occurred. Please check your connection.",
            ErrorCategory.SECURITY: "Security violation detected.",
        }
        
        return category_messages.get(category, "An unexpected error occurred.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error handler statistics."""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "recent_errors": len([
                e for e in self.error_history 
                if time.time() - e["timestamp"] < 3600  # Last hour
            ]),
            "history_size": len(self.error_history)
        }


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(
    category: Optional[ErrorCategory] = None,
    severity: Optional[ErrorSeverity] = None,
    user_message: Optional[str] = None,
    include_internal: bool = False
):
    """Decorator for automatic error handling."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Extract request from args if available
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
                
                context = error_handler._extract_context_from_request(request)
                error = error_handler.handle_exception(
                    e, category, severity, context, user_message
                )
                
                # For async functions, return HTTPException
                raise HTTPException(
                    status_code=error_handler.status_code_map.get(error.category, 500),
                    detail=error.to_dict(include_internal=include_internal)
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext()
                error = error_handler.handle_exception(
                    e, category, severity, context, user_message
                )
                
                raise HTTPException(
                    status_code=error_handler.status_code_map.get(error.category, 500),
                    detail=error.to_dict(include_internal=include_internal)
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Convenience functions for common error types
def create_validation_error(message: str, context: Optional[ErrorContext] = None) -> HashmancerError:
    """Create validation error."""
    return error_handler.create_error(
        ErrorCategory.VALIDATION,
        ErrorSeverity.LOW,
        message,
        "VALIDATION_ERROR",
        context
    )


def create_auth_error(message: str, context: Optional[ErrorContext] = None) -> HashmancerError:
    """Create authentication error."""
    return error_handler.create_error(
        ErrorCategory.AUTHENTICATION,
        ErrorSeverity.MEDIUM,
        message,
        "AUTH_ERROR",
        context
    )


def create_system_error(message: str, context: Optional[ErrorContext] = None) -> HashmancerError:
    """Create system error."""
    return error_handler.create_error(
        ErrorCategory.SYSTEM_ERROR,
        ErrorSeverity.HIGH,
        message,
        "SYSTEM_ERROR",
        context
    )


def create_rate_limit_error(retry_after: int, context: Optional[ErrorContext] = None) -> HashmancerError:
    """Create rate limit error."""
    return error_handler.create_error(
        ErrorCategory.RATE_LIMIT,
        ErrorSeverity.MEDIUM,
        "Rate limit exceeded. Please slow down your requests.",
        "RATE_LIMIT_EXCEEDED",
        context,
        retry_after=retry_after
    )