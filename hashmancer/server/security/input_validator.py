"""Input validation and sanitization for security hardening."""

import re
import html
import urllib.parse
import ipaddress
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import bleach
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class InputType(Enum):
    """Types of input validation."""
    USERNAME = "username"
    EMAIL = "email"
    PASSWORD = "password"
    IP_ADDRESS = "ip_address"
    HOSTNAME = "hostname"
    URL = "url"
    FILENAME = "filename"
    PATH = "path"
    HASH = "hash"
    UUID = "uuid"
    ALPHANUMERIC = "alphanumeric"
    NUMERIC = "numeric"
    TEXT = "text"
    HTML = "html"
    JSON = "json"
    SQL_SAFE = "sql_safe"


@dataclass
class ValidationRule:
    """Input validation rule."""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_chars: Optional[str] = None
    forbidden_chars: Optional[str] = None
    required: bool = True
    custom_validator: Optional[Callable] = None
    description: str = ""


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self):
        # Pre-compiled patterns for performance
        self.patterns = {
            InputType.USERNAME: re.compile(r'^[a-zA-Z0-9_-]{3,32}$'),
            InputType.EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            InputType.IP_ADDRESS: re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'),
            InputType.HOSTNAME: re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'),
            InputType.URL: re.compile(r'^https?://[^\s/$.?#].[^\s]*$', re.I),
            InputType.FILENAME: re.compile(r'^[a-zA-Z0-9._-]+$'),
            InputType.HASH: re.compile(r'^[a-fA-F0-9]+$'),
            InputType.UUID: re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'),
            InputType.ALPHANUMERIC: re.compile(r'^[a-zA-Z0-9]+$'),
            InputType.NUMERIC: re.compile(r'^[0-9]+$'),
        }
        
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            (re.compile(r'(?i)(union|select|insert|delete|drop|exec|script|create|alter)', re.I), 'SQL injection attempt'),
            (re.compile(r'(?i)(<script|javascript:|vbscript:|onload=|onerror=|onclick=)', re.I), 'XSS attempt'),
            (re.compile(r'(?i)(\.\.\/|\.\.\\|\/etc\/|\/proc\/|\/dev\/|file:\/\/)', re.I), 'Path traversal attempt'),
            (re.compile(r'(?i)(cmd|powershell|bash|sh|nc|netcat|curl|wget)[\s=]', re.I), 'Command injection attempt'),
            (re.compile(r'(?i)(\${|<%|%{|{{)', re.I), 'Template injection attempt'),
            (re.compile(r'(?i)(eval\s*\(|system\s*\(|exec\s*\(|subprocess|os\.)', re.I), 'Code execution attempt'),
            (re.compile(r'(?i)(ldap|smtp|ftp|sftp|ssh)://', re.I), 'Protocol injection attempt'),
            (re.compile(r'(?i)(0x[0-9a-f]+|\\\d{1,3})', re.I), 'Encoding bypass attempt'),
        ]
        
        # Default validation rules
        self.default_rules = {
            InputType.USERNAME: ValidationRule(
                min_length=3, max_length=32, pattern=r'^[a-zA-Z0-9_-]+$',
                description="Username: 3-32 chars, alphanumeric, underscore, hyphen"
            ),
            InputType.EMAIL: ValidationRule(
                min_length=5, max_length=254,
                description="Valid email address"
            ),
            InputType.PASSWORD: ValidationRule(
                min_length=8, max_length=128,
                description="Password: 8-128 characters"
            ),
            InputType.IP_ADDRESS: ValidationRule(
                description="Valid IPv4 or IPv6 address"
            ),
            InputType.HOSTNAME: ValidationRule(
                min_length=1, max_length=253,
                description="Valid hostname or domain name"
            ),
            InputType.URL: ValidationRule(
                min_length=10, max_length=2048,
                description="Valid HTTP/HTTPS URL"
            ),
            InputType.FILENAME: ValidationRule(
                min_length=1, max_length=255, 
                allowed_chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-',
                description="Safe filename characters only"
            ),
            InputType.PATH: ValidationRule(
                min_length=1, max_length=4096,
                forbidden_chars='<>"|*?',
                description="File system path"
            ),
            InputType.HASH: ValidationRule(
                min_length=32, max_length=128, pattern=r'^[a-fA-F0-9]+$',
                description="Hexadecimal hash"
            ),
            InputType.TEXT: ValidationRule(
                max_length=10000,
                description="General text input"
            )
        }
        
        # HTML sanitization settings
        self.allowed_html_tags = [
            'b', 'i', 'u', 'em', 'strong', 'p', 'br', 'span', 'div',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li',
            'blockquote', 'code', 'pre'
        ]
        
        self.allowed_html_attributes = {
            '*': ['class', 'id'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title'],
        }
    
    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Sanitize string input by removing dangerous content."""
        if not isinstance(value, str):
            return str(value)[:max_length]
        
        # Basic sanitization
        sanitized = html.escape(value)
        sanitized = sanitized[:max_length]
        
        return sanitized
    
    def validate(
        self, 
        value: Any, 
        input_type: InputType, 
        custom_rule: Optional[ValidationRule] = None,
        field_name: str = "input"
    ) -> Any:
        """Validate input against type and rules."""
        
        # Handle None/empty values
        if value is None or value == "":
            rule = custom_rule or self.default_rules.get(input_type)
            if rule and rule.required:
                raise ValidationError(f"{field_name} is required")
            return value
        
        # Convert to string for most validations
        str_value = str(value).strip()
        
        # Get validation rule
        rule = custom_rule or self.default_rules.get(input_type, ValidationRule())
        
        # Check length constraints
        if rule.min_length is not None and len(str_value) < rule.min_length:
            raise ValidationError(f"{field_name} must be at least {rule.min_length} characters")
        
        if rule.max_length is not None and len(str_value) > rule.max_length:
            raise ValidationError(f"{field_name} must be at most {rule.max_length} characters")
        
        # Check for dangerous patterns
        self._check_dangerous_patterns(str_value, field_name)
        
        # Type-specific validation
        validated_value = self._validate_by_type(str_value, input_type, field_name)
        
        # Check custom pattern
        if rule.pattern:
            if not re.match(rule.pattern, str_value):
                raise ValidationError(f"{field_name} format is invalid: {rule.description}")
        
        # Check allowed characters
        if rule.allowed_chars:
            if not all(c in rule.allowed_chars for c in str_value):
                raise ValidationError(f"{field_name} contains invalid characters")
        
        # Check forbidden characters
        if rule.forbidden_chars:
            if any(c in rule.forbidden_chars for c in str_value):
                raise ValidationError(f"{field_name} contains forbidden characters")
        
        # Custom validator
        if rule.custom_validator:
            try:
                validated_value = rule.custom_validator(validated_value)
            except Exception as e:
                raise ValidationError(f"{field_name} validation failed: {str(e)}")
        
        return validated_value
    
    def _check_dangerous_patterns(self, value: str, field_name: str):
        """Check for dangerous input patterns."""
        for pattern, description in self.dangerous_patterns:
            if pattern.search(value):
                logger.warning(f"Dangerous pattern detected in {field_name}: {description}")
                raise ValidationError(f"{field_name} contains potentially dangerous content")
    
    def _validate_by_type(self, value: str, input_type: InputType, field_name: str) -> Any:
        """Perform type-specific validation."""
        
        if input_type == InputType.USERNAME:
            if not self.patterns[InputType.USERNAME].match(value):
                raise ValidationError(f"{field_name} contains invalid characters for username")
            return value.lower()  # Normalize to lowercase
        
        elif input_type == InputType.EMAIL:
            if not self.patterns[InputType.EMAIL].match(value):
                raise ValidationError(f"{field_name} is not a valid email address")
            return value.lower()  # Normalize to lowercase
        
        elif input_type == InputType.PASSWORD:
            # Password strength checks
            if len(set(value)) < 4:
                raise ValidationError(f"{field_name} must contain at least 4 different characters")
            
            strength_checks = [
                (r'[a-z]', "lowercase letter"),
                (r'[A-Z]', "uppercase letter"),
                (r'[0-9]', "number"),
                (r'[^a-zA-Z0-9]', "special character")
            ]
            
            passed_checks = sum(1 for pattern, _ in strength_checks if re.search(pattern, value))
            if passed_checks < 3:
                raise ValidationError(f"{field_name} must contain at least 3 of: uppercase, lowercase, number, special character")
            
            return value  # Don't modify password
        
        elif input_type == InputType.IP_ADDRESS:
            try:
                # Validate IPv4 or IPv6
                ip = ipaddress.ip_address(value)
                return str(ip)
            except ValueError:
                raise ValidationError(f"{field_name} is not a valid IP address")
        
        elif input_type == InputType.HOSTNAME:
            if not self.patterns[InputType.HOSTNAME].match(value):
                raise ValidationError(f"{field_name} is not a valid hostname")
            return value.lower()
        
        elif input_type == InputType.URL:
            if not self.patterns[InputType.URL].match(value):
                raise ValidationError(f"{field_name} is not a valid URL")
            # Parse and validate URL
            try:
                parsed = urllib.parse.urlparse(value)
                if not parsed.scheme or not parsed.netloc:
                    raise ValidationError(f"{field_name} is not a complete URL")
                return value
            except Exception:
                raise ValidationError(f"{field_name} is not a valid URL")
        
        elif input_type == InputType.FILENAME:
            if not self.patterns[InputType.FILENAME].match(value):
                raise ValidationError(f"{field_name} contains invalid filename characters")
            # Check for reserved names on Windows
            reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
            if value.upper().split('.')[0] in reserved_names:
                raise ValidationError(f"{field_name} uses a reserved filename")
            return value
        
        elif input_type == InputType.HASH:
            if not self.patterns[InputType.HASH].match(value):
                raise ValidationError(f"{field_name} is not a valid hexadecimal hash")
            return value.lower()
        
        elif input_type == InputType.UUID:
            if not self.patterns[InputType.UUID].match(value):
                raise ValidationError(f"{field_name} is not a valid UUID")
            return value.lower()
        
        elif input_type == InputType.ALPHANUMERIC:
            if not self.patterns[InputType.ALPHANUMERIC].match(value):
                raise ValidationError(f"{field_name} must contain only letters and numbers")
            return value
        
        elif input_type == InputType.NUMERIC:
            if not self.patterns[InputType.NUMERIC].match(value):
                raise ValidationError(f"{field_name} must contain only numbers")
            try:
                return int(value)
            except ValueError:
                raise ValidationError(f"{field_name} is not a valid number")
        
        elif input_type == InputType.JSON:
            try:
                import json
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValidationError(f"{field_name} is not valid JSON")
        
        elif input_type == InputType.HTML:
            # Sanitize HTML
            return self.sanitize_html(value)
        
        elif input_type == InputType.SQL_SAFE:
            # Check for SQL injection patterns
            sql_patterns = [
                r"('|(\\'))",
                r"(\"|(\\\")",
                r"(;|\\;)",
                r"(\\|\\\\)",
                r"(\*|\\*)",
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, value, re.I):
                    raise ValidationError(f"{field_name} contains potentially unsafe SQL characters")
            
            return value
        
        return value
    
    def sanitize_input(self, value: str, input_type: InputType = InputType.TEXT) -> str:
        """Sanitize input for safe usage."""
        if not value:
            return value
        
        if input_type == InputType.HTML:
            return self.sanitize_html(value)
        elif input_type == InputType.TEXT:
            # Basic text sanitization
            return html.escape(value, quote=True)
        elif input_type == InputType.SQL_SAFE:
            # Escape single quotes for SQL
            return value.replace("'", "''")
        else:
            # Default: HTML escape
            return html.escape(value, quote=True)
    
    def sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML content using bleach."""
        try:
            return bleach.clean(
                html_content,
                tags=self.allowed_html_tags,
                attributes=self.allowed_html_attributes,
                strip=True
            )
        except Exception:
            # Fallback to simple HTML escaping
            return html.escape(html_content, quote=True)
    
    def validate_request_data(
        self, 
        data: Dict[str, Any], 
        validation_schema: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate entire request data against schema."""
        validated_data = {}
        errors = []
        
        for field_name, field_config in validation_schema.items():
            try:
                value = data.get(field_name)
                input_type = InputType(field_config.get('type', 'text'))
                custom_rule = field_config.get('rule')
                
                validated_value = self.validate(
                    value, input_type, custom_rule, field_name
                )
                validated_data[field_name] = validated_value
                
            except ValidationError as e:
                errors.append(str(e))
            except Exception as e:
                logger.error(f"Validation error for {field_name}: {e}")
                errors.append(f"Validation failed for {field_name}")
        
        if errors:
            raise ValidationError("; ".join(errors))
        
        return validated_data
    
    def is_safe_path(self, path: str) -> bool:
        """Check if a file path is safe (no directory traversal)."""
        try:
            # Normalize the path
            normalized = str(Path(path).resolve())
            
            # Check for directory traversal attempts
            if '..' in normalized or normalized.startswith('/'):
                return False
            
            # Check for suspicious patterns
            dangerous_paths = ['/etc/', '/proc/', '/sys/', '/dev/', '/root/', '/home/']
            if any(danger in normalized.lower() for danger in dangerous_paths):
                return False
            
            return True
        except Exception:
            return False
    
    def validate_worker_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate worker registration/update data."""
        validated = {}
        
        # Worker ID validation
        if 'worker_id' in data:
            validated['worker_id'] = self.validate(
                data['worker_id'], 
                InputType.ALPHANUMERIC, 
                min_length=8, 
                max_length=64,
                field_name="worker_id"
            )
        
        # Capabilities validation
        if 'capabilities' in data:
            caps = data['capabilities']
            if not isinstance(caps, dict):
                raise ValidationError("capabilities must be a dictionary")
            
            validated_caps = {}
            for key, value in caps.items():
                clean_key = self.validate(key, InputType.ALPHANUMERIC, field_name=f"capability key {key}")
                if isinstance(value, str):
                    clean_value = self.sanitize_string(value)
                elif isinstance(value, (int, float, bool)):
                    clean_value = value
                else:
                    raise ValidationError(f"Invalid capability value type for {key}")
                validated_caps[clean_key] = clean_value
            validated['capabilities'] = validated_caps
        
        # Status validation
        if 'status' in data:
            status = str(data['status']).lower()
            if status not in ['idle', 'busy', 'offline', 'error', 'ready']:
                raise ValidationError("Invalid worker status")
            validated['status'] = status
        
        # Metadata validation
        if 'metadata' in data:
            metadata = data['metadata']
            if not isinstance(metadata, dict):
                raise ValidationError("metadata must be a dictionary")
            
            validated_metadata = {}
            for key, value in metadata.items():
                clean_key = self.sanitize_string(key)[:50]  # Limit key length
                if isinstance(value, str):
                    clean_value = self.sanitize_string(value)[:200]  # Limit value length
                elif isinstance(value, (int, float, bool)):
                    clean_value = value
                else:
                    clean_value = str(value)[:200]
                validated_metadata[clean_key] = clean_value
            validated['metadata'] = validated_metadata
        
        return validated
    
    def validate_hash_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hash-related data."""
        validated = {}
        
        # Hash validation
        if 'hash' in data:
            hash_value = str(data['hash']).strip()
            validated['hash'] = self.validate(hash_value, InputType.HASH, field_name="hash")
        
        # Hash list validation
        if 'hashes' in data:
            hashes = data['hashes']
            if isinstance(hashes, str):
                # Parse JSON string
                try:
                    hashes = json.loads(hashes)
                except json.JSONDecodeError:
                    raise ValidationError("Invalid JSON in hashes field")
            
            if not isinstance(hashes, list):
                raise ValidationError("hashes must be a list")
            
            if len(hashes) > 10000:  # Limit batch size
                raise ValidationError("Too many hashes in batch (max 10000)")
            
            validated_hashes = []
            for i, hash_value in enumerate(hashes):
                try:
                    clean_hash = self.validate(str(hash_value).strip(), InputType.HASH, field_name=f"hash {i}")
                    validated_hashes.append(clean_hash)
                except ValidationError as e:
                    logger.warning(f"Invalid hash at index {i}: {e}")
                    continue  # Skip invalid hashes
            
            validated['hashes'] = validated_hashes
        
        # Algorithm validation
        if 'algorithm' in data:
            algo = str(data['algorithm']).lower().strip()
            # Allow common hash algorithms
            allowed_algos = {
                'md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512',
                'ntlm', 'lm', 'bcrypt', 'scrypt', 'argon2', 'pbkdf2',
                'mysql', 'mssql', 'oracle', 'postgresql'
            }
            if algo not in allowed_algos:
                raise ValidationError(f"Unsupported hash algorithm: {algo}")
            validated['algorithm'] = algo
        
        return validated
    
    def validate_job_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate job submission data."""
        validated = {}
        
        # Job ID validation
        if 'job_id' in data:
            validated['job_id'] = self.validate(
                data['job_id'], 
                InputType.UUID, 
                field_name="job_id"
            )
        
        # Priority validation
        if 'priority' in data:
            priority = data['priority']
            if isinstance(priority, str):
                priority = int(priority)
            if not isinstance(priority, int) or priority < 0 or priority > 10:
                raise ValidationError("priority must be an integer between 0 and 10")
            validated['priority'] = priority
        
        # Wordlist validation
        if 'wordlist' in data:
            wordlist = str(data['wordlist']).strip()
            if not self.is_safe_path(wordlist):
                raise ValidationError("Invalid wordlist path")
            validated['wordlist'] = wordlist
        
        # Mask validation
        if 'mask' in data:
            mask = str(data['mask']).strip()
            # Basic mask pattern validation
            if not re.match(r'^[\?a-zA-Z0-9\-_\.\s]+$', mask):
                raise ValidationError("Invalid mask pattern")
            if len(mask) > 100:
                raise ValidationError("Mask too long (max 100 characters)")
            validated['mask'] = mask
        
        # Rules validation
        if 'rules' in data:
            rules = str(data['rules']).strip()
            if not self.is_safe_path(rules):
                raise ValidationError("Invalid rules path")
            validated['rules'] = rules
        
        return validated
    
    def validate_api_request(self, data: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Validate API request data based on endpoint."""
        validated = {}
        
        # Common validations for all endpoints
        for key, value in data.items():
            if isinstance(value, str):
                # Check for dangerous patterns
                for pattern, threat_type in self.dangerous_patterns:
                    if pattern.search(value):
                        logger.warning(f"Dangerous pattern detected in {key}: {threat_type}")
                        raise ValidationError(f"Invalid content in {key}")
        
        # Endpoint-specific validation
        if endpoint.startswith('/api/worker/'):
            validated.update(self.validate_worker_data(data))
        elif endpoint.startswith('/api/hash/') or endpoint.startswith('/api/job/'):
            validated.update(self.validate_hash_data(data))
            validated.update(self.validate_job_data(data))
        else:
            # Generic validation for other endpoints
            for key, value in data.items():
                if isinstance(value, str):
                    validated[key] = self.sanitize_string(value)
                else:
                    validated[key] = value
        
        return validated
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics (placeholder for future metrics)."""
        return {
            'supported_input_types': [t.value for t in InputType],
            'dangerous_patterns_count': len(self.dangerous_patterns),
            'default_rules_count': len(self.default_rules)
        }


# Global validator instance
_input_validator: Optional[InputValidator] = None


def get_input_validator() -> InputValidator:
    """Get global input validator instance."""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator


def validate_input(value: Any, input_type: InputType, **kwargs) -> Any:
    """Validate input using global validator."""
    return get_input_validator().validate(value, input_type, **kwargs)


def sanitize_input(value: str, input_type: InputType = InputType.TEXT) -> str:
    """Sanitize input using global validator."""
    return get_input_validator().sanitize_input(value, input_type)


def validate_request(validation_schema: Dict[str, Dict[str, Any]]):
    """Decorator for validating request data."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Find request data in arguments
            request_data = None
            for arg in args:
                if hasattr(arg, 'dict'):  # Pydantic model
                    request_data = arg.dict()
                    break
                elif isinstance(arg, dict):
                    request_data = arg
                    break
            
            if request_data is None:
                # Try to find in kwargs
                for value in kwargs.values():
                    if hasattr(value, 'dict'):
                        request_data = value.dict()
                        break
                    elif isinstance(value, dict):
                        request_data = value
                        break
            
            if request_data:
                try:
                    validated_data = get_input_validator().validate_request_data(
                        request_data, validation_schema
                    )
                    # Update the original data with validated values
                    if hasattr(args[0], 'dict'):
                        # Update Pydantic model
                        for key, value in validated_data.items():
                            setattr(args[0], key, value)
                except ValidationError as e:
                    raise HTTPException(status_code=400, detail=str(e))
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator