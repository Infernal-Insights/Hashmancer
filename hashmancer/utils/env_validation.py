"""Environment variable validation utilities."""

import os
import logging
from typing import Any, Dict, Optional, Union, Callable
from pathlib import Path


class EnvValidationError(Exception):
    """Raised when environment variable validation fails."""
    pass


def validate_int(value: str, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Validate and convert string to integer."""
    try:
        int_val = int(value)
        if min_val is not None and int_val < min_val:
            raise ValueError(f"Value {int_val} is less than minimum {min_val}")
        if max_val is not None and int_val > max_val:
            raise ValueError(f"Value {int_val} is greater than maximum {max_val}")
        return int_val
    except ValueError as e:
        raise EnvValidationError(f"Invalid integer value '{value}': {e}")


def validate_float(value: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Validate and convert string to float."""
    try:
        float_val = float(value)
        if min_val is not None and float_val < min_val:
            raise ValueError(f"Value {float_val} is less than minimum {min_val}")
        if max_val is not None and float_val > max_val:
            raise ValueError(f"Value {float_val} is greater than maximum {max_val}")
        return float_val
    except ValueError as e:
        raise EnvValidationError(f"Invalid float value '{value}': {e}")


def validate_bool(value: str) -> bool:
    """Validate and convert string to boolean."""
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    elif value.lower() in {"0", "false", "no", "off"}:
        return False
    else:
        raise EnvValidationError(f"Invalid boolean value '{value}'. Use: true/false, yes/no, 1/0, on/off")


def validate_path(value: str, must_exist: bool = False, must_be_file: bool = False, must_be_dir: bool = False) -> Path:
    """Validate path string."""
    try:
        path = Path(value).resolve()
        
        if must_exist and not path.exists():
            raise EnvValidationError(f"Path does not exist: {path}")
        
        if must_be_file and path.exists() and not path.is_file():
            raise EnvValidationError(f"Path is not a file: {path}")
        
        if must_be_dir and path.exists() and not path.is_dir():
            raise EnvValidationError(f"Path is not a directory: {path}")
        
        return path
    except Exception as e:
        raise EnvValidationError(f"Invalid path '{value}': {e}")


def validate_url(value: str) -> str:
    """Basic URL validation."""
    if not (value.startswith("http://") or value.startswith("https://")):
        raise EnvValidationError(f"Invalid URL '{value}': must start with http:// or https://")
    return value


def validate_choice(value: str, choices: list[str], case_sensitive: bool = True) -> str:
    """Validate value is one of the allowed choices."""
    if case_sensitive:
        if value not in choices:
            raise EnvValidationError(f"Invalid choice '{value}'. Must be one of: {', '.join(choices)}")
    else:
        if value.lower() not in [c.lower() for c in choices]:
            raise EnvValidationError(f"Invalid choice '{value}'. Must be one of: {', '.join(choices)} (case insensitive)")
    return value


class EnvValidator:
    """Environment variable validator."""
    
    def __init__(self):
        self._validations: Dict[str, Dict[str, Any]] = {}
        self._errors: list[str] = []
    
    def add_validation(
        self,
        var_name: str,
        validator: Callable[[str], Any],
        default: Any = None,
        required: bool = False,
        description: str = ""
    ) -> 'EnvValidator':
        """Add a validation rule for an environment variable."""
        self._validations[var_name] = {
            "validator": validator,
            "default": default,
            "required": required,
            "description": description
        }
        return self
    
    def validate_all(self, strict: bool = True) -> Dict[str, Any]:
        """Validate all registered environment variables."""
        results = {}
        self._errors.clear()
        
        for var_name, config in self._validations.items():
            try:
                value = os.getenv(var_name)
                
                if value is None:
                    if config["required"]:
                        error_msg = f"Required environment variable {var_name} is not set"
                        if config["description"]:
                            error_msg += f": {config['description']}"
                        self._errors.append(error_msg)
                        if strict:
                            raise EnvValidationError(error_msg)
                    else:
                        results[var_name] = config["default"]
                else:
                    results[var_name] = config["validator"](value)
                    
            except EnvValidationError as e:
                error_msg = f"Environment variable {var_name}: {e}"
                self._errors.append(error_msg)
                if strict:
                    raise EnvValidationError(error_msg)
        
        if self._errors and strict:
            raise EnvValidationError(f"Environment validation failed: {'; '.join(self._errors)}")
        
        return results
    
    def get_errors(self) -> list[str]:
        """Get list of validation errors."""
        return self._errors.copy()
    
    def log_errors(self, logger: Optional[logging.Logger] = None) -> None:
        """Log validation errors."""
        if not logger:
            logger = logging.getLogger(__name__)
        
        for error in self._errors:
            logger.error(f"Environment validation error: {error}")


# Common validation presets
def create_redis_validator() -> EnvValidator:
    """Create validator for Redis configuration."""
    return (EnvValidator()
        .add_validation("REDIS_HOST", str, default="localhost", description="Redis server hostname")
        .add_validation("REDIS_PORT", lambda x: validate_int(x, 1, 65535), default=6379, description="Redis server port")
        .add_validation("REDIS_PASSWORD", str, description="Redis server password")
        .add_validation("REDIS_SSL", validate_bool, default=False, description="Enable Redis SSL")
        .add_validation("REDIS_SSL_CERT", lambda x: validate_path(x, must_exist=True, must_be_file=True), 
                       description="Redis SSL certificate file")
        .add_validation("REDIS_SSL_KEY", lambda x: validate_path(x, must_exist=True, must_be_file=True), 
                       description="Redis SSL key file")
        .add_validation("REDIS_SSL_CA_CERT", lambda x: validate_path(x, must_exist=True, must_be_file=True), 
                       description="Redis SSL CA certificate file"))


def create_worker_validator() -> EnvValidator:
    """Create validator for worker configuration."""
    return (EnvValidator()
        .add_validation("SERVER_URL", validate_url, default="http://localhost:8000", 
                       description="Hashmancer server URL")
        .add_validation("STATUS_INTERVAL", lambda x: validate_int(x, 1, 3600), default=30,
                       description="Worker status update interval in seconds")
        .add_validation("WORKER_PIN", str, description="Worker registration PIN")
        .add_validation("HASHCAT_WORKLOAD", lambda x: validate_int(x, 1, 4), 
                       description="Hashcat workload profile (1-4)")
        .add_validation("HASHCAT_OPTIMIZED", validate_bool, default=False,
                       description="Enable hashcat optimizations")
        .add_validation("GPU_POWER_LIMIT", lambda x: validate_int(x, 50, 500),
                       description="GPU power limit in watts")
        .add_validation("GPU_POWER_TIMEOUT", lambda x: validate_float(x, 1.0, 60.0), default=5.0,
                       description="GPU power command timeout"))


def create_server_validator() -> EnvValidator:
    """Create validator for server configuration."""
    return (EnvValidator()
        .add_validation("JOB_STREAM", str, default="jobs", description="Redis job stream name")
        .add_validation("HTTP_GROUP", str, default="http-workers", description="Redis consumer group for HTTP workers")
        .add_validation("LOW_BW_JOB_STREAM", str, default="darkling-jobs", description="Redis job stream for low bandwidth workers")
        .add_validation("LOW_BW_GROUP", str, default="darkling-workers", description="Redis consumer group for low bandwidth workers")
        .add_validation("LLM_MODEL_PATH", lambda x: validate_path(x, must_exist=True), 
                       description="Path to LLM model file"))