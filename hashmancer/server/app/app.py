from __future__ import annotations
from contextlib import asynccontextmanager
import time
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from .config import CONFIG, PORTAL_KEY

# Import security components
try:
    from ..security.worker_auth import get_worker_auth_manager
    from ..security.rate_limiter import get_rate_limiter
    from ..security.input_validator import get_input_validator
    from ..security.atomic_job_manager import get_atomic_job_manager
    from ..security.key_manager import get_secure_key_manager
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logging.warning("Security components not available, running in legacy mode")

# AI Strategy Engine (optional dependency)
_ai_engine = None
try:
    from ..ai_strategy_engine import get_ai_strategy_engine, initialize_ai_engine, shutdown_ai_engine
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global _ai_engine
    
    # Startup
    print("🚀 Hashmancer Enhanced Portal starting up...")
    
    # Initialize security components
    if SECURITY_AVAILABLE:
        try:
            print("🔒 Initializing security components...")
            
            # Initialize key manager and generate server keys
            key_manager = get_secure_key_manager()
            key_manager.generate_server_key_pair()
            print("✅ Secure key management initialized")
            
            # Initialize worker authentication
            worker_auth = get_worker_auth_manager()
            print("✅ Worker authentication system initialized")
            
            # Initialize rate limiter
            rate_limiter = get_rate_limiter()
            print("✅ Rate limiting system initialized")
            
            # Initialize input validator
            input_validator = get_input_validator()
            print("✅ Input validation system initialized")
            
            # Initialize atomic job manager
            job_manager = get_atomic_job_manager()
            print("✅ Atomic job management initialized")
            
            print("🔒 All security systems initialized successfully")
            
        except Exception as e:
            print(f"⚠️  Security system initialization failed: {e}")
    
    # Temporarily disable AI engine for faster startup
    # if AI_AVAILABLE:
    #     try:
    #         _ai_engine = await initialize_ai_engine()
    #         print("✅ AI Strategy Engine initialized")
    #     except Exception as e:
    #         print(f"⚠️  AI Strategy Engine failed to initialize: {e}")
    
    yield
    
    # Shutdown
    if _ai_engine:
        try:
            await shutdown_ai_engine()
            print("🛑 AI Strategy Engine shut down")
        except Exception as e:
            print(f"⚠️  Error shutting down AI engine: {e}")
    
    print("🛑 Hashmancer server shutdown complete")


app = FastAPI(lifespan=lifespan)

# CORS middleware with secure defaults
origins = CONFIG.get("allowed_origins", ["http://localhost:3000", "http://localhost:8080"])
if "*" in origins and SECURITY_AVAILABLE:
    logging.warning("⚠️  Wildcard CORS origins detected in security mode - consider restricting")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Global security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Global security middleware with rate limiting and validation."""
    if not SECURITY_AVAILABLE:
        return await call_next(request)
    
    start_time = time.time()
    
    try:
        # Get rate limiter and check global limits
        rate_limiter = get_rate_limiter()
        
        # Check if IP is blocked
        client_ip = request.client.host if request.client else "unknown"
        if await rate_limiter.is_ip_blocked(client_ip):
            raise HTTPException(
                status_code=429,
                detail="IP address is temporarily blocked"
            )
        
        # Apply global rate limiting
        if not await rate_limiter.allow_request(
            f"global:{client_ip}", 
            max_requests=100, 
            window_seconds=60
        ):
            raise HTTPException(
                status_code=429,
                detail="Global rate limit exceeded"
            )
        
        # Continue with request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Log request (only errors and warnings to avoid spam)
        process_time = time.time() - start_time
        if response.status_code >= 400:
            logging.warning(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s from {client_ip}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Security middleware error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

class PortalAuthMiddleware:
    """Simple ASGI middleware enforcing an API key for portal routes."""

    def __init__(self, app, key: str | None):
        self.app = app
        self.key = key

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and self.key:
            path = scope.get("path", "")
            if (
                path == "/"
                or path.startswith("/portal")
                or path.startswith("/glyph")
                or path.startswith("/admin")
            ):
                headers = {
                    k.decode().lower(): v.decode()
                    for k, v in scope.get("headers", [])
                }
                if headers.get("x-api-key") != self.key:
                    cookie_header = headers.get("cookie", "")
                    token = None
                    for part in cookie_header.split(";"):
                        if part.strip().startswith("session="):
                            token = part.strip().split("=", 1)[1]
                            break
                    from hashmancer.server.auth_middleware import verify_session_token
                    from hashmancer.server.app.config import PORTAL_PASSKEY, SESSION_TTL

                    if not token or not verify_session_token(token, PORTAL_PASSKEY or "", SESSION_TTL):
                        response = HTMLResponse("Unauthorized", status_code=401)
                        await response(scope, receive, send)
                        return
        await self.app(scope, receive, send)

# Temporarily disable auth for testing
# app.add_middleware(PortalAuthMiddleware, key=PORTAL_KEY)

# Add a simple test route
@app.get("/")
async def root():
    """Test route to verify server is working."""
    return {"message": "Hashmancer Enhanced Portal is running!", "status": "active"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    health_data = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "version": "2.0.0"
    }
    
    if SECURITY_AVAILABLE:
        health_data["security"] = {
            "worker_auth": "enabled",
            "rate_limiting": "enabled", 
            "input_validation": "enabled",
            "atomic_jobs": "enabled",
            "encrypted_keys": "enabled"
        }
    else:
        health_data["security"] = "legacy_mode"
    
    return health_data

# Worker Authentication Endpoints (only available if security is enabled)
if SECURITY_AVAILABLE:
    from ..security.input_validator import ValidationError
    from typing import Dict, Any
    
    async def get_authenticated_worker(request: Request) -> Dict[str, Any]:
        """Dependency to authenticate worker requests."""
        auth_token = request.headers.get("X-Worker-Token")
        client_ip = request.client.host if request.client else "unknown"
        
        if not auth_token:
            raise HTTPException(status_code=401, detail="Missing worker authentication token")
        
        worker_auth = get_worker_auth_manager()
        worker_info = await worker_auth.authenticate_worker(auth_token, client_ip)
        
        if not worker_info:
            raise HTTPException(status_code=401, detail="Invalid worker authentication")
        
        return worker_info
    
    async def validate_request_data(request: Request) -> Dict[str, Any]:
        """Dependency to validate request data."""
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                data = await request.json()
            else:
                data = dict(request.query_params)
            
            # Validate input
            validator = get_input_validator()
            validated_data = validator.validate_api_request(data, str(request.url.path))
            
            return validated_data
            
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Input validation failed: {e}")
        except Exception as e:
            logging.error(f"Request validation error: {e}")
            raise HTTPException(status_code=400, detail="Invalid request data")
    
    @app.post("/api/worker/register/initiate")
    async def initiate_worker_registration(request: Request):
        """Initiate worker registration process."""
        try:
            # Apply rate limiting
            rate_limiter = get_rate_limiter()
            client_ip = request.client.host if request.client else "unknown"
            
            if not await rate_limiter.allow_request(
                f"worker_reg:{client_ip}", 
                max_requests=5, 
                window_seconds=300
            ):
                raise HTTPException(status_code=429, detail="Registration rate limit exceeded")
            
            worker_auth = get_worker_auth_manager()
            result = await worker_auth.initiate_worker_registration(client_ip)
            return result
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Worker registration initiation failed: {e}")
            raise HTTPException(status_code=500, detail="Registration initiation failed")
    
    @app.post("/api/worker/register/complete")
    async def complete_worker_registration(request: Request):
        """Complete worker registration with key exchange."""
        try:
            # Validate and get request data
            data = await validate_request_data(request)
            
            worker_auth = get_worker_auth_manager()
            client_ip = request.client.host if request.client else "unknown"
            
            result = await worker_auth.complete_worker_registration(
                session_id=data["session_id"],
                worker_public_key=data["worker_public_key"],
                signed_challenge=data["signed_challenge"],
                client_ip=client_ip,
                worker_metadata=data.get("worker_metadata", {})
            )
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Worker registration completion failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # Add error handlers for security-related exceptions
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        """Handle validation errors."""
        client_ip = request.client.host if request.client else "unknown"
        logging.warning(f"Validation error from {client_ip}: {exc}")
        return JSONResponse(
            status_code=400,
            content={"error": "validation_failed", "detail": str(exc)}
        )
