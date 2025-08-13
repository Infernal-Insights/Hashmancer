from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from .config import CONFIG, PORTAL_KEY

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
    if AI_AVAILABLE:
        try:
            _ai_engine = await initialize_ai_engine()
            print("✅ AI Strategy Engine initialized")
        except Exception as e:
            print(f"⚠️  AI Strategy Engine failed to initialize: {e}")
    
    yield
    
    # Shutdown
    if _ai_engine:
        try:
            await shutdown_ai_engine()
            print("🛑 AI Strategy Engine shut down")
        except Exception as e:
            print(f"⚠️  Error shutting down AI engine: {e}")


app = FastAPI(lifespan=lifespan)

origins = CONFIG.get("allowed_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

app.add_middleware(PortalAuthMiddleware, key=PORTAL_KEY)
