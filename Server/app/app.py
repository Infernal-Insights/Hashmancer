from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ascii_logo import print_logo
from .config import CONFIG, PORTAL_KEY

app = FastAPI()

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
                    try:
                        from main import verify_session_token
                    except Exception:  # pragma: no cover - package import
                        from Server.main import verify_session_token

                    if not token or not verify_session_token(token):
                        response = HTMLResponse("Unauthorized", status_code=401)
                        await response(scope, receive, send)
                        return
        await self.app(scope, receive, send)

app.add_middleware(PortalAuthMiddleware, key=PORTAL_KEY)

@app.on_event("startup")
async def _show_logo() -> None:
    print_logo()
