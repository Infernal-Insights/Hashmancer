from .app.app import app
from .app.config import CONFIG
import uvicorn
import os


def main() -> None:
    port = int(os.getenv("SERVER_PORT", CONFIG.get("server_port", 8000)))
    uvicorn.run(app, host="0.0.0.0", port=port, loop="asyncio")


if __name__ == "__main__":
    main()
