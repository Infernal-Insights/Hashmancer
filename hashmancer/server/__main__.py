from .app.app import app
from .app.config import CONFIG
import uvicorn


def main() -> None:
    port = int(CONFIG.get("server_port", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
