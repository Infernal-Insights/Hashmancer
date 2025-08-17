import asyncio
import json
import socket
from hashmancer.utils.event_logger import log_error
from ..config import CONFIG, BROADCAST_PORT, BROADCAST_INTERVAL


async def broadcast_presence() -> None:
    """Periodically broadcast the server URL over UDP."""
    base = CONFIG.get("server_url", "http://localhost")
    port = CONFIG.get("server_port", 8000)
    url = f"{base}:{port}"
    payload = json.dumps({"server_url": url}).encode()
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                s.sendto(payload, ("255.255.255.255", BROADCAST_PORT))
        except Exception as e:  # pragma: no cover - network errors rarely tested
            log_error("server", "system", "S716", "Broadcast failed", e)
        await asyncio.sleep(BROADCAST_INTERVAL)
