import asyncio
import redis
from hashmancer.utils.event_logger import log_error
from ... import orchestrator_agent


async def dispatch_loop() -> None:
    """Periodically dispatch queued batches to workers."""
    while True:
        try:
            orchestrator_agent.dispatch_batches()
        except redis.exceptions.RedisError as e:
            log_error("server", "system", "SRED", "Redis unavailable", e)
        except Exception as e:  # pragma: no cover - network errors rarely tested
            log_error("server", "system", "S743", "Dispatch loop failed", e)
        await asyncio.sleep(5)
