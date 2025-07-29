from __future__ import annotations
import asyncio
from typing import Iterable, List

from ..config import BROADCAST_ENABLED

from .broadcast import broadcast_presence
from .hashes_jobs import poll_hashes_jobs, process_hashes_jobs
from .hashes_jobs import fetch_and_store_jobs  # re-export for convenience
from .dispatch import dispatch_loop
from .watchdog import watchdog_loop

__all__ = [
    "broadcast_presence",
    "fetch_and_store_jobs",
    "poll_hashes_jobs",
    "process_hashes_jobs",
    "dispatch_loop",
    "watchdog_loop",
    "start_loops",
    "stop_loops",
]


def start_loops() -> List[asyncio.Task]:
    """Start all background loops and return the created tasks."""
    try:
        from hashmancer.server import main  # type: ignore
        broadcast_enabled = getattr(main, "BROADCAST_ENABLED", BROADCAST_ENABLED)
    except Exception:  # pragma: no cover - import fallback
        broadcast_enabled = BROADCAST_ENABLED

    tasks: List[asyncio.Task] = []
    if broadcast_enabled:
        tasks.append(asyncio.create_task(broadcast_presence()))
    tasks.append(asyncio.create_task(poll_hashes_jobs()))
    tasks.append(asyncio.create_task(process_hashes_jobs()))
    tasks.append(asyncio.create_task(dispatch_loop()))
    tasks.append(asyncio.create_task(watchdog_loop()))
    return tasks


async def stop_loops(tasks: Iterable[asyncio.Task]) -> None:
    """Cancel running tasks and wait for them to finish."""
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
