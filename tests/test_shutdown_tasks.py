import sys, os
import asyncio
import sys
import os
import pytest

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,

    install_stubs,
    FakeApp,
)

install_stubs()


import hashmancer.server.main as main
sys.modules['main'] = main

@pytest.mark.asyncio
async def test_tasks_cancelled_on_shutdown(monkeypatch):
    started = set()
    cancelled = set()

    def make_stub(name):
        async def stub():
            started.add(name)
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.add(name)
                raise
        return stub

    def fake_start_loops():
        return [
            asyncio.create_task(make_stub("bcast")()),
            asyncio.create_task(make_stub("poll")()),
            asyncio.create_task(make_stub("process")()),
            asyncio.create_task(make_stub("dispatch")()),
            asyncio.create_task(make_stub("watchdog")()),
        ]

    monkeypatch.setattr(main, 'start_loops', fake_start_loops)
    monkeypatch.setattr(main, 'print_logo', lambda: None)
    monkeypatch.setattr(main, 'BROADCAST_ENABLED', True)

    main.BACKGROUND_TASKS.clear()

    before = len(main.BACKGROUND_TASKS)

    await main.start_broadcast()
    await asyncio.sleep(0)

    assert len(main.BACKGROUND_TASKS) - before == 5

    await main.shutdown_event()

    assert cancelled == {'bcast', 'poll', 'process', 'dispatch', 'watchdog'}
    assert main.BACKGROUND_TASKS == []
