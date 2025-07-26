import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import asyncio
import sys
import os
import pytest

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,
    pydantic_stub,
    install_stubs,
    FakeApp,
)

install_stubs()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Server'))

import main

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

    monkeypatch.setattr(main, 'broadcast_presence', make_stub('bcast'))
    monkeypatch.setattr(main, 'poll_hashes_jobs', make_stub('poll'))
    monkeypatch.setattr(main, 'process_hashes_jobs', make_stub('process'))
    monkeypatch.setattr(main, 'dispatch_loop', make_stub('dispatch'))
    monkeypatch.setattr(main, 'print_logo', lambda: None)
    monkeypatch.setattr(main, 'BROADCAST_ENABLED', True)

    main.BACKGROUND_TASKS.clear()

    before = len(main.BACKGROUND_TASKS)

    await main.start_broadcast()
    await asyncio.sleep(0)

    assert len(main.BACKGROUND_TASKS) - before == 4

    await main.shutdown_event()

    assert cancelled == {'bcast', 'poll', 'process', 'dispatch'}
    assert main.BACKGROUND_TASKS == []
