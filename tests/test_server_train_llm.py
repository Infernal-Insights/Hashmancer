import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import asyncio
import sys
import os
import types

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


def test_train_llm_invokes_helper(monkeypatch, tmp_path):
    called = {}

    def fake_to_thread(func, *a, **kw):
        called['func'] = func
        called['args'] = a
        called['kw'] = kw

        async def dummy():
            called['ran'] = True
        return dummy()

    monkeypatch.setattr(main.asyncio, 'to_thread', fake_to_thread)
    orig_create = asyncio.create_task

    def fake_create(coro):
        called['task'] = coro
        return orig_create(coro)

    monkeypatch.setattr(main.asyncio, 'create_task', fake_create)

    def fake_train(dataset, model, epochs, lr, out_dir):
        called['dataset'] = dataset
        called['model'] = model
        called['epochs'] = epochs
        called['lr'] = lr
        called['out'] = out_dir

    monkeypatch.setattr(main, '_train_llm', types.SimpleNamespace(train_model=fake_train))
    req = type('Req', (), {
        'dataset': str(tmp_path / 'data.txt'),
        'base_model': 'modelA',
        'epochs': 3,
        'learning_rate': 0.001,
        'output_dir': str(tmp_path / 'out')
    })
    resp = asyncio.run(main.train_llm_endpoint(req()))
    assert resp['status'] == 'scheduled'
    assert called['func'] == fake_train
    assert called['args'] == (tmp_path / 'data.txt', 'modelA', 3, 0.001, tmp_path / 'out')
