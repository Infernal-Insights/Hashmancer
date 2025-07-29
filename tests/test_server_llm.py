import sys, os
import asyncio
import sys
import os
import types

from tests.test_helpers import (
    fastapi_stub,
    cors_stub,
    resp_stub,

    install_stubs,
    FakeApp,
)

install_stubs()



import hashmancer.server.main as main


def test_train_llm_invokes_helper(monkeypatch, tmp_path):
    called = {}
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
    resp = asyncio.run(main.train_llm(req()))
    assert resp['status'] == 'ok'
    assert called['dataset'] == tmp_path / 'data.txt'
    assert called['model'] == 'modelA'
    assert called['epochs'] == 3
    assert called['lr'] == 0.001
    assert called['out'] == tmp_path / 'out'
