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

class DummyRedis:
    def __init__(self):
        self.store = {}
    def scard(self, key):
        return 0
    def llen(self, key):
        return 0


def test_train_markov_invokes_processor(monkeypatch, tmp_path):
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

    def fake_process(dir_path, lang='english'):
        called['dir'] = dir_path
        called['lang'] = lang

    monkeypatch.setattr(main, 'WORDLISTS_DIR', tmp_path)
    monkeypatch.setattr(main, 'learn_trends', types.SimpleNamespace(process_wordlists=fake_process))
    req = type('Req', (), {'lang': 'french', 'directory': None})
    resp = asyncio.run(main.train_markov(req()))
    assert resp['status'] == 'scheduled'
    assert called['func'] == fake_process
    assert called['args'] == (tmp_path,)
    assert called['kw'] == {'lang': 'french'}


def test_update_probabilistic_order(monkeypatch):
    monkeypatch.setattr(main, 'CONFIG', {})
    saved = {}
    monkeypatch.setattr(main, 'save_config', lambda: saved.setdefault('done', True))
    req = type('Req', (), {'enabled': True})
    resp = asyncio.run(main.set_probabilistic_order(req()))
    assert resp['status'] == 'ok'
    assert main.CONFIG['probabilistic_order'] is True
    assert main.PROBABILISTIC_ORDER is True
    assert saved.get('done')


def test_update_inverse_prob_order(monkeypatch):
    monkeypatch.setattr(main, 'CONFIG', {})
    saved = {}
    monkeypatch.setattr(main, 'save_config', lambda: saved.setdefault('done', True))
    req = type('Req', (), {'enabled': True})
    resp = asyncio.run(main.set_inverse_prob_order(req()))
    assert resp['status'] == 'ok'
    assert main.CONFIG['inverse_prob_order'] is True
    assert main.INVERSE_PROB_ORDER is True
    assert saved.get('done')


def test_update_markov_lang(monkeypatch):
    monkeypatch.setattr(main, 'CONFIG', {})
    saved = {}
    monkeypatch.setattr(main, 'save_config', lambda: saved.setdefault('done', True))
    req = type('Req', (), {'lang': 'german'})
    resp = asyncio.run(main.set_markov_lang(req()))
    assert resp['status'] == 'ok'
    assert main.CONFIG['markov_lang'] == 'german'
    assert main.MARKOV_LANG == 'german'
    assert saved.get('done')


def test_server_status_includes_settings(monkeypatch):
    fake = DummyRedis()
    monkeypatch.setattr(main, 'r', fake)
    monkeypatch.setattr(main, 'PROBABILISTIC_ORDER', True)
    monkeypatch.setattr(main, 'INVERSE_PROB_ORDER', True)
    monkeypatch.setattr(main, 'MARKOV_LANG', 'spanish')
    monkeypatch.setattr(main, 'LLM_TRAIN_EPOCHS', 2)
    monkeypatch.setattr(main, 'LLM_TRAIN_LEARNING_RATE', 0.002)
    monkeypatch.setattr(main.orchestrator_agent, 'compute_backlog_target', lambda: 5)
    monkeypatch.setattr(main.orchestrator_agent, 'pending_count', lambda: 2)
    status = asyncio.run(main.server_status())
    assert status['probabilistic_order'] is True
    assert status['inverse_prob_order'] is True
    assert status['markov_lang'] == 'spanish'
    assert status['llm_train_epochs'] == 2
    assert status['llm_train_learning_rate'] == 0.002
