import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import asyncio
import sys
import os

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


def test_hash_algos_endpoint():
    algos = asyncio.run(main.get_hash_algos())
    assert isinstance(algos, dict)
    assert algos.get('MD5') == 0
    assert algos.get('NTLM') == 1000
