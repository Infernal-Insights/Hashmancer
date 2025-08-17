import sys
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SERVER_PATH = os.path.join(REPO_ROOT, "hashmancer", "server")


def _ensure_paths():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    if SERVER_PATH not in sys.path:
        sys.path.insert(0, SERVER_PATH)


_ensure_paths()

import types
from hashmancer.darkling import charsets

darkling_mod = types.ModuleType("darkling")
darkling_mod.charsets = charsets
sys.modules.setdefault("darkling", darkling_mod)

import importlib
orchestrator_mod = importlib.import_module("hashmancer.server.orchestrator_agent")
sys.modules.setdefault("orchestrator_agent", orchestrator_mod)

wordlist_mod = importlib.import_module("hashmancer.server.wordlist_db")
sys.modules.setdefault("wordlist_db", wordlist_mod)

utils_mod = importlib.import_module("hashmancer.utils")
sys.modules.setdefault("utils", utils_mod)

pattern_mod = importlib.import_module("hashmancer.server.pattern_to_mask")
sys.modules.setdefault("pattern_to_mask", pattern_mod)

hashescom_mod = importlib.import_module("hashmancer.server.hashescom_client")
sys.modules.setdefault("hashescom_client", hashescom_mod)

from hashmancer.darkling import statistics
darkling_mod.statistics = statistics


@pytest.fixture(scope="session", autouse=True)
def add_repo_to_path():
    """Ensure repository paths are on sys.path once per session."""
    _ensure_paths()
    yield
