import sys
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SERVER_PATH = os.path.join(REPO_ROOT, "Server")


def _ensure_paths():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    if SERVER_PATH not in sys.path:
        sys.path.insert(0, SERVER_PATH)


_ensure_paths()


@pytest.fixture(scope="session", autouse=True)
def add_repo_to_path():
    """Ensure repository paths are on sys.path once per session."""
    _ensure_paths()
    yield
