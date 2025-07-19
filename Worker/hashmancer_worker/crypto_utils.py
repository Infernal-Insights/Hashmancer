"""Utility helpers for signing messages from the worker.

The worker keeps the private key loaded at module import time so that each
signing operation doesn't repeatedly read the key file.  The key object is
immutable in the ``cryptography`` library, so sharing a single instance across
threads is safe as long as the key is not modified.
"""

import os
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH", "./worker_private_key.pem")
PUBLIC_KEY_PATH = os.getenv("PUBLIC_KEY_PATH", "./worker_public_key.pem")

# Load the private key once so repeated signing doesn't hit the filesystem.
# The returned key object from the cryptography library is immutable and safe
# to use across threads for sign operations.


def load_private_key():
    with open(PRIVATE_KEY_PATH, "rb") as f:
        key_data = f.read()
    return serialization.load_pem_private_key(key_data, password=None)


# Cache the private key at import time for reuse.
try:
    _PRIVATE_KEY = load_private_key()
except FileNotFoundError:
    # Allow import in environments without the private key. The key will be
    # loaded on first use of ``sign_message``.
    _PRIVATE_KEY = None


def load_public_key_pem() -> str:
    with open(PUBLIC_KEY_PATH, "r") as f:
        return f.read()


def sign_message(message: str) -> str:
    """Return a base64 signature for the provided message."""
    key = _PRIVATE_KEY
    if key is None:
        key = load_private_key()
    signature = key.sign(
        message.encode(), padding.PKCS1v15(), hashes.SHA256()
    )
    return base64.b64encode(signature).decode()
