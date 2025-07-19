"""Helpers for signing server responses.

The server loads the private key once at import time.  The key object returned
by ``cryptography`` is immutable and safe to share between threads for signing
operations.
"""

import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KEY_PATH = "./keys/private_key.pem"


def load_private_key():
    with open(KEY_PATH, "rb") as f:
        key_data = f.read()
    return serialization.load_pem_private_key(key_data, password=None)


# Cache the server private key at import time.
try:
    _PRIVATE_KEY = load_private_key()
except FileNotFoundError:
    # Importing the module shouldn't fail in environments without the key.
    # The key will be loaded on first call to ``sign_message``.
    _PRIVATE_KEY = None


def sign_message(message: str) -> str:
    """Return a base64 signature for ``message`` using the cached key."""
    key = _PRIVATE_KEY
    if key is None:
        key = load_private_key()
    signature = key.sign(
        message.encode(), padding.PKCS1v15(), hashes.SHA256()
    )
    return base64.b64encode(signature).decode()
