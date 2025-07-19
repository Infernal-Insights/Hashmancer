import base64
import redis
from redis_utils import get_redis
import logging
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

r = get_redis()


def get_worker_pubkey(worker_id):
    key_data = r.hget(f"worker:{worker_id}", "pubkey")
    if not key_data:
        raise Exception("No public key found")
    return serialization.load_pem_public_key(key_data.encode())


def verify_signature(worker_id, payload, signature_b64):
    try:
        public_key = get_worker_pubkey(worker_id)
        signature = base64.b64decode(signature_b64)
        public_key.verify(
            signature, payload.encode(), padding.PKCS1v15(), hashes.SHA256()
        )
        return True
    except InvalidSignature:
        logging.warning(f"Signature invalid for {worker_id}")
        return False
    except Exception as e:
        logging.error(f"Verification error: {e}")
        return False


def verify_signature_with_key(pubkey_pem: str, payload: str, signature_b64: str) -> bool:
    """Verify a signature using a provided public key."""
    try:
        public_key = serialization.load_pem_public_key(pubkey_pem.encode())
        signature = base64.b64decode(signature_b64)
        public_key.verify(
            signature, payload.encode(), padding.PKCS1v15(), hashes.SHA256()
        )
        return True
    except InvalidSignature:
        logging.warning("Signature invalid for provided key")
        return False
    except Exception as e:
        logging.error(f"Verification error: {e}")
        return False
