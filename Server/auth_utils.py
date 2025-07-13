import base64
import redis
import logging
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

r = redis.Redis(host="localhost", port=6379, decode_responses=True)


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
