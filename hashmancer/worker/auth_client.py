#!/usr/bin/env python3
"""
Worker Authentication Client
Automated RSA key generation and server registration
"""

import os
import time
import base64
import requests
import logging
from typing import Dict, Optional
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)

class WorkerAuthClient:
    """Handles worker-side authentication with automated key management."""
    
    def __init__(self, server_url: str, worker_name: str = None):
        self.server_url = server_url.rstrip('/')
        self.worker_name = worker_name or f"worker_{os.getpid()}"
        
        # Worker key storage
        self.key_dir = Path.home() / ".hashmancer" / "worker_keys"
        self.key_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        self.private_key_path = self.key_dir / f"{self.worker_name}_private.pem"
        self.auth_token_path = self.key_dir / f"{self.worker_name}_token.txt"
        
        # Load or generate worker keys
        self.private_key = self._load_or_generate_worker_key()
        self.public_key = self.private_key.public_key()
        
        # Authentication state
        self.auth_token = None
        self.worker_id = None
        self.server_public_key = None
        
        # Load existing auth if available
        self._load_existing_auth()
    
    def _load_or_generate_worker_key(self) -> rsa.RSAPrivateKey:
        """Load existing worker key or generate new one."""
        if self.private_key_path.exists():
            try:
                with open(self.private_key_path, "rb") as f:
                    private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None
                    )
                logger.info(f"Loaded existing worker key for {self.worker_name}")
                return private_key
            except Exception as e:
                logger.warning(f"Failed to load worker key, generating new one: {e}")
        
        # Generate new key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        
        # Save private key
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open(self.private_key_path, "wb") as f:
            f.write(pem)
        
        os.chmod(self.private_key_path, 0o600)
        
        logger.info(f"Generated new worker key for {self.worker_name}")
        return private_key
    
    def _load_existing_auth(self):
        """Load existing authentication token if available."""
        if self.auth_token_path.exists():
            try:
                with open(self.auth_token_path, "r") as f:
                    lines = f.read().strip().split('\n')
                    if len(lines) >= 2:
                        self.worker_id = lines[0]
                        self.auth_token = lines[1]
                        logger.info(f"Loaded existing auth for worker {self.worker_id}")
            except Exception as e:
                logger.warning(f"Failed to load existing auth: {e}")
    
    def _save_auth(self, worker_id: str, auth_token: str):
        """Save authentication credentials."""
        try:
            with open(self.auth_token_path, "w") as f:
                f.write(f"{worker_id}\n{auth_token}\n")
            os.chmod(self.auth_token_path, 0o600)
            
            self.worker_id = worker_id
            self.auth_token = auth_token
            
            logger.info(f"Saved auth credentials for worker {worker_id}")
        except Exception as e:
            logger.error(f"Failed to save auth credentials: {e}")
    
    def get_public_key_pem(self) -> str:
        """Get worker public key in PEM format."""
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')
    
    def sign_data(self, data: str) -> str:
        """Sign data with worker private key."""
        signature = self.private_key.sign(
            data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()
    
    def verify_server_signature(self, data: str, signature: str) -> bool:
        """Verify signature from server."""
        if not self.server_public_key:
            return False
        
        try:
            signature_bytes = base64.b64decode(signature)
            self.server_public_key.verify(
                signature_bytes,
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.warning(f"Server signature verification failed: {e}")
            return False
    
    async def register_with_server(self, worker_metadata: Dict = None) -> bool:
        """Register worker with server using automated key exchange."""
        try:
            # Step 1: Initiate registration
            logger.info("Initiating worker registration...")
            
            response = requests.post(
                f"{self.server_url}/api/worker/register/initiate",
                timeout=30
            )
            response.raise_for_status()
            
            registration_data = response.json()
            session_id = registration_data["session_id"]
            challenge = registration_data["challenge"]
            server_public_key_pem = registration_data["server_public_key"]
            
            # Store server public key
            self.server_public_key = serialization.load_pem_public_key(
                server_public_key_pem.encode()
            )
            
            # Step 2: Sign challenge
            challenge_bytes = base64.b64decode(challenge)
            signed_challenge = self.private_key.sign(
                challenge_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Step 3: Complete registration
            logger.info("Completing worker registration...")
            
            completion_data = {
                "session_id": session_id,
                "worker_public_key": self.get_public_key_pem(),
                "signed_challenge": base64.b64encode(signed_challenge).decode(),
                "worker_metadata": worker_metadata or {
                    "name": self.worker_name,
                    "platform": os.name,
                    "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
                }
            }
            
            response = requests.post(
                f"{self.server_url}/api/worker/register/complete",
                json=completion_data,
                timeout=30
            )
            response.raise_for_status()
            
            auth_result = response.json()
            worker_id = auth_result["worker_id"]
            auth_token = auth_result["auth_token"]
            
            # Save credentials
            self._save_auth(worker_id, auth_token)
            
            logger.info(f"Worker registration successful! Worker ID: {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Worker registration failed: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if worker has valid authentication."""
        return bool(self.worker_id and self.auth_token)
    
    async def authenticate_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make authenticated request to server."""
        if not self.is_authenticated():
            raise Exception("Worker not authenticated")
        
        # Add authentication headers
        headers = kwargs.get('headers', {})
        headers.update({
            'X-Worker-ID': self.worker_id,
            'X-Worker-Token': self.auth_token,
            'X-Worker-Timestamp': str(int(time.time()))
        })
        kwargs['headers'] = headers
        
        # Make request
        response = getattr(requests, method.lower())(url, **kwargs)
        
        # Handle authentication errors
        if response.status_code == 401:
            logger.warning("Authentication failed, attempting re-registration...")
            self.auth_token = None
            self.worker_id = None
            if self.auth_token_path.exists():
                os.unlink(self.auth_token_path)
            raise Exception("Authentication expired")
        
        return response
    
    async def heartbeat(self) -> bool:
        """Send heartbeat to server."""
        try:
            response = await self.authenticate_request(
                'POST',
                f"{self.server_url}/api/worker/heartbeat",
                json={"timestamp": int(time.time())},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False
    
    async def get_job(self) -> Optional[Dict]:
        """Get next job from server."""
        try:
            response = await self.authenticate_request(
                'GET',
                f"{self.server_url}/api/worker/job",
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 204:
                return None  # No jobs available
            else:
                logger.warning(f"Failed to get job: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get job: {e}")
            return None
    
    async def submit_result(self, job_id: str, result: Dict) -> bool:
        """Submit job result to server."""
        try:
            # Sign the result for integrity
            result_data = f"{job_id}:{result}"
            signature = self.sign_data(result_data)
            
            payload = {
                "job_id": job_id,
                "result": result,
                "signature": signature,
                "timestamp": int(time.time())
            }
            
            response = await self.authenticate_request(
                'POST',
                f"{self.server_url}/api/worker/result",
                json=payload,
                timeout=60
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to submit result: {e}")
            return False

def create_worker_auth_client(server_url: str, worker_name: str = None) -> WorkerAuthClient:
    """Create and return a worker authentication client."""
    return WorkerAuthClient(server_url, worker_name)