from __future__ import annotations
from typing import Any
try:
    from pydantic import BaseModel, validator
except Exception:  # pragma: no cover - stubs
    class BaseModel:
        pass
    def validator(*a, **k):  # type: ignore
        def wrapper(f):
            return f
        return wrapper

class LoginRequest(BaseModel):
    passkey: str
    username: str | None = None
    password: str | None = None

class LogoutRequest(BaseModel):
    token: str

class RegisterWorkerRequest(BaseModel):
    worker_id: str
    timestamp: int
    signature: str
    pubkey: str
    mode: str = "eco"
    provider: str = "on-prem"
    hardware: dict = {}

class WorkerStatusRequest(BaseModel):
    name: str
    status: str
    timestamp: int
    signature: str
    temps: list[int] | None = None
    power: list[float] | None = None
    utilization: list[int] | None = None
    progress: dict | None = None

class SubmitHashrateRequest(BaseModel):
    worker_id: str
    gpu_uuid: str | None = None
    hashrate: float
    timestamp: int
    signature: str

class SubmitBenchmarkRequest(BaseModel):
    worker_id: str
    gpu_uuid: str
    engine: str
    hashrates: dict[str, float]
    timestamp: int
    signature: str

class FlashResult(BaseModel):
    worker_id: str
    gpu_uuid: str
    success: bool
    timestamp: int
    signature: str

class SubmitFoundsRequest(BaseModel):
    worker_id: str
    batch_id: str
    founds: list[str]
    timestamp: int
    signature: str
    job_id: str | None = None
    msg_id: str | None = None

class SubmitNoFoundsRequest(BaseModel):
    worker_id: str
    batch_id: str
    timestamp: int
    signature: str
    job_id: str | None = None
    msg_id: str | None = None

class TrainMarkovRequest(BaseModel):
    lang: str = "english"
    directory: str | None = None

class TrainLLMRequest(BaseModel):
    dataset: str
    base_model: str
    epochs: int
    learning_rate: float
    output_dir: str

class ApiKeyRequest(BaseModel):
    api_key: str

class AlgoRequest(BaseModel):
    algorithms: list[str]

class AlgoParamsRequest(BaseModel):
    algo: str
    params: dict[str, Any]

    @validator("algo")
    def _algo_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("algorithm required")
        return v

class HashesSettingsRequest(BaseModel):
    hashes_poll_interval: int | None = None
    algo_params: dict[str, dict[str, Any]] | None = None

    @validator("hashes_poll_interval")
    def _positive_interval(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("poll interval must be positive")
        return v

class JobPriorityRequest(BaseModel):
    job_id: str
    priority: int

class JobConfigRequest(BaseModel):
    job_id: str
    priority: int

class MaskListRequest(BaseModel):
    masks: list[str]

class ProbOrderRequest(BaseModel):
    enabled: bool

class InverseOrderRequest(BaseModel):
    enabled: bool

class MarkovLangRequest(BaseModel):
    lang: str

