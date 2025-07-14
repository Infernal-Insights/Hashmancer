import os
import time
import threading
import redis

STREAM = os.getenv("JOBS_STREAM", "jobs")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


class GPUSidecar(threading.Thread):
    """Background thread that consumes tasks for a single GPU."""

    def __init__(self, worker_id: str, gpu: dict):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.gpu = gpu
        self.consumer = f"{worker_id}-{gpu['uuid'][:8]}"
        self.running = True
        try:
            r.xgroup_create(STREAM, worker_id, id='$', mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def run(self):
        while self.running:
            try:
                messages = r.xreadgroup(self.worker_id, self.consumer, {STREAM: '>'}, count=1, block=5000)
                if not messages:
                    continue
                for _stream, entries in messages:
                    for msg_id, data in entries:
                        job_id = data.get('job_id')
                        if not job_id:
                            r.xack(STREAM, self.worker_id, msg_id)
                            continue
                        self.execute_job(job_id)
                        r.xack(STREAM, self.worker_id, msg_id)
            except Exception as e:
                print(f"Sidecar error on {self.gpu['uuid']}: {e}")
                time.sleep(5)

    def execute_job(self, job_id: str):
        """Simulate GPU work and mark the job done."""
        print(f"GPU {self.gpu['uuid']} processing {job_id}")
        time.sleep(1)
        r.hset(
            f"job:{job_id}",
            mapping={
                "status": "done",
                "worker": self.worker_id,
                "gpu": self.gpu['uuid'],
            },
        )
        r.rpush(f"cracked:{job_id}", "dummy:result")
