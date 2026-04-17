import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any

@dataclass
class TrainingJob:
    batch: Any
    commit_token: int

@dataclass
class InferenceJob:
    request: Any
    wait_for_commit: int
    future: asyncio.Future

class ComputeQueue:
    def __init__(self):
        self.training_queue = deque()
        self.inference_queue = deque()
        self.current_step = 0
        self.next_commit_token = 1
        self._event = asyncio.Event()

    def enqueue_training(self, batch: Any) -> int:
        """Enqueues a training batch and returns a commit token."""
        token = self.next_commit_token
        self.next_commit_token += 1
        self.training_queue.append(TrainingJob(batch, token))
        self._event.set()
        return token

    async def enqueue_inference(self, request: Any, wait_for_commit: int = 0) -> Any:
        """Enqueues an inference request, waiting until at least `wait_for_commit` step is processed."""
        fut = asyncio.get_running_loop().create_future()
        self.inference_queue.append(InferenceJob(request, wait_for_commit, fut))
        self._event.set()
        return await fut

    async def get_next_job(self):
        """Returns ('inference'|'training', job) prioritising inference that is ready."""
        while True:
            # Priority 1: Ready inference jobs
            for job in self.inference_queue:
                if job.wait_for_commit <= self.current_step:
                    self.inference_queue.remove(job)
                    return ("inference", job)
            
            # Priority 2: Training jobs
            if self.training_queue:
                job = self.training_queue.popleft()
                return ("training", job)
                
            self._event.clear()
            await self._event.wait()
            
    def mark_training_done(self, token: int):
        """Called by the worker loop after a training job finishes."""
        self.current_step = max(self.current_step, token)
        self._event.set()
