from __future__ import annotations

import asyncio
import pytest

from lile.queue import ComputeQueue
from lile.errors import QueueFullError

pytestmark = pytest.mark.cpu_only

async def dummy_handler(task):
    await asyncio.sleep(0.05)
    return task.token

def test_try_submit_queue_full() -> None:
    q = ComputeQueue(max_depth=2)

    async def run():
        await q.start(dummy_handler)
        
        if q._worker_task:
            q._hard_stop.set()
            q._stop.set()
            await q._worker_task
            
        q._accepting = True
        
        await q.try_submit("train", {"a": 1})
        await q.try_submit("train", {"a": 2})
        
        with pytest.raises(QueueFullError):
            await q.try_submit("train", {"a": 3})

    asyncio.run(run())
