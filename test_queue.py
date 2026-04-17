import asyncio
import pytest
from lile.queue import ComputeQueue

@pytest.mark.asyncio
async def test_commit_cursor_invariant():
    queue = ComputeQueue()
    executed_order = []

    async def worker_loop():
        while True:
            job_type, job = await queue.get_next_job()
            if job_type == "inference":
                executed_order.append(f"inf:{job.request}")
                job.future.set_result(f"done_{job.request}")
            elif job_type == "training":
                executed_order.append(f"train:{job.batch}")
                # Simulate GPU work
                await asyncio.sleep(0.01)
                queue.mark_training_done(job.commit_token)
                
            if len(executed_order) == 4:
                break

    # Client code
    # 1. Enqueue training
    token1 = queue.enqueue_training("batch1")
    token2 = queue.enqueue_training("batch2")
    
    # 2. Enqueue inference that MUST wait for batch2
    inf_task = asyncio.create_task(queue.enqueue_inference("req1", wait_for_commit=token2))
    
    # 3. Enqueue inference that DOES NOT wait
    inf_task2 = asyncio.create_task(queue.enqueue_inference("req2", wait_for_commit=0))

    worker_task = asyncio.create_task(worker_loop())

    await worker_task
    
    # req2 has no dependencies, so it should run FIRST
    # train1 runs, then train2, then req1
    
    assert executed_order == [
        "inf:req2",
        "train:batch1",
        "train:batch2",
        "inf:req1"
    ], f"Invariant violated: {executed_order}"
    print("Commit cursor invariant test passed.")

if __name__ == "__main__":
    asyncio.run(test_commit_cursor_invariant())
