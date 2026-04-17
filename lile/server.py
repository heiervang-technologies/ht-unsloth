import asyncio
import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn

from lile.state import LileState
from lile.queue import ComputeQueue
from lile.objectives import ccpd_v2_loss, compute_sft_loss

app = FastAPI(title="LiveLearn (lile) Daemon")
queue = ComputeQueue()
state = None

class FeedbackRequest(BaseModel):
    prompt: str
    response_neg: str
    critique: str | None = None
    response_pos: str | None = None
    kind: str = "nl_critique"

class InferenceRequest(BaseModel):
    prompt: str
    wait_for_commit: int = 0
    max_new_tokens: int = 128
    temperature: float = 0.7

class MergeRequest(BaseModel):
    pass

@app.on_event("startup")
async def startup_event():
    global state
    print("Loading LileState...")
    state = LileState()
    # Start the worker loop
    asyncio.create_task(worker_loop())
    print("Daemon ready.")

async def worker_loop():
    optimizer = torch.optim.AdamW(state.model.parameters(), lr=1e-4)
    
    while True:
        job_type, job = await queue.get_next_job()
        
        if job_type == "inference":
            req = job.request
            messages = [{"role": "user", "content": req.prompt}]
            prompt_text = state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = state.tokenizer(prompt_text, return_tensors="pt").input_ids.to(state.model.device)
            
            with torch.no_grad():
                outputs = state.model.generate(
                    input_ids,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    do_sample=True,
                    pad_token_id=state.tokenizer.eos_token_id,
                )
            gen_ids = outputs[0][input_ids.shape[1]:]
            text = state.tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            job.future.set_result({"response": text, "commit_step": queue.current_step})
            
        elif job_type == "training":
            batch = job.batch
            if batch["type"] == "merge":
                state.progressive_merge()
            elif batch["type"] == "feedback":
                req = batch["data"]
                optimizer.zero_grad()
                if req.kind == "nl_critique":
                    loss = ccpd_v2_loss(
                        state.model, state.tokenizer, req.prompt, req.response_neg, req.critique,
                        k=4, tau=0.0
                    )
                elif req.kind == "preferred":
                    loss = compute_sft_loss(state.model, state.tokenizer, req.prompt, req.response_pos)
                else:
                    loss = None
                    
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    
            queue.mark_training_done(job.commit_token)

@app.post("/v1/inference")
async def inference(req: InferenceRequest):
    result = await queue.enqueue_inference(req, wait_for_commit=req.wait_for_commit)
    return result

@app.post("/v1/feedback")
async def feedback(req: FeedbackRequest):
    token = queue.enqueue_training({"type": "feedback", "data": req})
    return {"status": "enqueued", "commit_token": token}

@app.post("/v1/state/merge")
async def merge(req: MergeRequest):
    token = queue.enqueue_training({"type": "merge"})
    return {"status": "enqueued", "commit_token": token}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
