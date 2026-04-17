# SPDX-License-Identifier: AGPL-3.0-only
# HT fork: Prompt baking integration using marksverdhei/bakery

"""Self-contained prompt baking training pipeline.

Uses bakery.PromptBakingTrainer to distill a system prompt into LoRA weights
via KL divergence.  Follows the same dispatch pattern as _run_embedding_training
in worker.py: a self-contained function that handles model loading, dataset
construction, training, and saving.
"""

import json
import logging
import math
import os
import queue as _queue
import threading
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)


def _send_status(event_queue: Any, message: str) -> None:
    event_queue.put({"type": "status", "message": message, "ts": time.time()})


def run_prompt_baking(event_queue: Any, stop_queue: Any, config: dict) -> None:
    """Run prompt baking training end-to-end.

    Mirrors the embedding fast-path pattern in worker.py:
    1. Import libraries
    2. Load model via FastLanguageModel
    3. Apply LoRA adapters
    4. Build dataset from the training data
    5. Construct BakeryConfig + PromptBakingTrainer
    6. Train with progress callbacks
    7. Save adapter weights
    """
    training_start_time = time.time()
    model_name = config["model_name"]
    system_prompt = config.get("baking_system_prompt", "")

    if not system_prompt or not system_prompt.strip():
        event_queue.put({
            "type": "error",
            "error": "No system prompt provided for prompt baking.",
            "stack": "",
            "ts": time.time(),
        })
        return

    # ── 1. Import libraries ──
    _send_status(event_queue, "Importing prompt baking libraries...")
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from bakery import (
            BakeryConfig,
            PromptBakingTrainer,
            create_dataset,
            prompt_baking_collator,
        )
        from datasets import load_dataset
        from transformers import TrainerCallback
        from utils.paths import datasets_root, resolve_output_dir
    except ImportError as e:
        event_queue.put({
            "type": "error",
            "error": f"Failed to import prompt baking libraries: {e}. "
            "Ensure 'bakery' is installed (pip install bakery-llm).",
            "stack": traceback.format_exc(limit=20),
            "ts": time.time(),
        })
        return

    # ── Stop signal handling ──
    _should_stop = False
    _save_on_stop = True

    def _poll_stop():
        nonlocal _should_stop, _save_on_stop
        while True:
            try:
                msg = stop_queue.get(timeout=1.0)
                if msg and msg.get("type") == "stop":
                    _save_on_stop = msg.get("save", True)
                    _should_stop = True
                    logger.info("Prompt baking: stop signal received (save=%s)", _save_on_stop)
                    return
            except _queue.Empty:
                continue
            except (EOFError, OSError):
                return

    stop_thread = threading.Thread(target=_poll_stop, daemon=True)
    stop_thread.start()

    # ── 2. Load model via FastLanguageModel ──
    _send_status(event_queue, "Loading model...")
    try:
        hf_token = config.get("hf_token", "")
        hf_token = hf_token if hf_token and hf_token.strip() else None
        max_seq_length = config.get("max_seq_length", 2048)
        load_in_4bit = config.get("load_in_4bit", True)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            token=hf_token,
            trust_remote_code=config.get("trust_remote_code", False),
        )
    except Exception as e:
        event_queue.put({
            "type": "error",
            "error": f"Failed to load model '{model_name}': {e}",
            "stack": traceback.format_exc(limit=20),
            "ts": time.time(),
        })
        return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 3. Apply LoRA adapters ──
    _send_status(event_queue, "Configuring LoRA adapters for prompt baking...")
    try:
        gradient_checkpointing = config.get("gradient_checkpointing", "unsloth")
        if gradient_checkpointing in ("none", "", None):
            gradient_checkpointing = False

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.get("lora_r", 64),
            target_modules=config.get("target_modules") or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=config.get("lora_alpha", 128),
            lora_dropout=0,  # Unsloth requires dropout=0
            bias="none",
            use_gradient_checkpointing=gradient_checkpointing,
            random_state=config.get("random_seed", 3407),
            use_rslora=config.get("use_rslora", False),
            loftq_config={"loftq_bits": 4, "loftq_iter": 1}
            if config.get("use_loftq")
            else None,
        )
    except Exception as e:
        event_queue.put({
            "type": "error",
            "error": f"Failed to configure LoRA adapters: {e}",
            "stack": traceback.format_exc(limit=20),
            "ts": time.time(),
        })
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 4. Build dataset ──
    _send_status(event_queue, "Loading dataset...")
    try:
        hf_dataset = config.get("hf_dataset", "")
        local_datasets = config.get("local_datasets") or []

        prompts = []
        responses = None

        if hf_dataset and hf_dataset.strip():
            # Load from HuggingFace — bakery can parse standard chat formats
            subset = config.get("subset") or None
            train_split = config.get("train_split", "train") or "train"
            hf_token_ds = config.get("hf_token", "")
            hf_token_ds = hf_token_ds if hf_token_ds and hf_token_ds.strip() else None
            raw_ds = load_dataset(
                hf_dataset.strip(), subset, split=train_split, token=hf_token_ds,
            )
            # Apply dataset slicing if requested
            slice_start = config.get("dataset_slice_start")
            slice_end = config.get("dataset_slice_end")
            if slice_start is not None or slice_end is not None:
                raw_ds = raw_ds.select(range(
                    slice_start or 0,
                    slice_end if slice_end is not None else len(raw_ds),
                ))
            # Extract prompts from dataset columns
            prompts, responses = _extract_prompts_from_dataset(raw_ds)
        elif local_datasets:
            # Load from local file(s)
            from pathlib import Path
            all_files = []
            for dataset_file in local_datasets:
                file_path = (
                    dataset_file if os.path.isabs(dataset_file)
                    else os.path.join(str(datasets_root()), dataset_file)
                )
                if os.path.isdir(file_path):
                    fp = Path(file_path)
                    parquet_dir = fp / "parquet-files" if (fp / "parquet-files").exists() else fp
                    parquet_files = sorted(parquet_dir.glob("*.parquet"))
                    if parquet_files:
                        all_files.extend(str(p) for p in parquet_files)
                        continue
                    for ext in (".json", ".jsonl", ".csv"):
                        all_files.extend(str(c) for c in sorted(fp.glob(f"*{ext}")))
                else:
                    all_files.append(file_path)

            if all_files:
                first_ext = Path(all_files[0]).suffix.lower()
                loader = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}.get(first_ext, "json")
                raw_ds = load_dataset(loader, data_files=all_files, split="train")
                prompts, responses = _extract_prompts_from_dataset(raw_ds)

        if not prompts:
            event_queue.put({
                "type": "error",
                "error": "No training prompts found in the dataset. "
                "The dataset should contain user messages/prompts.",
                "stack": "",
                "ts": time.time(),
            })
            return

        use_prefill = bool(config.get("baking_use_prefill", False))
        dataset_responses = responses if use_prefill else None
        dataset = create_dataset(prompts, dataset_responses)
        logger.info("Prompt baking dataset: %d prompts, responses=%s (prefill=%s)",
                     len(prompts),
                     "yes" if dataset_responses else "on-the-fly",
                     use_prefill)
    except Exception as e:
        event_queue.put({
            "type": "error",
            "error": f"Failed to load dataset: {e}",
            "stack": traceback.format_exc(limit=20),
            "ts": time.time(),
        })
        return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 5. Build training arguments ──
    _send_status(event_queue, "Configuring prompt baking training...")
    try:
        lr_value = float(config.get("learning_rate", "1e-4"))
    except ValueError:
        event_queue.put({
            "type": "error",
            "error": f"Invalid learning rate: {config.get('learning_rate')}",
            "stack": "",
            "ts": time.time(),
        })
        return

    output_dir = config.get("output_dir")
    if not output_dir:
        output_dir = str(
            resolve_output_dir(f"{model_name.replace('/', '_')}_baked_{int(time.time())}")
        )

    num_epochs = config.get("num_epochs", 3)
    batch_size = config.get("batch_size", 4)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
    max_steps_val = config.get("max_steps", 0)
    save_steps_val = config.get("save_steps", 0)
    warmup_steps_val = config.get("warmup_steps")
    log_frequency = config.get("log_frequency", 1)

    # Prompt-baking-specific params
    num_trajectories = config.get("baking_num_trajectories", 4)
    trajectory_length = config.get("baking_trajectory_length", 128)
    temperature = config.get("baking_temperature", 1.0)
    sampling_temperature = config.get("baking_sampling_temperature", 0.8)

    baking_args = {
        "output_dir": output_dir,
        "system_prompt": system_prompt.strip(),
        "num_trajectories": num_trajectories,
        "trajectory_length": trajectory_length,
        "temperature": temperature,
        "sampling_temperature": sampling_temperature,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": lr_value,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": log_frequency,
        "report_to": ["wandb"] if config.get("enable_wandb") else "none",
        "lr_scheduler_type": config.get("lr_scheduler_type", "linear"),
        "optim": config.get("optim", "adamw_8bit"),
        "weight_decay": config.get("weight_decay", 0.001),
        "seed": config.get("random_seed", 3407),
    }

    if max_steps_val and max_steps_val > 0:
        baking_args["max_steps"] = max_steps_val
    else:
        baking_args["num_train_epochs"] = num_epochs if num_epochs > 0 else 3

    if warmup_steps_val and warmup_steps_val > 0:
        baking_args["warmup_steps"] = warmup_steps_val

    if save_steps_val and save_steps_val > 0:
        baking_args["save_steps"] = save_steps_val
        baking_args["save_strategy"] = "steps"

    baking_config = BakeryConfig(**baking_args)

    # ── 6. Calculate total steps for progress ──
    if max_steps_val and max_steps_val > 0:
        total_steps = max_steps_val
    else:
        effective_epochs = num_epochs if num_epochs > 0 else 3
        len_dataloader = math.ceil(len(dataset) / batch_size)
        steps_per_epoch = max(len_dataloader // gradient_accumulation_steps, 1)
        total_steps = steps_per_epoch * effective_epochs

    # ── 7. Progress callback ──
    class _BakingProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss_value = logs.get("loss", logs.get("train_loss"))
            current_step = state.global_step
            elapsed = time.time() - training_start_time
            eta = None
            if current_step > 0 and total_steps > 0:
                remaining = total_steps - current_step
                if remaining > 0:
                    eta = (elapsed / current_step) * remaining

            event_queue.put({
                "type": "progress",
                "step": current_step,
                "epoch": round(state.epoch, 2) if state.epoch else 0,
                "loss": loss_value,
                "learning_rate": logs.get("learning_rate"),
                "total_steps": total_steps,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "grad_norm": logs.get("grad_norm"),
                "num_tokens": getattr(state, "num_input_tokens_seen", None),
                "eval_loss": logs.get("eval_loss"),
                "status_message": "",
                "ts": time.time(),
            })

        def on_step_end(self, args, state, control, **kwargs):
            if _should_stop:
                logger.info("Prompt baking: stop at step %d", state.global_step)
                control.should_training_stop = True
                return control

    # ── 8. Create trainer and train ──
    if config.get("enable_wandb") and config.get("wandb_token"):
        os.environ["WANDB_API_KEY"] = config["wandb_token"]
        if config.get("wandb_project"):
            os.environ["WANDB_PROJECT"] = config["wandb_project"]

    _send_status(event_queue, "Starting prompt baking training...")
    try:
        trainer = PromptBakingTrainer(
            model=model,
            args=baking_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=prompt_baking_collator,
            callbacks=[_BakingProgressCallback()],
        )
        trainer.train()
    except Exception as e:
        event_queue.put({
            "type": "error",
            "error": f"Prompt baking training failed: {e}",
            "stack": traceback.format_exc(limit=20),
            "ts": time.time(),
        })
        return

    # ── 9. Save model ──
    if _should_stop and not _save_on_stop:
        event_queue.put({
            "type": "complete",
            "output_dir": None,
            "status_message": "Training cancelled",
            "ts": time.time(),
        })
        return

    _send_status(event_queue, "Saving baked model...")
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Baked model saved to %s", output_dir)
    except Exception as e:
        logger.error("Failed to save baked model: %s", e)
        event_queue.put({
            "type": "error",
            "error": f"Training completed but failed to save: {e}",
            "stack": traceback.format_exc(limit=20),
            "ts": time.time(),
        })
        return

    # ── 10. Done ──
    event_queue.put({
        "type": "complete",
        "output_dir": output_dir,
        "status_message": "Prompt baking completed",
        "ts": time.time(),
    })


def _extract_prompts_from_dataset(dataset) -> tuple[list[str], list[str] | None]:
    """Extract (prompts, optional_responses) from a HuggingFace Dataset.

    Supports multiple column naming conventions:
    - Chat format: 'messages' or 'conversations' columns
    - Prompt/response: 'prompt'/'response', 'instruction'/'output', 'question'/'answer'
    - Prompt only: 'prompt', 'instruction', 'question', 'text', 'input'
    """
    columns = set(dataset.column_names)

    # Chat format (messages column)
    for col in ("messages", "conversations"):
        if col in columns:
            prompts = []
            responses = []
            for row in dataset:
                msgs = row[col]
                if isinstance(msgs, str):
                    try:
                        msgs = json.loads(msgs)
                    except (json.JSONDecodeError, TypeError):
                        continue
                user_msg = None
                assistant_msg = None
                for m in msgs:
                    role = m.get("role", m.get("from", ""))
                    content = m.get("content", m.get("value", ""))
                    if role in ("user", "human"):
                        user_msg = content
                    elif role in ("assistant", "gpt", "bot"):
                        assistant_msg = content
                if user_msg:
                    prompts.append(user_msg)
                    if assistant_msg:
                        responses.append(assistant_msg)
            return prompts, responses if len(responses) == len(prompts) else None

    # Prompt/response pairs
    prompt_cols = ["prompt", "instruction", "question", "input", "text"]
    response_cols = ["response", "output", "answer", "completion", "target"]

    prompt_col = next((c for c in prompt_cols if c in columns), None)
    response_col = next((c for c in response_cols if c in columns), None)

    if prompt_col:
        prompts = [str(row[prompt_col]) for row in dataset if row[prompt_col]]
        responses = None
        if response_col:
            responses = [
                str(row[response_col]) for row in dataset
                if row[prompt_col] and row[response_col] is not None
            ]
        return prompts, responses

    return [], None
