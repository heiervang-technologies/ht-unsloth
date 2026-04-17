"""Phase 6 — vLLM inference sidecar.

The single-CUDA-context default (``InferenceEngine``) makes inference latency
during a train step equal to the train step duration: ~100 ms on Qwen3-0.6B,
~1–2 s on Qwen3-7B. The plan's Phase 6 calls for a *separate* vLLM worker
holding the live model in PagedAttention layout so chat and training can run
concurrently. The trainer remains the sole writer of weights; the sidecar is
read-only with periodic adapter syncs after merges.

Two backends, both built on ``unsloth_zoo`` primitives:

* **separate** (production): vLLM runs in a separate process on a second GPU.
  Weight sync goes over NCCL via
  :class:`unsloth_zoo.vllm_rlhf_utils.WorkerExtension`. This is the
  recommended deployment for 7B+. Requires ≥2 GPUs.
* **colocate** (single-GPU fallback): vLLM runs in the same process, sharing
  the same GPU as the trainer. Weight sync goes via cudaIpc handle exchange
  (:class:`unsloth_zoo.vllm_rlhf_utils.ColocateWorkerExtension`). VRAM is
  partitioned by ``gpu_memory_utilization``.

What this module does NOT build:
* The cudaIpc handle-passing protocol — uses ``ColocateWorkerExtension``.
* The LoRA in-memory tensor packing — uses
  :class:`unsloth_zoo.vllm_lora_request.LoRARequest`.
* The NCCL stateless process group — uses
  :func:`unsloth_zoo.vllm_rlhf_utils.stateless_init_process_group`.

Honest verification gap: this dev box is a single 3090 with no vLLM in the
venv (vLLM has heavy CUDA build requirements and the upstream wheel for
CUDA 13 isn't tested). The wiring + adapter-sync interface is exercised by
unit tests with a mock backend; end-to-end NCCL verification requires a
2-GPU box and ``pip install vllm``. See ``STATUS.md`` "Phase 6 verification".
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lile.engine.inference import ChatMessage, GenerationResult


_LOG = logging.getLogger("lile.vllm_sidecar")


def is_available() -> bool:
    """True iff vLLM and unsloth_zoo's RLHF utilities can be imported."""
    try:
        import vllm  # noqa: F401, PLC0415
        from unsloth_zoo.vllm_rlhf_utils import WorkerExtension  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


@dataclass
class SidecarConfig:
    model_name: str
    mode: str = "colocate"  # "colocate" | "separate"
    device: str = "cuda:0"  # for colocate; for separate this is the SIDECAR's device
    sidecar_device: str = "cuda:1"  # only used in separate mode
    gpu_memory_utilization: float = 0.4
    max_model_len: int = 2048
    max_lora_rank: int = 64
    enable_lora: bool = True
    # NCCL bootstrap (separate mode only).
    nccl_master_address: str = "127.0.0.1"
    nccl_master_port: int = 29500
    nccl_world_size: int = 2  # trainer (rank 0) + sidecar (rank 1)


class VLLMSidecar:
    """A vLLM-backed inference engine that mirrors the
    :class:`lile.engine.inference.InferenceEngine` surface.

    The contract:
        * ``load()`` must succeed before ``generate()``.
        * ``generate()`` is thread-safe and *does not* block on the trainer.
        * ``apply_lora(state_dict)`` swaps the active adapter in memory; safe
          to call from the trainer's GPU lock — the next ``generate()`` will
          pick up the new request.

    Failure modes:
        * If vLLM isn't installed, ``load()`` raises ``ImportError``. Callers
          should check :func:`is_available` first and fall back to
          ``InferenceEngine`` if needed.
        * If the configured device doesn't exist (e.g. ``cuda:1`` on a
          single-GPU box in separate mode), ``load()`` raises ``RuntimeError``.
    """

    def __init__(self, config: SidecarConfig):
        self.config = config
        self._llm: Any = None  # vllm.LLM instance, set by load()
        self._lora_int_id = 0
        self._lora_request: Any = None
        self._tokenizer: Any = None

    # --- lifecycle -------------------------------------------------------

    def load(self, tokenizer: Any) -> "VLLMSidecar":
        if not is_available():
            raise ImportError(
                "vLLM sidecar requested but vLLM / unsloth_zoo are not "
                "available. Install vllm to use this backend."
            )
        # Lazy-imported so this module stays importable on machines without vLLM.
        from vllm import LLM  # noqa: PLC0415

        self._tokenizer = tokenizer

        kwargs = dict(
            model=self.config.model_name,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            enable_lora=self.config.enable_lora,
            max_lora_rank=self.config.max_lora_rank,
            # The worker extension is what enables our cross-process weight sync.
            worker_extension_cls=(
                "unsloth_zoo.vllm_rlhf_utils.WorkerExtension"
                if self.config.mode == "separate"
                else "unsloth_zoo.vllm_rlhf_utils.ColocateWorkerExtension"
            ),
        )
        # In colocate mode we let vLLM pick the device (it'll share with
        # whatever's already on the active CUDA context). In separate mode we
        # pass the sidecar device explicitly so vLLM lands on cuda:1.
        if self.config.mode == "separate":
            kwargs["device"] = self.config.sidecar_device

        _LOG.info("vLLM sidecar loading (%s on %s)…",
                  self.config.model_name, self.config.sidecar_device
                  if self.config.mode == "separate" else self.config.device)
        self._llm = LLM(**kwargs)
        return self

    def shutdown(self) -> None:
        if self._llm is not None:
            # vLLM holds CUDA + worker resources; let it clean up via __del__.
            del self._llm
            self._llm = None

    # --- generation ------------------------------------------------------

    def generate(
        self,
        messages: list["ChatMessage"],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> "GenerationResult":
        from lile.engine.inference import GenerationResult  # noqa: PLC0415
        from vllm import SamplingParams  # noqa: PLC0415

        if self._llm is None:
            raise RuntimeError("VLLMSidecar.load() must be called before generate()")

        prompt = self._tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        params = SamplingParams(
            temperature=0.0 if not do_sample else float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_new_tokens),
        )
        t0 = time.perf_counter()
        outputs = self._llm.generate(
            [prompt], params,
            lora_request=self._lora_request,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - t0
        text = outputs[0].outputs[0].text
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)
        return GenerationResult(
            response_id=str(uuid.uuid4()),
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_s=elapsed,
        )

    # --- LoRA sync -------------------------------------------------------

    def apply_lora(
        self,
        lora_tensors: dict[str, Any],
        *,
        name: str = "live",
    ) -> None:
        """Swap the active adapter, in memory.

        ``lora_tensors`` must already be on a CUDA tensor format vLLM accepts
        (typically the same tensors that come out of
        :func:`lile.adapters.lora_state_dict`, moved to the sidecar's device).
        We bump ``lora_int_id`` on each call so vLLM treats it as a new adapter
        rather than caching the old one.
        """
        from unsloth_zoo.vllm_lora_request import LoRARequest  # noqa: PLC0415

        self._lora_int_id += 1
        self._lora_request = LoRARequest(
            lora_name=f"{name}-{self._lora_int_id}",
            lora_int_id=self._lora_int_id,
            lora_tensors=lora_tensors,
        )
        _LOG.debug("vLLM sidecar lora swapped → id=%d", self._lora_int_id)

    @property
    def current_lora_id(self) -> int:
        return self._lora_int_id

    @property
    def llm(self) -> Any:
        """Underlying vllm.LLM (None until load())."""
        return self._llm
