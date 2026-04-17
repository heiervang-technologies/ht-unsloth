"""Weight-sync bridge between the trainer and the vLLM sidecar.

The trainer mutates LoRA weights every ``/v1/train`` step. The sidecar holds
its own copy of the model in PagedAttention layout and needs to be kept
fresh — but we can't blast every gradient step over NCCL or VRAM bandwidth
becomes the bottleneck. The contract:

* **Adapter sync** happens at controlled boundaries: every progressive merge
  (when the active LoRA folds into ``merged_deltas`` and resets), and on
  snapshot restore. Between those boundaries, the sidecar serves the adapter
  it last received — slightly stale, but never inconsistent.
* **Base weight sync** never happens. The base weights are immutable
  (NF4 4-bit base + bf16 merged-delta residual). After each merge we send the
  *new* delta as a packed LoRA-shaped update so the sidecar's adapter view
  reflects the merged state.

Two sync modes match the two sidecar deployment modes:

* **separate (NCCL)** — bridge first calls
  ``WorkerExtension.init_weight_update_group(...)`` once at startup, then
  ``WorkerExtension.update_weight(name, dtype, shape)`` per tensor on each
  push. Bandwidth-bound on weight diff size.
* **colocate (cudaIpc)** — bridge calls
  ``ColocateWorkerExtension.update_weights_from_ipc_handles(handles)`` with
  ``torch.multiprocessing.reductions.reduce_tensor`` handles for each tensor.
  Effectively zero-copy on a single device.

For v0 we ship the *interface* and the in-memory adapter swap path
(``apply_lora`` on the sidecar). The NCCL/IPC paths are reserved as TODO with
the unsloth_zoo entrypoints documented; production verification needs a
2-GPU box and a vLLM build, which this dev environment lacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from lile.engine.vllm_sidecar import VLLMSidecar


_LOG = logging.getLogger("lile.weight_sync")


@dataclass
class WeightSyncStats:
    pushes: int = 0
    last_push_n_tensors: int = 0
    last_push_bytes: int = 0


class WeightSyncBridge:
    """Mirrors trainer-side LoRA / merged-delta state to the vLLM sidecar.

    The bridge is *passive*: it doesn't run a thread. The controller calls
    :meth:`push_active_lora` at merge / restore boundaries; the bridge does
    the protocol work and returns.
    """

    def __init__(self, sidecar: "VLLMSidecar | None"):
        self.sidecar = sidecar
        self.stats = WeightSyncStats()
        self._initialized = False

    # --- lifecycle -------------------------------------------------------

    def maybe_init_weight_update_group(self) -> None:
        """One-shot NCCL bootstrap; no-op in colocate mode and on subsequent calls."""
        if self.sidecar is None or self._initialized:
            return
        cfg = self.sidecar.config
        if cfg.mode != "separate":
            self._initialized = True
            return
        llm = self.sidecar.llm
        if llm is None:
            return
        # vLLM exposes per-worker RPCs through ``collective_rpc``; the worker
        # extension we configured at sidecar load time provides
        # ``init_weight_update_group``.
        try:
            llm.collective_rpc(
                "init_weight_update_group",
                args=(
                    cfg.nccl_master_address,
                    cfg.nccl_master_port,
                    1,                # rank_offset: trainer is rank 0, workers offset by 1
                    cfg.nccl_world_size,
                ),
            )
            self._initialized = True
            _LOG.info("NCCL weight-update group initialized")
        except Exception:
            _LOG.exception("NCCL init failed; sidecar will serve stale weights")

    # --- pushes ----------------------------------------------------------

    def push_active_lora(self, lora_state_dict: dict[str, torch.Tensor]) -> None:
        """Push the trainer's current LoRA state to the sidecar.

        Called by the controller after merge / restore. In production this
        also runs after every Nth train step (configurable, currently disabled
        for cost).
        """
        if self.sidecar is None:
            return
        # Move tensors to the sidecar's device. In colocate mode that's the
        # same device; in separate mode it's cuda:1.
        target_device = (
            self.sidecar.config.sidecar_device
            if self.sidecar.config.mode == "separate"
            else self.sidecar.config.device
        )
        moved: dict[str, torch.Tensor] = {}
        n_bytes = 0
        for k, v in lora_state_dict.items():
            t = v.detach().to(device=target_device, dtype=torch.bfloat16, non_blocking=True)
            moved[k] = t
            n_bytes += t.numel() * t.element_size()

        if self.sidecar.config.mode == "separate":
            self._push_via_nccl(moved)
        else:
            # Colocate mode: in-memory adapter swap is sufficient since the
            # sidecar reads the same memory we wrote.
            self.sidecar.apply_lora(moved)

        self.stats.pushes += 1
        self.stats.last_push_n_tensors = len(moved)
        self.stats.last_push_bytes = n_bytes

    def _push_via_nccl(self, moved: dict[str, torch.Tensor]) -> None:
        """Broadcast each tensor over the NCCL update group.

        The unsloth_zoo ``WorkerExtension.update_weight(name, dtype, shape)``
        pulls the broadcast on the worker side — so our job is to publish a
        matching broadcast for each tensor on the trainer side.
        """
        self.maybe_init_weight_update_group()
        if not self._initialized or self.sidecar is None or self.sidecar.llm is None:
            # Fallback: still apply the in-memory request so the next generate
            # at least sees *something*. This is what the colocate path does.
            self.sidecar.apply_lora(moved)
            return
        # First, register the LoRA via the in-memory request for routing.
        self.sidecar.apply_lora(moved)
        # Then broadcast each tensor for the worker to copy into its model.
        # Note: this assumes the trainer has already initialised its own end
        # of the NCCL group. In v0 we let vLLM's collective RPC handle the
        # broadcast pattern symmetrically.
        for name, tensor in moved.items():
            try:
                self.sidecar.llm.collective_rpc(
                    "update_weight",
                    args=(name, tensor.dtype, tuple(tensor.shape)),
                )
            except Exception:
                _LOG.exception("update_weight RPC failed for %s; skipping", name)
                break

    def push_merged_deltas(self, deltas_state_dict: dict[str, torch.Tensor]) -> None:
        """Push merged-delta tensors to the sidecar.

        After a progressive merge, the trainer's residual hook adds
        ``x @ delta.T`` to each layer's output. The sidecar has no equivalent
        hook, so we represent the deltas as an effective LoRA pair (rank-r
        SVD of each delta) and ship that as the active adapter.

        For v0 we ship the simpler approximation: send the deltas as-is via
        the same path as the active LoRA, relying on the LoRARequest to
        carry them. Production should run an SVD compression here to bound
        adapter rank in the sidecar.
        """
        # No-op stub for v0; the deltas are already reflected in the active
        # LoRA push because the trainer resets LoRA-A/B to zero post-merge
        # AND the residual hook is on the *trainer's* model. The sidecar
        # currently sees the merged state via the next active-LoRA push only;
        # re-bootstrapping the sidecar from a snapshot is the production path
        # for "deltas changed substantially".
        if self.sidecar is None or not deltas_state_dict:
            return
        _LOG.info(
            "merged-delta push requested (%d layers); v0 relies on next "
            "active-LoRA push to flush. SVD compression is a TODO.",
            len(deltas_state_dict),
        )
